"""Sandboxed Python execution for opt-in tool mode.

``run_python`` runs model-written code in a fresh bubblewrap process:

* no network (``--unshare-all``) and no host secrets anywhere a
  sandboxed process can look.  The whole host-side chain (``timeout``
  and bubblewrap itself, which is PID 1 inside the PID namespace and
  therefore the process behind ``/proc/1/environ``) is launched with
  a fixed minimal environment (``PATH``/``LANG`` only), and the child
  is started with ``env -i`` plus bwrap ``--clearenv``.  Bubblewrap's
  options travel through an ``--args`` file descriptor instead of
  argv, so ``/proc/1/cmdline`` contains no host paths either;
* read-only system/Python binds only, plus exactly one task CSV bound
  at ``/data/task.csv``.  Interpreter directories that live outside
  the system prefix (a venv, a user site-packages, a toolcache
  prefix) are bound at neutral in-sandbox paths under ``/opt/py`` --
  never at their host paths -- so no ``/home`` or workspace path
  node exists inside the sandbox, and ``PYTHONPATH`` contains only
  in-sandbox paths;
* everything except ``/tmp`` is read-only (``--remount-ro /`` before
  the one writable mount), and ``/tmp`` is a per-task directory (a
  fresh temporary directory when the caller does not supply one), so
  intermediate files can persist across calls for one task but never
  leak between tasks;
* wall-clock, CPU, address-space, process-count, file-size and
  open-file limits via ``timeout``/``prlimit``; stdin is /dev/null;
* stdout/stderr truncated head+tail to a fixed character budget.

The repo, ``answers/``, ``tests/`` (reference solvers), ``src/``,
``$HOME`` and ``/workspace`` are never bound.
"""

from __future__ import annotations

import os
import re
import shutil
import signal
import site
import subprocess
import sys
import sysconfig
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

_CSV_NAME = re.compile(r"^task_[^/]+\.csv$")
_FORBIDDEN_PARTS = frozenset({"answers", "tests", "src"})
_MAX_OUTPUT_DEFAULT = 8_000

# The host-side launcher environment.  Fixed values (never copied from
# the parent environment): whatever launches bubblewrap becomes PID 1
# in the sandbox's PID namespace, and its initial environment stays
# readable at /proc/1/environ even when the child is --clearenv'd.
_LAUNCH_ENV = {"PATH": "/usr/local/bin:/usr/bin:/bin", "LANG": "C.UTF-8"}


class SandboxUnavailable(RuntimeError):
    """Raised when bubblewrap cannot run on this host."""


@dataclass(frozen=True)
class SandboxLimits:
    """Resource limits for one sandboxed call."""

    timeout_seconds: float = 30.0
    cpu_seconds: int = 20
    memory_bytes: int = 2 * 1024**3
    max_processes: int = 64
    max_file_bytes: int = 10 * 1024**2
    max_open_files: int = 64
    max_output_chars: int = _MAX_OUTPUT_DEFAULT


@dataclass
class SandboxCommand:
    """One prepared invocation.

    ``argv`` is the host command (timeout -> bwrap); all bubblewrap
    options (which mention host bind sources) are in ``options`` and
    reach bwrap through an ``--args`` fd, keeping them out of every
    in-sandbox ``/proc/<pid>/cmdline``.
    """

    argv: list[str]
    options: list[str] = field(default_factory=list)


def truncate_output(text: str, max_chars: int = _MAX_OUTPUT_DEFAULT) -> tuple[str, bool]:
    """Truncate ``text`` head+tail, returning ``(text, was_truncated)``."""
    if max_chars <= 0 or len(text) <= max_chars:
        return text, False
    head = max_chars // 2
    tail = max_chars - head
    dropped = len(text) - head - tail
    return (
        f"{text[:head]}\n...[truncated {dropped} chars]...\n{text[-tail:]}",
        True,
    )


def _check_bind_path(path: Path, *, what: str) -> Path:
    """Reject bind sources under a forbidden component."""
    resolved = Path(path).expanduser().resolve()
    parts = {part.lower() for part in resolved.parts}
    bad = parts & _FORBIDDEN_PARTS
    if bad:
        raise ValueError(
            f"[run_python] refusing to bind {what} {resolved}: "
            f"path contains forbidden component(s) {sorted(bad)}")
    return resolved


def validate_csv_path(csv_path: str | Path) -> Path:
    """Validate the single CSV that may be bound at ``/data/task.csv``."""
    raw = Path(csv_path).expanduser()
    if not _CSV_NAME.match(raw.name):
        raise ValueError(
            f"[run_python] csv_path must be one file named task_*.csv, "
            f"got {raw.name!r}")
    resolved = _check_bind_path(raw, what="task CSV")
    if not resolved.is_file():
        raise ValueError(f"[run_python] task CSV is not a regular file: {resolved}")
    return resolved


def _is_under(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


def _covered_dest(binds: list[tuple[Path, Path]], src: Path) -> Path | None:
    """In-sandbox path ``src`` is already bound at, if any."""
    best: tuple[Path, Path] | None = None
    for bound_src, bound_dest in binds:
        if _is_under(src, bound_src):
            if best is None or len(bound_src.parts) > len(best[0].parts):
                best = (bound_src, bound_dest)
    if best is None:
        return None
    bound_src, bound_dest = best
    return bound_dest if src == bound_src else bound_dest / src.relative_to(bound_src)


def _add_bind(binds: list[tuple[Path, Path]], src: Path, dest: Path) -> Path | None:
    """Record a read-only bind of ``src`` at in-sandbox ``dest``.

    Returns the in-sandbox path for ``src`` (an earlier, broader bind
    may already map it there), or ``None`` when the source is unusable.
    """
    try:
        resolved = src.resolve()
    except OSError:
        return None
    if not (resolved.is_dir() or resolved.is_file()):
        return None
    if {p.lower() for p in resolved.parts} & _FORBIDDEN_PARTS:
        return None
    covered = _covered_dest(binds, resolved)
    if covered == dest:
        return dest
    binds.append((resolved, dest))
    return dest


def _python_layout() -> tuple[list[tuple[Path, Path]], str, list[str]]:
    """(binds, in-sandbox python executable, in-sandbox PYTHONPATH).

    Only ``/usr`` is bound at its own path.  Every other interpreter
    directory -- a venv's or toolcache prefix's stdlib/lib, the
    current environment's site-packages, extra site-packages, the
    user site-packages -- is bound under ``/opt/py``, so host layout
    (``$HOME``, a workspace-local venv, ...) never appears as a path
    inside the sandbox.
    """
    binds: list[tuple[Path, Path]] = []
    usr = Path("/usr")
    if usr.is_dir():
        _add_bind(binds, usr, usr)

    paths = sysconfig.get_paths()
    stdlib = Path(paths.get("stdlib") or "/usr/lib/python3").resolve()
    base_prefix = Path(sys.base_prefix).resolve()
    version_dir = stdlib.name or f"python{sys.version_info.major}." \
        f"{sys.version_info.minor}"
    stdlib_dest: Path | None = None
    if not _is_under(stdlib, usr):
        prefix_lib = base_prefix / "lib"
        if prefix_lib.is_dir():
            _add_bind(binds, prefix_lib, Path("/opt/py/lib"))
        stdlib_dest = _add_bind(
            binds, stdlib, Path("/opt/py/lib") / version_dir)
    else:
        stdlib_dest = stdlib

    # Package directories, most specific first: user site (a locally
    # installed pandas must win over system packages), then the
    # running environment's own packages, then system site-packages.
    candidates: list[tuple[str, str | None]] = []
    try:
        candidates.append(("user", site.getusersitepackages()))
    except Exception:  # pragma: no cover - site layout varies
        pass
    candidates.append(("pure", paths.get("purelib")))
    candidates.append(("plat", paths.get("platlib")))
    try:
        candidates.extend(("site", p) for p in site.getsitepackages())
    except Exception:  # pragma: no cover
        pass

    pythonpath: list[str] = []
    extra = 0
    for kind, cand in candidates:
        if not cand:
            continue
        try:
            resolved = Path(cand).resolve()
        except OSError:
            continue
        if not resolved.is_dir():
            continue
        if {p.lower() for p in resolved.parts} & _FORBIDDEN_PARTS:
            continue
        dest = _covered_dest(binds, resolved)
        if dest is None:
            if stdlib_dest is not None and not _is_under(stdlib, usr) \
                    and resolved.parent == stdlib:
                dest_path = Path("/opt/py/lib") / version_dir / "site-packages"
            elif kind == "user":
                dest_path = Path("/opt/py/user-site-packages")
            elif kind in ("pure", "plat"):
                dest_path = Path("/opt/py/site-packages")
            else:
                extra += 1
                dest_path = Path(f"/opt/py/site-packages-{extra}")
            dest = _add_bind(binds, resolved, dest_path)
        if dest is not None and str(dest) not in pythonpath:
            pythonpath.append(str(dest))
    if stdlib_dest is not None and str(stdlib_dest) not in pythonpath:
        pythonpath.append(str(stdlib_dest))

    # The interpreter itself: the resolved (base) executable, bound at
    # a neutral path when it lives outside /usr.  PYTHONPATH above
    # supplies the running environment's packages, so a venv needs no
    # pyvenv.cfg and its directory is never bound.
    try:
        exe = Path(sys.executable).resolve()
    except (OSError, RuntimeError):
        exe = Path("/usr/bin/python3")
    if _is_under(exe, usr):
        python_in_sandbox = str(exe)
    elif exe.is_file():
        dest = _add_bind(binds, exe, Path("/opt/py/bin/python"))
        python_in_sandbox = str(dest) if dest else "/usr/bin/python3"
    else:  # pragma: no cover
        python_in_sandbox = "/usr/bin/python3"
    return binds, python_in_sandbox, pythonpath


def build_sandbox_command(code: str, csv_path: Path, workdir: Path | None,
                          limits: SandboxLimits) -> SandboxCommand:
    """Build the timeout/bwrap invocation for one call (no exec)."""
    bwrap = shutil.which("bwrap")
    if not bwrap:
        raise SandboxUnavailable("[run_python] bubblewrap (bwrap) not found on PATH")
    env_bin = "/usr/bin/env" if Path("/usr/bin/env").exists() else shutil.which("env")
    binds, python, pythonpath = _python_layout()

    options: list[str] = [
        "--unshare-all",
        "--die-with-parent",
        "--new-session",
        "--clearenv",
    ]
    for src, dest in binds:
        options += ["--ro-bind", str(src), str(dest)]
    # The dynamic-linker cache alone (no other /etc content), so
    # system libraries in multiarch dirs resolve inside the sandbox.
    ld_cache = Path("/etc/ld.so.cache")
    if ld_cache.is_file():
        options += ["--ro-bind", str(ld_cache), str(ld_cache)]
    # Debian alternatives symlinks (e.g. libblas) live here and point
    # back into /usr; without it those system libraries look broken.
    alternatives = Path("/etc/alternatives")
    if alternatives.is_dir():
        options += ["--ro-bind", str(alternatives), str(alternatives)]
    # Conventional merged-usr symlinks (harmless when targets differ).
    options += [
        "--symlink", "usr/lib", "/lib",
        "--symlink", "usr/lib64", "/lib64",
        "--symlink", "usr/bin", "/bin",
        "--proc", "/proc",
        "--dev", "/dev",
        "--dir", "/data",
        # One task CSV at /data/task.csv (existing tiers), or, for the
        # multi-file tools-frontier tier, one task directory read-only
        # at /data/ itself (spec.md plus its data files, nothing else).
        *(["--ro-bind", str(csv_path), "/data"] if Path(csv_path).is_dir()
          else ["--ro-bind", str(csv_path), "/data/task.csv"]),
        "--dir", "/tmp",
        # Everything mounted so far (and the root itself) goes
        # read-only; only the /tmp mount below stays writable.
        "--remount-ro", "/",
    ]
    if workdir is not None:
        options += ["--bind", str(workdir), "/tmp"]
    else:
        options += ["--tmpfs", "/tmp"]
    options += ["--chdir", "/tmp"]

    argv: list[str] = []
    timeout_bin = shutil.which("timeout")
    if timeout_bin:
        # -k 2: SIGKILL two seconds after the initial TERM.
        argv += [timeout_bin, "-k", "2",
                 str(max(1, int(limits.timeout_seconds)))]
    # The fd number is filled in at spawn time (see _spawn).
    argv += [bwrap, "--args", "{ARGS_FD}", "--", env_bin, "-i"]
    # The environment starts empty (`env -i`); the only variable set
    # is PYTHONPATH, and it names only in-sandbox (/opt/py, /usr)
    # directories bound read-only above.
    if pythonpath:
        argv += [f"PYTHONPATH={os.pathsep.join(pythonpath)}"]
    # prlimit runs *inside* the sandbox: applying RLIMIT_NPROC outside
    # would count the host user's existing processes and can stop
    # bwrap from creating its namespaces at all.
    prlimit = "/usr/bin/prlimit" if Path("/usr/bin/prlimit").exists() \
        else shutil.which("prlimit")
    if prlimit and _is_under(Path(prlimit), Path("/usr")):
        argv += [
            prlimit,
            f"--cpu={limits.cpu_seconds}",
            f"--as={limits.memory_bytes}",
            f"--nproc={limits.max_processes}",
            f"--fsize={limits.max_file_bytes}",
            f"--nofile={limits.max_open_files}",
            "--",
        ]
    argv += [python, "-c", code]
    return SandboxCommand(argv=argv, options=options)


def _spawn(command: SandboxCommand, timeout_seconds: float):
    """Spawn a prepared command and communicate with it.

    Bubblewrap's options are written to a pipe whose read end is the
    ``--args`` fd (inheritable through ``timeout``), and the entire
    chain is launched with the fixed ``_LAUNCH_ENV`` and stdin from
    /dev/null, so no host environment value, argument or open file
    can be observed from inside via /proc.
    """
    read_fd, write_fd = os.pipe()
    try:
        payload = ("\0".join(command.options) + "\0").encode()
        # Options are a few KB at most: far under the pipe buffer.
        os.write(write_fd, payload)
    finally:
        os.close(write_fd)
    argv = [str(read_fd) if part == "{ARGS_FD}" else part
            for part in command.argv]
    try:
        proc = subprocess.Popen(
            argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL, text=True, env=dict(_LAUNCH_ENV),
            pass_fds=(read_fd,), start_new_session=True)
    finally:
        os.close(read_fd)
    timed_out = False
    try:
        stdout, stderr = proc.communicate(timeout=timeout_seconds)
        exit_code: int | None = proc.returncode
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            proc.kill()
        stdout, stderr = proc.communicate()
        exit_code = None
    return stdout, stderr, exit_code, timed_out


def sandbox_available() -> bool:
    """True when a trivial bubblewrap invocation succeeds here."""
    if not shutil.which("bwrap"):
        return False
    try:
        cmd = build_sandbox_command(
            "pass", _probe_csv(), None, SandboxLimits(timeout_seconds=5))
        _stdout, _stderr, exit_code, _timed = _spawn(cmd, timeout_seconds=10)
        return exit_code == 0
    except Exception:
        return False


_PROBE_CSV: Path | None = None


def _probe_csv() -> Path:
    global _PROBE_CSV
    if _PROBE_CSV is None or not _PROBE_CSV.exists():
        fd, name = tempfile.mkstemp(prefix="task_probe_", suffix=".csv")
        # mkstemp name is task_probe_*.csv, which matches task_*.csv.
        with os.fdopen(fd, "w") as fh:
            fh.write("date,close\n2024-01-01,1\n")
        _PROBE_CSV = Path(name)
    return _PROBE_CSV


def run_python(code: str, csv_path: str | Path,
               workdir: str | Path | None = None,
               limits: SandboxLimits | None = None,
               timeout_seconds: float | None = None) -> dict:
    """Run ``code`` sandboxed with only ``csv_path`` visible at /data/task.csv.

    Returns a dict with ``stdout``, ``stderr``, ``exit`` (``None`` when
    killed), ``timed_out`` and ``truncated``.  Raises
    :class:`SandboxUnavailable` when bwrap is missing/broken and
    ``ValueError`` for a forbidden or malformed CSV path.
    """
    if not isinstance(code, str) or not code.strip():
        raise ValueError("[run_python] code must be a non-empty string")
    limits = limits or SandboxLimits()
    if timeout_seconds is not None:
        limits = SandboxLimits(
            timeout_seconds=float(timeout_seconds),
            cpu_seconds=limits.cpu_seconds,
            memory_bytes=limits.memory_bytes,
            max_processes=limits.max_processes,
            max_file_bytes=limits.max_file_bytes,
            max_open_files=limits.max_open_files,
            max_output_chars=limits.max_output_chars,
        )
    if Path(csv_path).expanduser().is_dir():
        # Multi-file task directory: bind the directory itself at /data.
        # It must be a real directory and must not live under a forbidden
        # component (answers/tests/src), exactly like the single-CSV case.
        csv = _check_bind_path(Path(csv_path), what="task data directory")
    else:
        csv = validate_csv_path(csv_path)

    work: Path | None = None
    if workdir is not None:
        work = _check_bind_path(Path(workdir), what="task /tmp")
        work.mkdir(parents=True, exist_ok=True)
    cmd = build_sandbox_command(code, csv, work, limits)

    stdout, stderr, exit_code, timed_out = _spawn(
        cmd, timeout_seconds=float(limits.timeout_seconds) + 5.0)

    # `timeout` reports 124 (or 137 after -k) for a wall-clock kill.
    if exit_code in (124, 137) and not timed_out:
        # Only treat as timeout when the process ran ~the full budget;
        # a program exiting 124 itself is indistinguishable, and the
        # timeout wrapper is the overwhelmingly likely source.
        timed_out = True
        exit_code = None
    stdout, trunc_out = truncate_output(stdout or "", limits.max_output_chars)
    stderr, trunc_err = truncate_output(stderr or "", limits.max_output_chars)
    return {
        "stdout": stdout,
        "stderr": stderr,
        "exit": exit_code,
        "timed_out": timed_out,
        "truncated": bool(trunc_out or trunc_err),
    }
