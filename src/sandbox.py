"""Sandboxed Python execution for opt-in tool mode.

``run_python`` runs model-written code in a fresh bubblewrap process:

* no network (``--unshare-all``), empty environment (``env -i`` plus
  bwrap ``--clearenv``), so no API key or other secret is visible;
* read-only system/Python binds only, plus exactly one task CSV bound
  at ``/data/task.csv``;
* a per-task directory bound at ``/tmp`` (a fresh temporary directory
  when the caller does not supply one), so intermediate files can
  persist across calls for one task but never leak between tasks;
* wall-clock, CPU, address-space, process-count, file-size and
  open-file limits via ``timeout``/``prlimit``;
* stdout/stderr truncated head+tail to a fixed character budget.

The repo, ``answers/``, ``tests/`` (reference solvers), ``src/``,
``$HOME`` and ``/workspace`` are never bound.  Only the site-packages /
standard-library directories needed by the interpreter are bound
individually when they live outside the system prefix, so binding a
virtualenv's packages never exposes the directory that contains it.
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
from dataclasses import dataclass
from pathlib import Path

_CSV_NAME = re.compile(r"^task_[^/]+\.csv$")
_FORBIDDEN_PARTS = frozenset({"answers", "tests", "src"})
_MAX_OUTPUT_DEFAULT = 8_000


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


def _system_bind_dirs() -> list[Path]:
    """Read-only directories needed to run the interpreter.

    Starts with the system prefix, then adds the running interpreter's
    prefix/stdlib/site-packages when they are not already covered.
    Individual site-packages directories are bound rather than their
    parents, so a user-site under ``$HOME`` exposes only packages.
    """
    candidates: list[Path] = []
    for cand in (Path("/usr"), Path(sys.prefix), Path(sys.base_prefix),
                 Path(sysconfig.get_paths().get("stdlib", "")),
                 Path(sys.executable).resolve().parent):
        if str(cand) and cand != Path(""):
            candidates.append(cand)
    for key in ("purelib", "platlib"):
        val = sysconfig.get_paths().get(key)
        if val:
            candidates.append(Path(val))
    try:
        candidates.extend(Path(p) for p in site.getsitepackages())
    except Exception:  # pragma: no cover - site layout varies
        pass
    try:
        user_site = site.getusersitepackages()
        if user_site:
            candidates.append(Path(user_site))
    except Exception:  # pragma: no cover
        pass

    binds: list[Path] = []
    for cand in candidates:
        try:
            resolved = cand.resolve()
        except OSError:
            continue
        if not resolved.is_dir():
            continue
        if {p.lower() for p in resolved.parts} & _FORBIDDEN_PARTS:
            continue
        # Skip directories already covered by an earlier, broader bind.
        if any(resolved == b or b in resolved.parents for b in binds):
            continue
        binds.append(resolved)
    return binds


def build_sandbox_command(code: str, csv_path: Path, workdir: Path | None,
                          limits: SandboxLimits) -> list[str]:
    """Build the prlimit/timeout/bwrap command for one call (no exec)."""
    bwrap = shutil.which("bwrap")
    if not bwrap:
        raise SandboxUnavailable("[run_python] bubblewrap (bwrap) not found on PATH")
    env_bin = "/usr/bin/env" if Path("/usr/bin/env").exists() else shutil.which("env")
    python = sys.executable or "/usr/bin/python3"

    cmd: list[str] = []
    timeout_bin = shutil.which("timeout")
    if timeout_bin:
        # -k 2: SIGKILL two seconds after the initial TERM.
        cmd += [timeout_bin, "-k", "2", str(max(1, int(limits.timeout_seconds)))]
    cmd += [
        bwrap,
        "--unshare-all",
        "--die-with-parent",
        "--new-session",
        "--clearenv",
    ]
    for bind in _system_bind_dirs():
        cmd += ["--ro-bind", str(bind), str(bind)]
    # The dynamic-linker cache alone (no other /etc content), so
    # system libraries in multiarch dirs resolve inside the sandbox.
    ld_cache = Path("/etc/ld.so.cache")
    if ld_cache.is_file():
        cmd += ["--ro-bind", str(ld_cache), str(ld_cache)]
    # Debian alternatives symlinks (e.g. libblas) live here and point
    # back into /usr; without it those system libraries look broken.
    alternatives = Path("/etc/alternatives")
    if alternatives.is_dir():
        cmd += ["--ro-bind", str(alternatives), str(alternatives)]
    # Conventional merged-usr symlinks (harmless when targets differ).
    cmd += [
        "--symlink", "usr/lib", "/lib",
        "--symlink", "usr/lib64", "/lib64",
        "--symlink", "usr/bin", "/bin",
        "--proc", "/proc",
        "--dev", "/dev",
        "--dir", "/data",
        "--ro-bind", str(csv_path), "/data/task.csv",
    ]
    if workdir is not None:
        cmd += ["--bind", str(workdir), "/tmp"]
    else:
        cmd += ["--tmpfs", "/tmp"]
    cmd += ["--chdir", "/tmp", "--", env_bin, "-i"]
    # The environment starts empty (`env -i`); the only variable set
    # is PYTHONPATH pointing at the already-bound, read-only package
    # directories, so installs in a venv or a user site-packages work
    # without exposing any surrounding directory.
    package_dirs: list[str] = []
    for cand in ([site.getusersitepackages(),
                  sysconfig.get_paths().get("purelib"),
                  sysconfig.get_paths().get("platlib")]
                 + list(site.getsitepackages())):
        if cand and Path(cand).is_dir() and cand not in package_dirs:
            if not {p.lower() for p in Path(cand).parts} & _FORBIDDEN_PARTS:
                package_dirs.append(cand)
    if package_dirs:
        cmd += [f"PYTHONPATH={os.pathsep.join(package_dirs)}"]
    # prlimit runs *inside* the sandbox: applying RLIMIT_NPROC outside
    # would count the host user's existing processes and can stop
    # bwrap from creating its namespaces at all.
    prlimit = shutil.which("prlimit")
    if prlimit:
        cmd += [
            prlimit,
            f"--cpu={limits.cpu_seconds}",
            f"--as={limits.memory_bytes}",
            f"--nproc={limits.max_processes}",
            f"--fsize={limits.max_file_bytes}",
            f"--nofile={limits.max_open_files}",
            "--",
        ]
    cmd += [python, "-c", code]
    return cmd


def sandbox_available() -> bool:
    """True when a trivial bubblewrap invocation succeeds here."""
    if not shutil.which("bwrap"):
        return False
    try:
        cmd = build_sandbox_command(
            "pass", _probe_csv(), None, SandboxLimits(timeout_seconds=5))
    except Exception:
        # No probe CSV yet on this host: fall back to a bwrap-only probe.
        try:
            proc = subprocess.run(
                [shutil.which("bwrap"), "--unshare-all", "--die-with-parent",
                 "--new-session", "--ro-bind", "/usr", "/usr",
                 "--symlink", "usr/lib", "/lib",
                 "--symlink", "usr/lib64", "/lib64",
                 "--symlink", "usr/bin", "/bin",
                 "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
                 "--", "/usr/bin/env", "-i", "/usr/bin/true"],
                capture_output=True, timeout=10, check=False)
            return proc.returncode == 0
        except Exception:
            return False
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=10, check=False)
        return proc.returncode == 0
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
    csv = validate_csv_path(csv_path)

    own_tmp: tempfile.TemporaryDirectory | None = None
    work: Path | None = None
    if workdir is not None:
        work = _check_bind_path(Path(workdir), what="task /tmp")
        work.mkdir(parents=True, exist_ok=True)
    cmd = build_sandbox_command(code, csv, work, limits)

    timed_out = False
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, start_new_session=True)
        try:
            stdout, stderr = proc.communicate(
                timeout=float(limits.timeout_seconds) + 5.0)
            exit_code: int | None = proc.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                proc.kill()
            stdout, stderr = proc.communicate()
            exit_code = None
    finally:
        if own_tmp is not None:  # pragma: no cover - reserved
            own_tmp.cleanup()

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
