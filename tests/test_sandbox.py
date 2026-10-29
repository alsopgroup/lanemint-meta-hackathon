"""Sandbox tests: skipped entirely when bubblewrap cannot run here."""

import os
from pathlib import Path

import pytest

from src.sandbox import SandboxLimits, run_python, sandbox_available

pytestmark = pytest.mark.skipif(
    not sandbox_available(), reason="bubblewrap user namespaces unavailable")


@pytest.fixture()
def csv_path(tmp_path):
    path = tmp_path / "task_001.csv"
    path.write_text("date,close\n2024-01-01,10\n2024-01-02,11\n")
    return path


def test_runs_code_and_reads_only_task_csv(csv_path):
    result = run_python(
        "import pandas as pd\n"
        "df = pd.read_csv('/data/task.csv')\n"
        "print(df['close'].sum())\n"
        "print(sorted(__import__('os').listdir('/data')))",
        csv_path)
    assert result["exit"] == 0, result["stderr"]
    assert "21" in result["stdout"]
    assert "['task.csv']" in result["stdout"]


def test_network_blocked(csv_path):
    result = run_python(
        "import socket\n"
        "try:\n"
        "    socket.create_connection(('1.1.1.1', 53), timeout=3)\n"
        "    print('NETWORK_OPEN')\n"
        "except OSError as e:\n"
        "    print('NETWORK_BLOCKED', type(e).__name__)",
        csv_path)
    assert result["exit"] == 0, result["stderr"]
    assert "NETWORK_BLOCKED" in result["stdout"]
    assert "NETWORK_OPEN" not in result["stdout"]


def test_sentinel_env_var_not_visible(csv_path, monkeypatch):
    monkeypatch.setenv("FIREWORKS_API_KEY", "sentinel-value-do-not-leak")
    monkeypatch.setenv("META_API_KEY", "sentinel-value-do-not-leak")
    result = run_python(
        "import os\n"
        "print(sorted(os.environ.keys()))\n"
        "print(os.environ.get('FIREWORKS_API_KEY'))\n"
        "print(os.environ.get('META_API_KEY'))",
        csv_path)
    assert result["exit"] == 0, result["stderr"]
    assert "sentinel-value-do-not-leak" not in result["stdout"]
    assert "sentinel-value-do-not-leak" not in result["stderr"]
    assert "FIREWORKS_API_KEY" not in result["stdout"]


def test_answers_tests_src_and_repo_not_reachable(csv_path, tmp_path):
    workdir = tmp_path / "work"
    code = """
import os
paths = [
    '/data/../answers', '/data/../tests', '/data/../src',
    '/answers', '/tests', '/src', '/workspace', '/workspace/src/scorer.py',
    '/proc/self/root/workspace', '/proc/self/root/src',
    '/proc/self/root/tests', os.path.expanduser('~') + '/.ssh',
]
for p in paths:
    print(p, os.path.exists(p))
# A symlink planted in /tmp must not lead out of the sandbox either.
os.symlink('/etc/passwd', '/tmp/escape')
try:
    open('/tmp/escape').read()
    print('SYMLINK_ESCAPE_OPEN')
except OSError:
    print('SYMLINK_BLOCKED')
try:
    open('/data/task.csv/../../etc/passwd').read()
    print('TRAVERSAL_OPEN')
except OSError:
    print('TRAVERSAL_BLOCKED')
print(sorted(os.listdir('/')))
"""
    result = run_python(code, csv_path, workdir)
    assert result["exit"] == 0, result["stderr"]
    for line in result["stdout"].splitlines():
        if " True" in line:
            raise AssertionError(f"path visible inside sandbox: {line}")
    assert "SYMLINK_BLOCKED" in result["stdout"]
    assert "TRAVERSAL_BLOCKED" in result["stdout"]


def test_timeout_kills_infinite_loop(csv_path):
    result = run_python("while True:\n    pass", csv_path,
                        limits=SandboxLimits(timeout_seconds=2))
    assert result["timed_out"] is True
    assert result["exit"] is None


def test_memory_limit_kills_large_allocation(csv_path):
    limits = SandboxLimits(timeout_seconds=20, memory_bytes=256 * 1024**2)
    result = run_python(
        "x = bytearray(4 * 1024**3)\nprint('ALLOCATED', len(x))",
        csv_path, limits=limits)
    assert "ALLOCATED" not in result["stdout"]
    assert result["exit"] not in (0, None) or result["timed_out"]


def test_fork_bomb_contained(csv_path):
    limits = SandboxLimits(timeout_seconds=15, max_processes=16)
    code = """
import os
pids = []
try:
    for _ in range(200):
        pid = os.fork()
        if pid == 0:
            os._exit(0)
        pids.append(pid)
except OSError as e:
    print('FORK_LIMITED', type(e).__name__)
for pid in pids:
    try:
        os.waitpid(pid, 0)
    except OSError:
        pass
print('SURVIVED')
"""
    result = run_python(code, csv_path, limits=limits)
    assert "SURVIVED" in result["stdout"]


def test_output_truncation_head_and_tail(csv_path):
    result = run_python(
        "print('A' * 10000 + 'MIDDLE' + 'B' * 10000, end='')",
        csv_path, limits=SandboxLimits(max_output_chars=1000))
    assert result["truncated"] is True
    assert "[truncated" in result["stdout"]
    assert len(result["stdout"]) <= 1100
    assert result["stdout"].startswith("A")
    assert result["stdout"].rstrip().endswith("B")


def test_tmp_persists_within_workdir_but_not_data(csv_path, tmp_path):
    workdir = tmp_path / "work"
    first = run_python("open('/tmp/note.txt', 'w').write('hello')",
                       csv_path, workdir)
    second = run_python("print(open('/tmp/note.txt').read())",
                        csv_path, workdir)
    assert first["exit"] == 0 and second["exit"] == 0
    assert "hello" in second["stdout"]


def test_rejects_non_task_csv_and_forbidden_paths(tmp_path):
    bad_name = tmp_path / "data.csv"
    bad_name.write_text("a\n1\n")
    with pytest.raises(ValueError, match="task_"):
        run_python("print(1)", bad_name)

    answers = tmp_path / "answers"
    answers.mkdir()
    planted = answers / "task_001.csv"
    planted.write_text("a\n1\n")
    with pytest.raises(ValueError, match="forbidden"):
        run_python("print(1)", planted)

    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    planted2 = tests_dir / "task_001.csv"
    planted2.write_text("a\n1\n")
    with pytest.raises(ValueError, match="forbidden"):
        run_python("print(1)", planted2)


def test_empty_env_inside(csv_path):
    result = run_python(
        "import os\nprint([k for k in os.environ if 'KEY' in k or 'TOKEN' in k])",
        csv_path)
    assert result["exit"] == 0, result["stderr"]
    assert "[]" in result["stdout"]


def test_proc_environ_and_cmdline_carry_no_host_secrets(csv_path, monkeypatch):
    """Regression: PID 1 inside is bwrap, launched with the host env.

    ``--clearenv``/``env -i`` clean the child, but /proc/1/environ used
    to expose every parent variable.  The launcher now starts with a
    fixed PATH/LANG-only environment.  The probe needle is assembled
    from fragments so the needle itself never appears in the probed
    process's own cmdline.
    """
    monkeypatch.setenv("META_API_KEY", "sentinel-value-do-not-leak")
    monkeypatch.setenv("FIREWORKS_API_KEY", "sentinel-value-do-not-leak")
    monkeypatch.setenv("SOME_OTHER_TOKEN", "sentinel-value-do-not-leak")
    code = """
import glob, os
needle = ("sentinel-" "value-do-" "not-leak").encode()
hits, keys = [], set()
for pid_dir in glob.glob('/proc/[0-9]*'):
    for name in ('environ', 'cmdline'):
        try:
            blob = open(os.path.join(pid_dir, name), 'rb').read()
        except OSError:
            continue
        if needle in blob:
            hits.append(pid_dir + '/' + name)
        if name == 'environ':
            keys.update(part.split(b'=', 1)[0].decode('utf-8', 'replace')
                        for part in blob.split(b'\\0') if b'=' in part)
    for fd in glob.glob(pid_dir + '/fd/*'):
        try:
            target = os.readlink(fd)
        except OSError:
            continue
        if needle.decode() in target:
            hits.append(fd + '->' + target)
print('HITS', hits)
print('KEYS', sorted(keys))
print('DONE')
"""
    result = run_python(code, csv_path)
    assert result["exit"] == 0, result["stderr"]
    assert "HITS []" in result["stdout"]
    keys_line = next(ln for ln in result["stdout"].splitlines()
                     if ln.startswith("KEYS"))
    for forbidden in ("META_API_KEY", "FIREWORKS_API_KEY", "SOME_OTHER_TOKEN",
                      "sentinel"):
        assert forbidden not in keys_line
    assert "sentinel-value" not in result["stdout"]
    # Only the fixed launcher vars and the child's own PYTHONPATH may
    # appear in any /proc/<pid>/environ.
    allowed = {"PATH", "LANG", "LC_CTYPE", "PYTHONPATH", "PWD"}
    found = set(keys_line.removeprefix("KEYS ").strip("[]").replace("'", "")
                .split(", ")) - {""}
    assert found <= allowed, found


def test_proc1_cmdline_has_no_host_paths(csv_path):
    code = """
blob = open('/proc/1/cmdline', 'rb').read().decode('utf-8', 'replace')
# PID 1's argv ends with the python -c argument (the probe's own
# source, which the model obviously knows); only the launcher part
# before it could leak host paths or bind sources.
launcher = blob.split('\\0-c\\0', 1)[0]
print('LAUNCHER', launcher.replace('\\0', '|'))
print('HAS_HOME_BIND', '/home/' in launcher)
print('HAS_RO_BIND', '--ro-bind' in launcher)
print('HAS_ARGS_FD', '--args' in launcher)
print('DONE')
"""
    result = run_python(code, csv_path)
    assert result["exit"] == 0, result["stderr"]
    assert "HAS_HOME_BIND False" in result["stdout"]
    assert "HAS_RO_BIND False" in result["stdout"]
    assert "HAS_ARGS_FD True" in result["stdout"]


def test_pythonpath_and_paths_are_sandbox_local(csv_path):
    code = """
import os, sys
print('PYTHONPATH', os.environ.get('PYTHONPATH', ''))
print('ROOT', sorted(os.listdir('/')))
print('HAS_HOME', os.path.exists('/home'))
print('HAS_WORKSPACE', os.path.exists('/workspace'))
print('DONE')
"""
    result = run_python(code, csv_path)
    assert result["exit"] == 0, result["stderr"]
    pp_line = next(ln for ln in result["stdout"].splitlines()
                   if ln.startswith("PYTHONPATH"))
    for entry in pp_line.removeprefix("PYTHONPATH ").split(":"):
        if entry:
            assert entry.startswith(("/opt/", "/usr/")), entry
    assert "/home/" not in pp_line and "workspace" not in pp_line
    assert "HAS_HOME False" in result["stdout"]
    assert "HAS_WORKSPACE False" in result["stdout"]


def test_workspace_local_venv_site_packages_bound_neutrally(csv_path, tmp_path,
                                                            monkeypatch):
    """A venv/user-site under a workspace-like tree must still work,
    but only via a neutral /opt/py bind: no workspace/$HOME node, and
    the host path must not appear in PYTHONPATH or sys.path output
    beyond the marker the probe itself prints."""
    import site as site_module

    fake_site = (tmp_path / "workspace" / ".venv" / "lib"
                 / "python3.12" / "site-packages")
    fake_site.mkdir(parents=True)
    (fake_site / "toolmode_probe_pkg.py").write_text("MARKER = 'probe-ok'\n")
    monkeypatch.setattr(site_module, "getusersitepackages",
                        lambda: str(fake_site))
    code = """
import os, sys
import toolmode_probe_pkg
print('MARKER', toolmode_probe_pkg.MARKER)
print('PKG_FILE', toolmode_probe_pkg.__file__)
print('HAS_WORKSPACE', os.path.exists('/workspace'))
print('HAS_HOME', os.path.exists('/home'))
print('DONE')
"""
    result = run_python(code, csv_path, tmp_path / "work")
    assert result["exit"] == 0, result["stderr"]
    assert "MARKER probe-ok" in result["stdout"]
    pkg_line = next(ln for ln in result["stdout"].splitlines()
                    if ln.startswith("PKG_FILE"))
    assert pkg_line.startswith("PKG_FILE /opt/py/")
    assert "workspace" not in pkg_line and "/home" not in pkg_line
    assert "HAS_WORKSPACE False" in result["stdout"]
    assert "HAS_HOME False" in result["stdout"]


def test_everything_but_tmp_is_read_only(csv_path, tmp_path):
    code = """
targets = ['/etc/toolmode_probe', '/home/toolmode_probe',
           '/opt/toolmode_probe', '/usr/toolmode_probe',
           '/data/toolmode_probe', '/toolmode_probe']
for path in targets:
    try:
        open(path, 'w').write('x')
        print(path, 'WRITE_OK')
    except OSError:
        print(path, 'BLOCKED')
try:
    open('/tmp/toolmode_probe', 'w').write('x')
    print('/tmp/toolmode_probe WRITE_OK')
except OSError:
    print('/tmp/toolmode_probe BLOCKED')
try:
    import os
    os.mkdir('/etc/toolmode_dir')
    print('MKDIR_ETC WRITE_OK')
except OSError:
    print('MKDIR_ETC BLOCKED')
print('DONE')
"""
    result = run_python(code, csv_path, tmp_path / "work")
    assert result["exit"] == 0, result["stderr"]
    for line in result["stdout"].splitlines():
        if " WRITE_OK" in line:
            assert line.startswith("/tmp/"), line
    assert "MKDIR_ETC BLOCKED" in result["stdout"]
    assert "/tmp/toolmode_probe WRITE_OK" in result["stdout"]


def test_stdin_is_dev_null(csv_path, tmp_path):
    result = run_python(
        "import os, sys\n"
        "print('STDIN', os.readlink('/proc/self/fd/0'))\n"
        "print('READ', repr(sys.stdin.read()))",
        csv_path, tmp_path / "work")
    assert result["exit"] == 0, result["stderr"]
    assert "/dev/null" in result["stdout"]


def test_parent_cwd_files_not_visible(csv_path, tmp_path, monkeypatch):
    # Even when the caller's cwd is a repo-like tree, none of it is bound.
    secret = tmp_path / "answers_secret.txt"
    secret.write_text("nope")
    monkeypatch.chdir(tmp_path)
    result = run_python(
        "import os\n"
        "print(os.path.exists('/tmp/answers_secret.txt'))\n"
        "print(os.path.exists('answers_secret.txt'))",
        csv_path, tmp_path / "work")
    assert result["exit"] == 0, result["stderr"]
    assert result["stdout"].count("False") == 2
    assert Path(os.getcwd())  # keep os import used on host side too
