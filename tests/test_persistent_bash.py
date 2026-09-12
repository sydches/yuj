"""PersistentBashSession regression tests.

The persistent path replaces per-call ``subprocess.run(bwrap + bash + cmd)``
with one long-lived bwrap+bash subprocess that streams commands via
stdin and detects per-command boundaries with a marker. Sandbox
semantics MUST match the per-call path bytewise — these tests pin the
critical properties so a refactor can't silently regress isolation.

Skipped when bwrap is missing or cannot create the required namespaces.
"""
from __future__ import annotations

import os
import threading
import tempfile
import time
from pathlib import Path

import pytest

from scripts.llm_solver.harness.sandbox import (
    PersistentBashSession,
    bwrap_preflight,
    get_persistent_runner,
    set_persistent_runner,
)


BWRAP = "/usr/bin/bwrap"

@pytest.fixture(scope="module", autouse=True)
def require_bwrap():
    ok, reason = bwrap_preflight(BWRAP)
    if not ok:
        pytest.skip(reason)


@pytest.fixture
def runner(tmp_path: Path):
    p = PersistentBashSession(cwd=str(tmp_path))
    p.start()
    yield p
    p.close()


# ── Basic semantics ────────────────────────────────────────────────────


def test_basic_exec(runner, tmp_path):
    out, ec, to = runner.run("echo hello", cwd=str(tmp_path), timeout=10)
    assert ec == 0 and out == "hello\n" and not to


def test_exit_code_propagation(runner, tmp_path):
    out, ec, to = runner.run("false", cwd=str(tmp_path), timeout=10)
    assert ec == 1 and not to


def test_multiline_output(runner, tmp_path):
    out, ec, to = runner.run("printf 'a\\nb\\nc\\n'", cwd=str(tmp_path), timeout=10)
    assert ec == 0 and out == "a\nb\nc\n"


@pytest.mark.parametrize('ending', ['', r'\n', r'\n\n', r'\r', r'\r\n'])
def test_persistent_preserves_partial_lines_and_trailing_newlines(runner, tmp_path, ending):
    from scripts.llm_solver.harness._tools._run_in_sandbox import _run_in_sandbox
    command = "printf 'payload" + ending + "'"
    fresh = _run_in_sandbox(command, cwd=str(tmp_path), timeout=10, sandbox=True,
                            bwrap_bin=BWRAP, use_persistent=False)
    retained = runner.run(command, cwd=str(tmp_path), timeout=10)
    assert retained == fresh


def test_stderr_merged(runner, tmp_path):
    out, ec, to = runner.run(
        "echo OUT; echo ERR 1>&2", cwd=str(tmp_path), timeout=10,
    )
    assert ec == 0
    assert "OUT" in out and "ERR" in out


def test_pipefail_active(runner, tmp_path):
    """`set -o pipefail` is set once at session start and inherited by
    subshells — pipe-stage failures must propagate."""
    out, ec, to = runner.run(
        "false | true", cwd=str(tmp_path), timeout=10,
    )
    assert ec != 0, f"pipefail not honored: ec={ec}"


def test_binary_transport_preserves_streams_and_exit_status(runner, tmp_path):
    payload = b'\x00\xff\r\nno final newline'
    result = runner.run_binary(
        "cat; printf '\\377\\000\\r\\n' >&2; exit 7", cwd=str(tmp_path),
        timeout=10, input_bytes=payload,
    )
    assert result.returncode == 7
    assert result.stdout == payload
    assert result.stderr == b'\xff\x00\r\n'


def test_binary_file_runner_uses_the_existing_mount_namespace(runner, tmp_path):
    from scripts.llm_solver.harness._tools._run_in_sandbox import _run_in_sandbox
    marker = '/tmp/persistent-only-marker'
    out, code, timed_out = runner.run(
        f"printf 'native namespace' > {marker}", cwd=str(tmp_path), timeout=10,
    )
    assert code == 0 and not timed_out
    pid = runner._proc.pid
    set_persistent_runner(runner)
    try:
        result = _run_in_sandbox(
            f'cat {marker}', cwd=str(tmp_path), timeout=10, sandbox=True,
            bwrap_bin=BWRAP, raw_result=True,
        )
        assert result.stdout == b'native namespace'
        assert result.returncode == 0
        assert runner._proc.pid == pid
    finally:
        set_persistent_runner(None)


@pytest.mark.parametrize('size', [16384, 196608])
def test_large_binary_file_input_preserves_retained_namespace(runner, tmp_path, size):
    from scripts.llm_solver.harness._tools._run_in_sandbox import _run_in_sandbox
    payload = bytes(range(256)) * (size // 256)
    assert runner.run('printf keep > /tmp/binary-private-state',
                      cwd=str(tmp_path), timeout=10)[1] == 0
    pid = runner._proc.pid
    set_persistent_runner(runner)
    try:
        result = _run_in_sandbox(
            'cat > large.bin; cat large.bin; cat /tmp/binary-private-state >&2',
            cwd=str(tmp_path), timeout=10, sandbox=True, bwrap_bin=BWRAP,
            raw_result=True, input_bytes=payload,
        )
        assert result.returncode == 0
        assert result.stdout == payload
        assert result.stderr == b'keep'
        assert (tmp_path / 'large.bin').read_bytes() == payload
        assert runner._proc.pid == pid
    finally:
        set_persistent_runner(None)


@pytest.fixture
def private_home_runner(tmp_path):
    environment = {'PATH': '/usr/bin:/bin', 'HOME': str(tmp_path.parent / 'private-home')}
    p = PersistentBashSession(cwd=str(tmp_path), effective_env=environment)
    p.start()
    yield p
    p.close()


def test_persistent_runner_rebinds_a_changed_mask(private_home_runner, tmp_path):
    runner = private_home_runner
    from scripts.llm_solver.harness._tools._run_in_sandbox import _run_in_sandbox
    secret = tmp_path / 'secret'
    secret.write_bytes(b'previously visible')
    assert runner.run('printf keep > "$HOME/private-state"', cwd=str(tmp_path), timeout=10)[1] == 0
    pid = runner._proc.pid
    set_persistent_runner(runner)
    try:
        result = _run_in_sandbox(
            'cat secret', cwd=str(tmp_path), timeout=10, sandbox=True,
            bwrap_bin=BWRAP, raw_result=True, unreadable_paths=(str(secret),),
            effective_env=runner.effective_env,
        )
        assert result.returncode != 0
        assert b'previously visible' not in result.stdout
        assert runner._proc.pid == pid
        text, code, timed_out = _run_in_sandbox(
            'cat secret', cwd=str(tmp_path), timeout=10, sandbox=True,
            bwrap_bin=BWRAP, unreadable_paths=(str(secret),), effective_env=runner.effective_env,
        )
        assert code != 0 and not timed_out
        assert 'previously visible' not in text
        state = _run_in_sandbox(
            'cat "$HOME/private-state"', cwd=str(tmp_path), timeout=10, sandbox=True,
            bwrap_bin=BWRAP, raw_result=True, unreadable_paths=(str(secret),),
            effective_env=runner.effective_env,
        )
        assert state.stdout == b'keep'
    finally:
        set_persistent_runner(None)
    assert secret.read_bytes() == b'previously visible'


def test_persistent_runner_refreshes_an_optional_mask_when_file_appears(runner, tmp_path):
    from scripts.llm_solver.harness._tools._run_in_sandbox import _run_in_sandbox
    secret = tmp_path / 'late-secret'
    options = dict(cwd=str(tmp_path), timeout=10, sandbox=True, bwrap_bin=BWRAP,
                   unreadable_paths=(f'optional:{secret}',), raw_result=True)
    set_persistent_runner(runner)
    try:
        assert _run_in_sandbox('printf ready', **options).stdout == b'ready'
        pid = runner._proc.pid
        secret.write_bytes(b'late host secret')
        result = _run_in_sandbox('cat late-secret', **options)
        assert result.returncode != 0
        assert b'late host secret' not in result.stdout
        assert runner._proc.pid == pid
    finally:
        set_persistent_runner(None)


def test_mask_update_preserves_a_newly_visible_descendant_and_private_state(private_home_runner, tmp_path):
    runner = private_home_runner
    from scripts.llm_solver.harness._tools._run_in_sandbox import _run_in_sandbox
    directory = tmp_path / 'private'
    directory.mkdir()
    (directory / 'secret').write_text('secret host content')
    (directory / 'keep').write_text('permitted descendant')
    assert runner.run('printf preserved > "$HOME/state"', cwd=str(tmp_path), timeout=10)[1] == 0
    pid = runner._proc.pid
    options = dict(cwd=str(tmp_path), timeout=10, sandbox=True,
                   bwrap_bin=BWRAP, raw_result=True, effective_env=runner.effective_env)
    set_persistent_runner(runner)
    try:
        hidden = _run_in_sandbox('ls private', **options, unreadable_paths=(str(directory),))
        assert hidden.stdout == b''
        visible = _run_in_sandbox('cat private/keep "$HOME/state"', **options,
                                  unreadable_paths=(str(directory / 'secret'),))
        assert visible.returncode == 0
        assert visible.stdout == b'permitted descendantpreserved'
        hidden = _run_in_sandbox('cat private/secret', **options,
                                 unreadable_paths=(str(directory / 'secret'),))
        assert hidden.returncode != 0
        assert b'secret host content' not in hidden.stdout
        assert runner._proc.pid == pid
    finally:
        set_persistent_runner(None)
    assert (directory / 'secret').read_text() == 'secret host content'


def test_absent_native_mask_target_is_retried_when_it_appears(runner, tmp_path):
    import shlex
    from scripts.llm_solver.harness._tools._run_in_sandbox import _run_in_sandbox
    outside = tmp_path.parent / ('late-artifact-' + tmp_path.name)
    outside.mkdir()
    (outside / 'host-secret').write_text('host-only content')
    options = dict(cwd=str(tmp_path), timeout=10, sandbox=True, bwrap_bin=BWRAP,
                   unreadable_paths=(str(outside),), raw_result=True)
    set_persistent_runner(runner)
    try:
        pid = runner._proc.pid
        assert _run_in_sandbox('printf ready', **options).stdout == b'ready'
        assert str(outside) not in runner._namespace.mounts
        command = 'mkdir -p -- ' + shlex.quote(str(outside)) + '; printf native > ' + shlex.quote(str(outside / 'file'))
        assert _run_in_sandbox(command, **options).returncode == 0
        result = _run_in_sandbox('cat -- ' + shlex.quote(str(outside / 'file')), **options)
        assert result.returncode != 0 and b'native' not in result.stdout
        assert str(outside) in runner._namespace.mounts
        assert runner._proc.pid == pid
    finally:
        set_persistent_runner(None)
    assert (outside / 'host-secret').read_text() == 'host-only content'


def test_mask_controller_refuses_to_remove_a_different_kernel_mount(runner, tmp_path):
    runner._namespace.mounts['/usr'] = {'kind': 'directory', 'id': -1}
    with pytest.raises(RuntimeError, match='unowned mount'):
        runner._namespace.update({}, timeout=10)
    assert runner.run('test -x /usr/bin/cat', cwd=str(tmp_path), timeout=10)[1] == 0


def test_read_workers_inherit_and_serialize_the_selected_namespace(runner, tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from contextvars import copy_context
    from scripts.llm_solver.harness._tools._run_in_sandbox import _run_in_sandbox
    out, code, timed_out = runner.run(
        'printf shared > /tmp/worker-namespace-marker', cwd=str(tmp_path), timeout=10,
    )
    assert code == 0 and not timed_out
    set_persistent_runner(runner)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(
                copy_context().run, _run_in_sandbox,
                'cat /tmp/worker-namespace-marker', cwd=str(tmp_path), timeout=10,
                sandbox=True, bwrap_bin=BWRAP, raw_result=True,
            ) for _ in range(2)]
            for future in futures:
                result = future.result()
                assert result.returncode == 0
                assert result.stdout == b'shared'
    finally:
        set_persistent_runner(None)


def test_binary_transport_timeout_does_not_return_partial_file(runner, tmp_path):
    import subprocess
    with pytest.raises(subprocess.TimeoutExpired):
        runner.run_binary('printf partial; sleep 10', cwd=str(tmp_path), timeout=0.05)
    result = runner.run_binary('printf recovered', cwd=str(tmp_path), timeout=10)
    assert result.stdout == b'recovered'


def test_waiting_for_the_shared_pipe_consumes_the_call_allowance(runner, tmp_path):
    from scripts.llm_solver.harness.time_budget import BudgetExhausted
    from concurrent.futures import ThreadPoolExecutor
    pool = ThreadPoolExecutor(max_workers=1)
    runner._lock.acquire()
    try:
        waiting = pool.submit(runner.run, 'printf should-not-run',
                              cwd=str(tmp_path), timeout=0.01)
        with pytest.raises(BudgetExhausted, match='before submission'):
            waiting.result(timeout=1)
    finally:
        runner._lock.release()
        pool.shutdown(wait=True)


def test_startup_wait_uses_the_declared_command_allowance(tmp_path, monkeypatch):
    from scripts.llm_solver.harness import sandbox
    from scripts.llm_solver.harness.time_budget import BudgetExhausted
    monkeypatch.setattr(sandbox, '_build_bwrap_argv',
                        lambda *args, **kwargs: ['/usr/bin/sleep', '10'])
    # A real child with no startup response must be stopped within the same
    # command allowance. The wrapper ignores the bwrap-only information args.
    import subprocess
    real_popen = subprocess.Popen
    monkeypatch.setattr(subprocess, 'Popen',
                        lambda argv, **kwargs: real_popen(['/usr/bin/sleep', '10'], **kwargs))
    instance = PersistentBashSession(cwd=str(tmp_path))
    try:
        with pytest.raises(BudgetExhausted, match='before submission'):
            instance.run('printf never', cwd=str(tmp_path), timeout=0.02)
        assert instance._proc is None
    finally:
        instance.close()


# ── Sandbox semantics — bytewise parity with per-call ─────────────────


def test_cwd_writable(runner, tmp_path):
    out, ec, to = runner.run(
        "echo data > inside.txt && cat inside.txt",
        cwd=str(tmp_path), timeout=10,
    )
    assert ec == 0 and "data" in out
    assert (tmp_path / "inside.txt").is_file()


def test_outside_cwd_readonly(runner, tmp_path):
    out, ec, to = runner.run(
        "touch /usr/local/lib/yuj_persistent_escape 2>&1",
        cwd=str(tmp_path), timeout=10,
    )
    assert ec != 0
    assert "Read-only" in out or "read-only" in out.lower()
    assert not Path("/usr/local/lib/yuj_persistent_escape").exists()


def test_home_write_blocked(runner, tmp_path):
    home = os.path.expanduser("~")
    target = f"{home}/.yuj_persistent_escape_marker"
    if os.path.exists(target):
        os.remove(target)
    out, ec, to = runner.run(
        f"touch {target} 2>&1", cwd=str(tmp_path), timeout=10,
    )
    assert ec != 0
    assert not os.path.exists(target), (
        f"persistent bash leaked write to {target}"
    )


def test_tmpfs_isolated_from_host(runner, tmp_path):
    """Writes to /tmp inside the bwrap mount go to the per-session tmpfs,
    NOT the host /tmp. Host /tmp must be unaffected."""
    target_name = f"yuj_persistent_tmpfs_test_{os.getpid()}"
    out, ec, to = runner.run(
        f"touch /tmp/{target_name} 2>&1", cwd=str(tmp_path), timeout=10,
    )
    assert ec == 0  # write succeeds inside bwrap tmpfs
    assert not Path(f"/tmp/{target_name}").exists(), "tmpfs leaked to host"


def test_tmp_cleared_between_calls_when_cwd_outside_tmp(tmp_path_factory):
    """The wrapper runs `find /tmp -mindepth 1 -delete` between calls
    so /tmp is fresh per call (matching the per-call --tmpfs /tmp
    mount). Skipped when cwd is under /tmp because the wrapper opts
    out of the clear there to avoid destroying cwd's own files —
    that's the documented edge-case escape hatch (see sandbox.py).
    To exercise the clear, we need a cwd OUTSIDE /tmp."""
    # Use a non-/tmp dir we know is writable: the worktree itself.
    cwd = str(Path(__file__).resolve().parents[1])
    if cwd == "/tmp" or cwd.startswith("/tmp/"):
        pytest.skip("checkout is under /tmp; this check needs a separate tmpfs")
    p = PersistentBashSession(cwd=cwd)
    try:
        p.start()
        p.run("echo hi > /tmp/cleared_test_marker", cwd=cwd, timeout=10)
        out, ec, to = p.run("ls /tmp", cwd=cwd, timeout=10)
        assert ec == 0
        assert "cleared_test_marker" not in out, (
            "/tmp not cleared between persistent bash calls "
            "with non-/tmp cwd"
        )
    finally:
        p.close()


# ── Timeout + restart ─────────────────────────────────────────────────


def test_timeout_signals_correctly(runner, tmp_path):
    out, ec, to = runner.run("sleep 5", cwd=str(tmp_path), timeout=2)
    assert to is True
    assert ec is None


def test_lazy_restart_after_timeout(runner, tmp_path):
    """After a timeout kills bash, the next call must lazy-restart."""
    runner.run("sleep 5", cwd=str(tmp_path), timeout=2)
    out, ec, to = runner.run("echo recovered", cwd=str(tmp_path), timeout=10)
    assert ec == 0 and "recovered" in out


# ── Registry isolation ────────────────────────────────────────────────


def test_registry_isolated_across_threads(tmp_path):
    """Unrelated threads do not inherit the owning session's context."""
    p = PersistentBashSession(cwd=str(tmp_path))
    set_persistent_runner(p)
    try:
        assert get_persistent_runner() is p
        seen = {}
        def worker():
            seen["runner"] = get_persistent_runner()
        t = threading.Thread(target=worker)
        t.start()
        t.join()
        assert seen["runner"] is None, "runner leaked across threads"
    finally:
        set_persistent_runner(None)
        p.close()


# ── Performance smoke (informational, not asserted) ───────────────────


def test_per_call_overhead_is_small(runner, tmp_path):
    """Sanity: persistent bash should be much faster than per-call.
    Per-call subprocess.run+bwrap is ~50–150 ms; persistent is ~1–3 ms.
    Asserts a generous ceiling so flakes don't fail CI."""
    t0 = time.perf_counter()
    n = 30
    for i in range(n):
        out, ec, to = runner.run(f"echo {i}", cwd=str(tmp_path), timeout=10)
        assert ec == 0 and out.strip() == str(i)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    per_call = elapsed_ms / n
    # Generous ceiling — cold start is fine; observed ~1–2 ms steady state.
    assert per_call < 25.0, f"persistent bash too slow: {per_call:.2f} ms/call"
