"""The retained protocol runs real native children without requiring Docker."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import base64
import os
from pathlib import Path
import shutil
import signal
import subprocess
import threading
import time

import pytest

from scripts.llm_solver.harness.process_identity import (
    OBSERVE_SCRIPT, ProcessIdentityError, guarded_process_argv, parse_process_identity,
)
from scripts.llm_solver.harness.sandbox.env_policy import build_bash_argv, build_clean_exec_argv
from scripts.llm_solver.harness.sandbox._persistent_container import (
    PersistentContainerSession, UnsupportedPersistentContainer,
)
from scripts.llm_solver.harness.task_environment import TaskEnvironment
from scripts.llm_solver.harness.time_budget import run_time_budget


@pytest.fixture
def native(tmp_path, monkeypatch):
    identity = parse_process_identity(subprocess.run(
        ['bash', '-p', '-c', OBSERVE_SCRIPT], capture_output=True, check=True).stdout)
    environment = TaskEnvironment(str(tmp_path), str(tmp_path), (), container='fixture',
                                  container_id='fixture', process_identity=identity)
    effective = {'PATH': os.environ['PATH'], 'HOME': str(tmp_path), 'LANG': 'C.UTF-8'}
    original_popen, original_run = subprocess.Popen, subprocess.run

    def unwrap(argv, kwargs):
        if list(argv[:2]) == ['docker', 'exec']:
            kwargs = dict(kwargs, cwd=argv[argv.index('--workdir') + 1])
            argv = argv[argv.index('fixture') + 1:]
        return argv, kwargs

    def popen(argv, *args, **kwargs):
        argv, kwargs = unwrap(argv, kwargs)
        return original_popen(argv, *args, **kwargs)

    def run(argv, *args, **kwargs):
        argv, kwargs = unwrap(argv, kwargs)
        return original_run(argv, *args, **kwargs)

    monkeypatch.setattr(subprocess, 'Popen', popen)
    monkeypatch.setattr(subprocess, 'run', run)
    owners = []

    def create(selected=None):
        values = effective if selected is None else selected
        owner = PersistentContainerSession(str(tmp_path), environment, effective_env=values)
        owners.append(owner)

        def execute(command, *, timeout=3, binary=True, input_bytes=None):
            argv = guarded_process_argv(['docker', 'exec', '--workdir', str(tmp_path), 'fixture'],
                build_clean_exec_argv(build_bash_argv(command), values), identity)
            return owner.execute(argv, timeout=timeout, binary=binary, input_bytes=input_bytes)
        return owner, execute

    yield create, environment, effective
    for owner in owners:
        owner.close()


def test_separate_bytes_status_large_input_and_next_request(native):
    create, _, _ = native
    owner, execute = create()
    for data in [b'', b'\0\xff\r\n', bytes(range(256))*4096, b'next\n\0']:
        result = execute('cat; printf err >&2; exit 23', input_bytes=data)
        assert result.stdout == data and result.stderr == b'err' and result.returncode == 23
    assert execute('printf final', timeout=None).stdout == b'final'
    owner.close()
    assert owner._process is None


def test_text_decode_newlines_and_strict_invalid_bytes(native):
    _, execute = native[0]()
    result = execute("printf 'a\\r\\nb\\r'; printf 'err\\r\\n' >&2", binary=False)
    assert result.stdout == 'a\nb\n' and result.stderr == 'err\n'
    with pytest.raises(UnicodeDecodeError):
        execute("printf '\\377'", binary=False)


def test_filesystem_surrogate_argv_and_embedded_nul(native, tmp_path):
    create, environment, _ = native
    owner, execute = create()
    assert execute("printf '%s' '" + os.fsdecode(b'\xff') + "'").stdout == b'\xff'
    other, _ = create()
    argv = guarded_process_argv(['docker', 'exec', '--workdir', str(tmp_path), 'fixture'],
                               ['bash', '-c', 'printf wrong\0'], environment.process_identity)
    with pytest.raises(ValueError, match='embedded null byte'):
        other.execute(argv, timeout=1, binary=True)
    assert other._process is None


def test_fresh_environment_and_shell_state(native, tmp_path):
    create, _, effective = native
    startup = tmp_path / 'startup'
    startup.write_text('printf startup\n')
    owner, execute = create(dict(effective, BASH_ENV=str(startup)))
    assert execute('export LEAK=1; cd /; umask 077; set -e').stdout == b'startup'
    assert execute('printf "%s:%s" "$PWD" "${LEAK-unset}"').stdout == b'startup' + str(tmp_path).encode() + b':unset'
    process = owner._process
    owner.close()
    assert process.poll() is not None


def test_optional_capability_failure_precedes_task_and_is_cached(native, tmp_path):
    create, _, effective = native
    tools = tmp_path / 'bin'
    tools.mkdir()
    for name in ['bash', 'wc', 'base64', 'mktemp', 'rm', 'rmdir', 'setsid', 'cat']:
        (tools / name).symlink_to(shutil.which(name))
    owner, execute = create(dict(effective, PATH=str(tools)))
    for _ in range(2):
        with pytest.raises(UnsupportedPersistentContainer):
            execute('printf wrong > started')
    assert owner._unsupported and owner._process is None
    assert not (tmp_path / 'started').exists()


def test_missing_native_setsid_allows_unbounded_one_shot_fallback(native, tmp_path, monkeypatch):
    from scripts.llm_solver.harness._tools._run_in_sandbox import _execute
    create, environment, effective = native
    tools = tmp_path / 'native-bin'
    tools.mkdir()
    (tools / 'bash').symlink_to(shutil.which('bash'))
    original = subprocess.Popen
    def without_setsid(argv, *args, **kwargs):
        if list(argv[:2]) == ['docker', 'exec']:
            kwargs['env'] = dict(os.environ, PATH=str(tools))
        return original(argv, *args, **kwargs)
    monkeypatch.setattr(subprocess, 'Popen', without_setsid)
    owner, execute = create()
    with pytest.raises(UnsupportedPersistentContainer):
        execute('printf wrong > started', timeout=None)
    assert not (tmp_path / 'started').exists()
    argv = guarded_process_argv(['docker', 'exec', '--workdir', str(tmp_path), 'fixture'],
        build_clean_exec_argv(build_bash_argv('printf fallback'), effective), environment.process_identity)
    assert _execute(argv, timeout=None, binary=True).stdout == b'fallback'


def test_failed_capture_cannot_interpret_unconsumed_payload(native, tmp_path):
    create, _, effective = native
    tools = tmp_path / 'bin'
    tools.mkdir()
    for name in ['bash', 'head', 'wc', 'base64', 'mktemp', 'rm', 'rmdir', 'setsid', 'cat']:
        (tools / name).symlink_to(shutil.which(name))
    owner, execute = create(dict(effective, PATH=str(tools)))
    assert execute(':').returncode == 0
    (tools / 'mktemp').unlink()
    (tools / 'mktemp').write_text('#!/bin/sh\nexit 74\n')
    (tools / 'mktemp').chmod(0o755)
    payload = base64.b64encode(b'printf injected > injected') + b'\n'
    with pytest.raises(RuntimeError, match='capture failed'):
        execute('printf task > started', input_bytes=payload)
    assert not (tmp_path / 'injected').exists()
    assert not (tmp_path / 'started').exists()
    assert owner._process is None


def test_lock_wait_uses_same_call_allowance(native):
    owner, execute = native[0]()
    entered, release = threading.Event(), threading.Event()
    def hold():
        with owner._lock:
            entered.set()
            release.wait(2)
    thread = threading.Thread(target=hold)
    thread.start()
    assert entered.wait(1)
    began = time.monotonic()
    try:
        with pytest.raises(subprocess.TimeoutExpired):
            execute(':', timeout=.03)
        assert time.monotonic() - began < .2
        assert owner._process is None
    finally:
        release.set()
        thread.join()


def test_pipe_streams_wait_for_inherited_background_output(native):
    _, execute = native[0]()
    result = execute('[[ -p /dev/stdin && -p /dev/stdout && -p /dev/stderr ]] || exit 19; '
                     '(sleep .03; printf late; printf err >&2) &')
    assert result.returncode == 0
    assert result.stdout == b'late' and result.stderr == b'err'


def test_closed_background_streams_do_not_hold_internal_capture_fds(native):
    _, execute = native[0]()
    execute(':')
    began = time.monotonic()
    result = execute('(exec 1>&- 2>&-; sleep .4) & printf done')
    assert result.stdout == b'done' and result.returncode == 0
    assert time.monotonic() - began < .25


def test_truncated_payload_stops_before_task(native, tmp_path, monkeypatch):
    from scripts.llm_solver.harness.sandbox._persistent_container import _ACK, _RAW_CAPTURE
    result = subprocess.run(['bash', '-p', '-c', _RAW_CAPTURE, 'capture', '3',
                             'bash', '-c', 'printf wrong > started'],
                            input=_ACK + b'ab', cwd=tmp_path, capture_output=True)
    assert result.returncode == 74
    assert not (tmp_path / 'started').exists()


def test_cancel_before_startup_frame_cannot_launch_unbounded_task(native, tmp_path, monkeypatch):
    owner, execute = native[0]()
    execute(':')
    process = owner._process
    original = os.read
    interrupted = False
    def interrupt(fd, size):
        nonlocal interrupted
        if fd == process.stdout.fileno() and owner._call_group is not None and not interrupted:
            interrupted = True
            raise KeyboardInterrupt('before native handle consumption')
        return original(fd, size)
    monkeypatch.setattr(os, 'read', interrupt)
    with pytest.raises(KeyboardInterrupt):
        execute('printf wrong > started; sleep 30', timeout=None)
    assert interrupted and not (tmp_path / 'started').exists()
    assert process.poll() is not None and owner._process is None


def test_blocked_large_write_times_out_and_stops_owned_transport(native):
    owner, execute = native[0]()
    execute(':')
    process = owner._process
    os.killpg(owner._group.handle[0], signal.SIGSTOP)
    began = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        execute('cat', input_bytes=b'x' * (2 * 1024 * 1024), timeout=.03)
    assert time.monotonic() - began < .4
    assert process.poll() is not None and owner._process is None


def test_native_cleanup_error_still_reaps_local_transport(native, monkeypatch):
    owner, execute = native[0]()
    execute(':')
    process, group = owner._process, owner._group
    original = group.terminate
    monkeypatch.setattr(group, 'terminate', lambda **kwargs: (_ for _ in ()).throw(OSError('cleanup failed')))
    with pytest.raises(OSError, match='cleanup failed'):
        owner._stop(force=True)
    assert process.poll() is not None and owner._process is None
    original(grace=0)


def test_timeout_with_unconfirmed_cleanup_remains_a_hard_error(native, monkeypatch):
    owner, execute = native[0]()
    execute(':')
    process, group = owner._process, owner._group
    original = group.terminate
    monkeypatch.setattr(group, 'terminate', lambda **kwargs: (_ for _ in ()).throw(OSError('cleanup denied')))
    with pytest.raises(ProcessIdentityError, match='termination was not confirmed.*cleanup denied') as failure:
        execute('sleep 1', timeout=.03)
    assert isinstance(failure.value.__cause__, subprocess.TimeoutExpired)
    assert process.poll() is not None and owner._process is None
    original(grace=0)


@pytest.mark.parametrize('timeout', [None, 10])
def test_active_run_deadline_limits_retained_execution(native, tmp_path, timeout):
    owner, execute = native[0]()
    execute(':')
    began = time.monotonic()
    with run_time_budget(.08):
        with pytest.raises(subprocess.TimeoutExpired):
            execute('sleep .5; printf late > late', timeout=timeout)
    assert time.monotonic() - began < .4
    time.sleep(.5)
    assert not (tmp_path / 'late').exists()
    assert owner._process is None


def test_serialization_and_split_response_marker(native, monkeypatch):
    owner, execute = native[0]()
    execute(':')
    original = os.read
    fd = owner._process.stdout.fileno()
    monkeypatch.setattr(os, 'read', lambda observed, count: original(observed, 1 if observed == fd else count))
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(execute, f'sleep .01; printf result-{index}') for index in range(2)]
        assert [future.result().stdout for future in futures] == [b'result-0', b'result-1']


def test_binding_and_credential_mismatch_do_not_run_task(native, tmp_path):
    create, environment, _ = native
    owner, execute = create()
    assert owner.supports(environment, str(tmp_path))
    assert not owner.supports(replace(environment, container_id='other'), str(tmp_path))
    owner.environment = replace(environment, process_identity=replace(environment.process_identity, uids=(999,)*4))
    with pytest.raises(ProcessIdentityError):
        execute('printf wrong > started')
    assert not (tmp_path / 'started').exists()
