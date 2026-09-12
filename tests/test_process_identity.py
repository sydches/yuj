"""Credential checks belong to the entry process and leave task I/O intact."""
from dataclasses import replace
import os
import subprocess
import time

import pytest

from scripts.llm_solver.harness.process_identity import (
    OBSERVE_SCRIPT, VERIFIED, ProcessIdentityError, guarded_process_argv,
    parse_process_identity, guarded_script_argv,
)
from scripts.llm_solver.harness._tools._run_in_sandbox import _execute


def observed():
    result = subprocess.run(['bash', '--noprofile', '--norc', '-p', '-c', OBSERVE_SCRIPT],
                            capture_output=True, check=True)
    return parse_process_identity(result.stdout)


def test_credentials_come_from_the_native_process():
    identity = observed()
    assert identity.uids[:2] == (os.getuid(), os.geteuid())
    assert identity.gids[:2] == (os.getgid(), os.getegid())
    assert set(identity.groups) == set(os.getgroups())


@pytest.mark.parametrize('binary', [False, True])
def test_guard_preserves_command_output_and_its_exit_status(binary):
    argv = guarded_process_argv([], ['bash', '-c', 'printf output; printf diagnostic >&2; exit 77'], observed())
    result = _execute(argv, timeout=None, binary=binary)
    assert result.stdout == (b'output' if binary else 'output')
    assert result.stderr == (b'diagnostic' if binary else 'diagnostic')
    assert result.returncode == 77


def test_guard_preserves_binary_stdin_and_does_not_strip_task_frames():
    data = VERIFIED + b'\0\xff\r\n'
    result = _execute(guarded_process_argv([], ['cat'], observed()), timeout=None,
                      binary=True, input_bytes=data)
    assert result.stdout == data


def test_failed_container_entry_keeps_the_actual_launch_error():
    from scripts.llm_solver.harness.process_identity import verified_process_result
    argv = guarded_process_argv(['docker', 'exec', 'fixture'], ['true'], observed())
    result = subprocess.CompletedProcess(argv, 128, b'', b'OCI runtime exec failed: procReady not received')
    with pytest.raises(ProcessIdentityError, match='OCI runtime exec failed: procReady not received'):
        verified_process_result(argv, result)


def test_container_deadline_stops_a_child_before_it_can_write(tmp_path):
    from scripts.llm_solver.harness.process_identity import with_container_deadline
    marker = tmp_path / 'late-write'
    argv = guarded_process_argv(['docker', 'exec', 'fixture'],
        ['bash', '-c', 'sleep 0.6; printf escaped > "$1"', 'fixture', str(marker)], observed())
    wrapped = with_container_deadline(argv, 0.2)
    # Run the exact container-side command locally, including its guard.
    result = subprocess.run(wrapped[3:], capture_output=True, timeout=2)
    assert result.returncode == 124
    import time
    time.sleep(0.65)
    assert not marker.exists()


@pytest.mark.parametrize('field', ['uids', 'gids', 'groups', 'capabilities'])
def test_changed_credentials_prevent_the_command_from_starting(tmp_path, field):
    identity = observed()
    values = list(getattr(identity, field))
    if field == 'capabilities':
        values[0] = format(int(values[0], 16) ^ 1, '016x')
    elif field == 'groups':
        values.append(max(values, default=0) + 1)
    else:
        values[1] += 1
    different = replace(identity, **{field: tuple(values)})
    target = tmp_path / 'must-not-exist'
    argv = guarded_process_argv([], ['touch', str(target)], different)
    with pytest.raises(ProcessIdentityError, match='did not start'):
        _execute(argv, timeout=None, binary=True)
    assert not target.exists()


def test_entry_guard_does_not_source_bash_env(tmp_path):
    script = tmp_path / 'startup'
    target = tmp_path / 'startup-ran'
    script.write_text('touch "$SIDE_EFFECT"\n')
    env = {**os.environ, 'BASH_ENV': str(script), 'SIDE_EFFECT': str(target)}
    result = _execute(guarded_process_argv([], ['true'], observed()), timeout=None,
                      binary=True, env=env)
    assert result.returncode == 0 and not target.exists()


def test_environment_observation_preserves_initial_values_without_shell_additions(tmp_path):
    from scripts.llm_solver.harness.sandbox.env_policy import _READ_INITIAL_ENVIRONMENT
    script = tmp_path / 'startup'
    target = tmp_path / 'startup-ran'
    script.write_text('touch "$SIDE_EFFECT"\n')
    initial = {'PATH': os.environ['PATH'], 'HOME': '/fixture/home', 'SHLVL': '37',
               'PWD': '/initial/spelling', 'BASH_ENV': str(script), 'SIDE_EFFECT': str(target),
               'LITERAL': 'line1\nline2=$(do-not-execute)=value'}
    result = _execute(guarded_script_argv([], _READ_INITIAL_ENVIRONMENT, observed()),
                      timeout=None, binary=True, env=initial)
    values = dict(entry.decode().split('=', 1) for entry in result.stdout.split(b'\0')[:-1])
    assert result.returncode == 0 and values == initial
    assert not target.exists()


@pytest.mark.parametrize('payload', [b'', b'Uid\x000 0 0 0\x00', b'\xff'])
def test_incomplete_process_identity_is_not_inferred(payload):
    with pytest.raises(ProcessIdentityError):
        parse_process_identity(payload)


@pytest.mark.parametrize('split', range(len(VERIFIED) + 1))
def test_stream_verification_preserves_task_bytes_at_every_frame_split(split):
    from scripts.llm_solver.harness.process_identity import GuardedProcessArgv, ProcessVerification
    verification = ProcessVerification(GuardedProcessArgv([]))
    assert verification.feed(VERIFIED[:split]) == b''
    assert verification.feed(VERIFIED[split:] + VERIFIED + b'\0\xff') == VERIFIED + b'\0\xff'
    verification.finish()


@pytest.mark.parametrize('data', [b'', VERIFIED[:-1], b'invalid-frame'])
def test_stream_verification_rejects_missing_partial_and_invalid_frames(data):
    from scripts.llm_solver.harness.process_identity import GuardedProcessArgv, ProcessVerification
    verification = ProcessVerification(GuardedProcessArgv([]))
    with pytest.raises(ProcessIdentityError):
        verification.feed(data)
        verification.finish()


def supervised_command(command):
    from scripts.llm_solver.harness.process_identity import with_container_process_group
    argv = with_container_process_group(
        guarded_process_argv(['docker', 'exec', 'fixture'], command, observed()))
    # Execute the exact task-side programs locally, without a Docker daemon.
    argv.container_process_group.cleanup_argv = argv.container_process_group.cleanup_argv[3:]
    return argv


@pytest.mark.parametrize('start_new_session', [False, True])
def test_supervised_group_preserves_stdin_output_status_and_split_startup_frame(start_new_session):
    from scripts.llm_solver.harness.process_identity import ProcessVerification
    argv = supervised_command(['bash', '-c', 'cat; printf diagnostic >&2; exit 23'])
    data = VERIFIED + b'\0\xff\r\n'
    result = subprocess.run(argv[3:], input=data, capture_output=True, timeout=3,
                            start_new_session=start_new_session)
    verification = ProcessVerification(argv)
    output = b''.join(verification.feed(bytes([byte])) for byte in result.stdout)
    verification.finish()
    assert output == data
    assert result.returncode == 23 and result.stderr == b'diagnostic'
    # Completion and a reused PID must not signal a different process.
    argv.container_process_group.terminate(grace=0)


def test_native_group_cancellation_stops_term_resistant_descendant(tmp_path):
    from scripts.llm_solver.harness.process_identity import ProcessVerification
    ready, late = tmp_path / 'ready', tmp_path / 'late'
    descendant = 'trap "" TERM; printf ready > "$1"; sleep 0.5; printf leaked > "$2"'
    argv = supervised_command(['bash', '-c',
        'bash -c "$1" descendant "$2" "$3" & wait',
        'primary', descendant, str(ready), str(late)])
    process = subprocess.Popen(argv[3:], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    group = argv.container_process_group
    try:
        ProcessVerification(argv).read(process.stdout)
        deadline = time.monotonic() + 2
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready.exists()
        handle = group.handle
        group.handle = (handle[0], handle[1] + 1)
        group.terminate(grace=0)
        assert process.poll() is None  # The start-time check rejected this PID.
        group.handle = handle
        group.terminate(grace=0.02)
        process.wait(timeout=2)
        time.sleep(0.55)
        assert not late.exists()
    finally:
        if process.poll() is None:
            group.terminate(grace=0)
            process.wait(timeout=2)
