"""Credential checks belong to the entry process and leave task I/O intact."""
from dataclasses import replace
import os
import subprocess

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
