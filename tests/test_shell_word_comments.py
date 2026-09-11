"""Quoting a literal executable must not change shared request recognition."""
import os
import subprocess

import pytest
from _config_helpers import make_config
from scripts.llm_solver.harness._shell_patterns import matches_command
from scripts.llm_solver.harness._guardrails.checks_post import observe_contract_state
from scripts.llm_solver.harness._guardrails.extractors import _is_test_command
from scripts.llm_solver.harness._guardrails.state import GuardrailState


@pytest.mark.parametrize('spelling', ['pytest#notes', "'pytest#notes'", r'pytest\#notes'])
def test_literal_helper_has_same_identity_and_guardrail_effect(tmp_path, spelling):
    helper = tmp_path / 'pytest#notes'
    helper.write_text('#!/bin/sh\nprintf "ordinary helper\\n"\n')
    helper.chmod(0o700)
    env = {**os.environ, 'PATH': str(tmp_path) + os.pathsep + os.environ.get('PATH', os.defpath)}
    result = subprocess.run(['bash', '--noprofile', '--norc', '-c', spelling],
                            cwd=tmp_path, env=env, capture_output=True, text=True, check=True)
    assert result.stdout == 'ordinary helper\n'
    assert not matches_command(spelling)
    assert not _is_test_command('bash', {'cmd': spelling})
    state = GuardrailState()
    observe_contract_state(state, make_config(contract_recovery_verify_repeat_threshold=0),
                           tc_name='bash', tc_args={'cmd': spelling},
                           result=result.stdout, gate_blocked=False)
    assert state.verify_repeat_count == 0


@pytest.mark.parametrize('command,recognized', [
    ("pytest # comment with an unmatched '\"", True),
    ('echo ok # comment ; pytest', False),
    ('echo ok # comment <<TEXT\npytest', True),
    ("printf '%s' 'pytest # mention'", False),
    ("pytest -k 'case # literal'", True),
    ('/ordinary#location/pytest tests/', True),
    ('cat note; cargo test', True),
    ('pytest_suffix', False),
])
def test_comments_mentions_and_supported_invocations(command, recognized):
    assert matches_command(command) is recognized
