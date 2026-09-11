"""Native report context and coverage must survive their consumers."""
import shlex
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from _config_helpers import make_config
from scripts.llm_solver.harness._guardrails.checks_pre import done_guard
from scripts.llm_solver.harness._guardrails.state import Action, GuardrailState
from scripts.llm_solver.harness._loop.pretest_resume import run_pretest
from scripts.llm_solver.harness._loop.state_projection import update_parity_from_report
from scripts.llm_solver.harness.tools import dispatch


COMMAND = shlex.join([sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider'])


def config():
    return make_config(sandbox_bash=False, analysis_task_format='pytest',
        tools_run_tests_enabled=True, done_guard_enabled=True, done_require_pretest_parity=True,
        done_require_verify=False, done_require_mutation=False, done_parity_runs_required=1,
        done_loop_abort_after=0, post_mutation_verification_gate_after=0)


def baseline(root, commands, cfg):
    script = root / 'baseline.sh'
    script.write_text(commands + '\n')
    report = run_pretest(root, pretest_script=script, pretest_timeout=15,
        pretest_head_chars=20000, pretest_tail_chars=0, report_cfg=cfg).native_test_report
    assert report['status'] == 'available'
    return report


def session(cfg, report):
    return SimpleNamespace(cfg=cfg, _guards=GuardrailState(
        pretest_failing_tests={k for k, v in report['tests'].items() if v in {'FAILED', 'ERROR'}},
        pretest_passing_tests={k for k, v in report['tests'].items() if v == 'PASSED'}))


def followup(root, command, state):
    facts = {}
    dispatch('bash', {'cmd': command}, cwd=str(root), cfg=state.cfg, execution_metadata=facts)
    report = facts['native_test_report']
    assert report['status'] == 'available'
    update_parity_from_report(state, report)
    return report


def test_distinct_collection_roots_cannot_substitute_for_baseline(tmp_path):
    for name, value in [('package_a', False), ('package_b', True)]:
        project = tmp_path / name
        project.mkdir()
        (project / 'pytest.ini').write_text('[pytest]\n')
        (project / 'test_subject.py').write_text(f'def test_target():\n    assert {value}\n')
    cfg = config()
    before = baseline(tmp_path, 'cd package_a && ' + COMMAND + ' test_subject.py', cfg)
    state = session(cfg, before)
    other = followup(tmp_path, 'cd package_b && ' + COMMAND + ' test_subject.py', state)
    assert set(other['tests']).isdisjoint(before['tests'])
    assert done_guard(state._guards, cfg, tc_name='done').action == Action.BLOCK
    (tmp_path / 'package_a/test_subject.py').write_text('def test_target():\n    assert True\n')
    actual = followup(tmp_path, 'cd package_a && ' + COMMAND + ' -s test_subject.py', state)
    assert set(actual['tests']) == set(before['tests'])
    assert done_guard(state._guards, cfg, tc_name='done').action == Action.PASS


def test_multi_invocation_baseline_preserves_both_cases(tmp_path):
    (tmp_path / 'pytest.ini').write_text('[pytest]\n')
    for name in ('first', 'second'):
        (tmp_path / f'test_{name}.py').write_text(f'def test_{name}():\n    assert False\n')
    cfg = config()
    commands = COMMAND + ' test_first.py\n' + COMMAND + ' test_second.py'
    before = baseline(tmp_path, commands, cfg)
    assert len(before['tests']) == 2
    state = session(cfg, before)
    (tmp_path / 'test_second.py').write_text('def test_second():\n    assert True\n')
    followup(tmp_path, COMMAND + ' test_second.py', state)
    assert done_guard(state._guards, cfg, tc_name='done').action == Action.BLOCK
    (tmp_path / 'test_first.py').write_text('def test_first():\n    assert True\n')
    followup(tmp_path, commands, state)
    assert done_guard(state._guards, cfg, tc_name='done').action == Action.PASS


def test_redirected_later_invocation_cannot_leave_partial_baseline_available(tmp_path):
    (tmp_path / 'test_subject.py').write_text('def test_target(): pass\n')
    script = tmp_path / 'baseline.sh'
    script.write_text(COMMAND + '\n' + COMMAND + ' --junitxml=other.xml\n')
    result = run_pretest(tmp_path, pretest_script=script, pretest_timeout=15,
        pretest_head_chars=20000, pretest_tail_chars=0, report_cfg=config())
    assert result.native_test_report['status'] == 'unavailable'
    assert (tmp_path / 'other.xml').is_file()
    assert not list((tmp_path / '.tool_output').glob('report_*.xml'))


@pytest.mark.parametrize('available', [True, False])
def test_automatic_report_reaches_parity_and_trace_once(tmp_path, available):
    from test_automatic_verification_execution import automatic_state
    from scripts.llm_solver.harness._guardrails.verification import ComponentVerificationTarget
    from scripts.llm_solver.harness._loop._dispatch_tool_call import _run_automatic_component_verification
    from dataclasses import replace

    cfg = replace(config(), post_mutation_verification_gate_after=2)
    identity = 'pytest:observed-root::test_subject::test_target'
    guards = GuardrailState(has_mutated=True, pretest_failing_tests={identity},
                            green_parity_streak=0 if available else 1)
    state = automatic_state((tmp_path, cfg, guards))
    report = dict(runner='pytest', invocation='owned-fixture',
                  status='available' if available else 'unavailable',
                  tests={identity: 'PASSED'} if available else {})

    def completed(tool, arguments, **kwargs):
        kwargs['execution_metadata'].update(executed=True, exit_status_known=True,
            exit_status=0, verification_status='passed', native_test_report=report)
        return 'controlled completed check'

    target = ComponentVerificationTarget('test_subject.py', 'test_subject.py', 'pytest', 'subject.py')
    with patch('scripts.llm_solver.harness._guardrails.verification.resolve_component_verification_target',
               return_value=target), patch.object(state, 'dispatch', side_effect=completed) as dispatch_check:
        _run_automatic_component_verification(SimpleNamespace(id='check'), state, 'prior output', {})
    dispatch_check.assert_called_once()
    state.session._emit.assert_called_once_with('native_test_report', turn_number=1, report=report)
    assert guards.green_parity_streak == int(available)
    assert done_guard(guards, cfg, tc_name='done').action == (Action.PASS if available else Action.BLOCK)
