"""Generated outputs need task declarations and observed creation, not names."""
import shlex
import subprocess
import sys

import pytest

from _config_helpers import make_config
from _verification_fixture import account_check_changes, seed_edited_input
from scripts.llm_solver.harness._guardrails.checks_post import mark_bash_verified
from scripts.llm_solver.harness._guardrails.state import GuardrailState
from scripts.llm_solver.harness._guardrails.verification import observe_post_mutation_verification
from scripts.llm_solver.harness.tools import dispatch
from tests.test_task_files import bwrap


def git(root, *args):
    return subprocess.run(['git', '-C', str(root), *args], check=True, capture_output=True)


def task(root, monkeypatch, *, ignored=True, sandbox=False):
    monkeypatch.delenv('YUJ_CONTAINER', raising=False)
    git(root, 'init', '-q')
    (root / '.gitignore').write_text('/check-output/\n' if ignored else '')
    (root / 'module.py').write_text('edited input\n')
    git(root, 'add', '.gitignore', 'module.py')
    cfg = make_config(sandbox_bash=sandbox, turn_snapshots_enabled=False)
    guards = GuardrailState(has_mutated=True)
    seed_edited_input(guards, cfg, root, 'module.py')
    return cfg, guards


def run_check(root, cfg, guards, *, path='check-output/chart.png', extra='',
              exit_code=0, pipeline=False, prefix='', executable=None):
    (root / 'runtests.py').write_text(
        'from pathlib import Path\nimport subprocess\n'
        f'p = Path({path!r})\np.parent.mkdir(parents=True, exist_ok=True)\n'
        "p.write_bytes((p.read_bytes() if p.exists() else b'') + b'generated')\n" + extra
        + f'raise SystemExit({exit_code})\n'
    )
    args = {'cmd': prefix + shlex.join([executable or sys.executable, 'runtests.py'])
            + (' | tail -10' if pipeline else '')}
    facts = {}
    result = dispatch('bash', args, cwd=str(root), cfg=cfg, execution_metadata=facts)
    assert facts['exit_status'] == exit_code, result
    account_check_changes(guards, cfg, root, 'bash', args, result, facts)
    for observer in (mark_bash_verified, observe_post_mutation_verification):
        observer(guards, cfg, tc_name='bash', tc_args=args, result=result,
                 gate_blocked=False, execution_metadata=facts, cwd=root)
    return facts


@pytest.mark.parametrize('suffix', ['png', 'dat'])
def test_new_declared_output_and_later_updates_do_not_invalidate_check(tmp_path, monkeypatch, suffix):
    cfg, guards = task(tmp_path, monkeypatch)
    path = f'check-output/result.{suffix}'
    for _ in range(2):
        facts = run_check(tmp_path, cfg, guards, path=path)
        assert path in facts['file_changes']['changed_paths']
        assert path in facts['file_changes']['test_artifact_paths']
        assert guards.verified_since_mutation
        assert guards.formal_verification_passed_since_mutation


@pytest.mark.parametrize('case', ['tracked', 'tracked_missing', 'existing', 'unignored',
                                 'edited', 'changed_ignore', 'changed_index'])
def test_real_inputs_and_changed_declarations_remain_mutations(tmp_path, monkeypatch, case):
    cfg, guards = task(tmp_path, monkeypatch, ignored=case not in {'unignored', 'changed_ignore'})
    path = 'check-output/chart.png'
    target = tmp_path / path
    if case in {'tracked', 'tracked_missing', 'existing', 'edited'}:
        target.parent.mkdir()
        target.write_bytes(b'input image')
    if case in {'tracked', 'tracked_missing'}:
        git(tmp_path, 'add', '-f', path)
    if case == 'tracked_missing':
        target.unlink()
    if case == 'edited':
        seed_edited_input(guards, cfg, tmp_path, path)
    extra = ''
    if case == 'changed_ignore':
        extra = "Path('.gitignore').write_text('/check-output/\\n')\n"
    elif case == 'changed_index':
        extra = f"subprocess.run(['git', 'add', '-f', {path!r}], check=True)\n"
    facts = run_check(tmp_path, cfg, guards, extra=extra)
    assert path not in facts['file_changes'].get('test_artifact_paths', ())
    assert not guards.verified_since_mutation
    assert not guards.formal_verification_passed_since_mutation


def test_ordinary_edit_revokes_generated_output_provenance(tmp_path, monkeypatch):
    cfg, guards = task(tmp_path, monkeypatch)
    path = 'check-output/chart.png'
    run_check(tmp_path, cfg, guards, path=path)
    assert path in guards.generated_test_artifacts
    args = {'cmd': f'printf manually-edited > {shlex.quote(path)}'}
    facts = {}
    result = dispatch('bash', args, cwd=str(tmp_path), cfg=cfg, execution_metadata=facts)
    account_check_changes(guards, cfg, tmp_path, 'bash', args, result, facts)
    assert path not in guards.generated_test_artifacts
    facts = run_check(tmp_path, cfg, guards, path=path)
    assert path not in facts['file_changes'].get('test_artifact_paths', ())
    assert not guards.verified_since_mutation


def test_failed_pipeline_records_generation_without_claiming_success(tmp_path, monkeypatch):
    cfg, guards = task(tmp_path, monkeypatch)
    path = 'check-output/chart.png'
    failed = run_check(tmp_path, cfg, guards, exit_code=1, pipeline=True)
    assert failed['verification_status'] == 'shell_unresolved'
    assert path in guards.generated_test_artifacts
    assert not guards.verified_since_mutation
    passed = run_check(tmp_path, cfg, guards, pipeline=True)
    assert passed['verification_status'] == 'passed'
    assert path in passed['file_changes']['test_artifact_paths']
    assert guards.verified_since_mutation


def test_manual_write_cannot_gain_artifact_status_by_appending_a_check(tmp_path, monkeypatch):
    cfg, guards = task(tmp_path, monkeypatch)
    program = "from pathlib import Path; Path('check-output').mkdir(); Path('check-output/chart.png').write_bytes(b'input')"
    prefix = shlex.join([sys.executable, '-c', program]) + ' && '
    facts = run_check(tmp_path, cfg, guards, prefix=prefix)
    assert facts['verification_status'] == 'passed'
    assert 'check-output/chart.png' not in guards.generated_test_artifacts
    assert not guards.verified_since_mutation


def test_failed_rerun_does_not_keep_previous_verification_credit(tmp_path, monkeypatch):
    cfg, guards = task(tmp_path, monkeypatch)
    run_check(tmp_path, cfg, guards, pipeline=True)
    assert guards.verified_since_mutation
    assert guards.formal_verification_passed_since_mutation
    failed = run_check(tmp_path, cfg, guards, exit_code=1, pipeline=True)
    assert failed['verification_status'] == 'shell_unresolved'
    assert 'check-output/chart.png' in failed['file_changes']['test_artifact_paths']
    assert not guards.verified_since_mutation
    assert not guards.formal_verification_passed_since_mutation


def test_sandboxed_check_uses_the_same_project_output_contract(tmp_path, monkeypatch, bwrap):
    cfg, guards = task(tmp_path, monkeypatch, sandbox=True)
    facts = run_check(tmp_path, cfg, guards, executable='/usr/bin/python3')
    assert 'check-output/chart.png' in facts['file_changes']['test_artifact_paths']
    assert guards.verified_since_mutation


@pytest.mark.parametrize('case', ['generated', 'tracked', 'existing', 'edited', 'unignored', 'mixed'])
@pytest.mark.parametrize('script', ['build', 'test:unit'])
def test_build_outputs_stay_out_of_verification_inputs_without_granting_a_pass(tmp_path, monkeypatch, case, script):
    import json
    import shutil
    if shutil.which('npm') is None:
        pytest.skip('npm unavailable')
    cfg, guards = task(tmp_path, monkeypatch, ignored=case != 'unignored')
    path = 'check-output/result.js'
    target = tmp_path / path
    if case in {'tracked', 'existing', 'edited'}:
        target.parent.mkdir()
        target.write_text('input')
    if case == 'tracked':
        git(tmp_path, 'add', '-f', path)
    if case == 'edited':
        seed_edited_input(guards, cfg, tmp_path, path)
    (tmp_path / 'build.py').write_text(
        "from pathlib import Path\np = Path('check-output/result.js')\n"
        "p.parent.mkdir(exist_ok=True)\np.write_text('built')\n")
    (tmp_path / 'package.json').write_text(json.dumps({'scripts': {
        script: shlex.join([sys.executable, 'build.py'])}}))
    args = {'cmd': f'npm run {script} 2>&1 | tail -10'}
    if case == 'mixed':
        args['cmd'] += ' && printf changed > module.py'
    for _ in range(2):
        facts = {}
        result = dispatch('bash', args, cwd=str(tmp_path), cfg=cfg, execution_metadata=facts)
        assert facts['exit_status'] == 0, result
        account_check_changes(guards, cfg, tmp_path, 'bash', args, result, facts)
        assert path in facts['file_changes']['changed_paths']
        assert (path in facts['file_changes'].get('test_artifact_paths', ())) is (case == 'generated')
        assert (path in guards.verification_file_revisions) is (case != 'generated')
        assert not guards.verified_since_mutation
        assert facts['verification_status'] not in {'passed', 'custom_passed'}
