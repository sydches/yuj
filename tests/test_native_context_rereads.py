"""Resume context and revision evidence read the selected task view."""
from collections import deque
from contextlib import contextmanager
import hashlib
import json

import pytest

from tests._config_helpers import make_config
from tests.test_task_files import bwrap, namespace_files
from tests.test_context_state_location import write_state
from scripts.llm_solver.harness.task_path import activate_task_files
from scripts.llm_solver.harness._guardrails.verification import _file_revision
from scripts.llm_solver.harness.context_strategies._solver_state_io import prepopulate_from_trace


@pytest.mark.parametrize('mode', ['stateful', 'concise'])
@pytest.mark.parametrize('spelling', ['relative', 'host', 'native'])
def test_resume_factory_reads_native_file_from_private_state(bwrap, tmp_path, monkeypatch, mode, spelling):
    from scripts.llm_solver.harness import task_file_runtime
    from scripts.llm_solver.harness._loop._session_setup import build_context_manager
    from scripts.llm_solver.harness.context_strategies import resolve_context_class
    overlay = tmp_path / 'overlay'
    overlay.mkdir()
    (overlay / 'source.py').write_text('NATIVE_RESUMED_SOURCE\n')
    source, files = namespace_files(bwrap, tmp_path, overlay=overlay)
    (source / 'nested/source.py').write_text('HIDDEN_HOST_SOURCE\n')
    artifacts = tmp_path / 'private_records'
    path = {'relative': 'nested/source.py', 'host': str(source / 'nested/source.py'),
            'native': str(files.root / 'nested/source.py')}[spelling]
    write_state(artifacts, 'selected record', path)
    write_state(source, 'wrong record', 'wrong.py')
    (source / 'wrong.py').write_text('WRONG_STATE_FILE\n')
    observed_environment = {'PATH': '/fixture-runtime'}
    calls = []

    @contextmanager
    def scope(cwd, cfg, **options):
        assert cwd == source
        assert options['environment'] is observed_environment
        calls.append(True)
        with activate_task_files(files, host_root=source):
            yield files

    monkeypatch.setattr(task_file_runtime, 'task_file_scope', scope)
    context = build_context_manager(
        resolve_context_class(mode), make_config(sandbox_bash=True), source, 'Task', 2, None,
        artifact_dir=artifacts, effective_env=observed_environment,
    )
    body = (json.dumps(list(context._recent_tool_results)) if hasattr(context, '_recent_tool_results')
            else str(context._ws.files))
    assert calls == [True]
    assert 'NATIVE_RESUMED_SOURCE' in body
    assert 'HIDDEN_HOST_SOURCE' not in body and 'WRONG_STATE_FILE' not in body
    with activate_task_files(files, host_root=source):
        assert _file_revision(source, path) == hashlib.sha256(b'NATIVE_RESUMED_SOURCE\n').hexdigest()


def test_unavailable_native_read_is_not_replaced_with_host_contents(bwrap, tmp_path):
    overlay = tmp_path / 'overlay'
    overlay.mkdir()
    source, files = namespace_files(bwrap, tmp_path, overlay=overlay)
    (source / 'nested/hidden.py').write_text('HIDDEN_HOST_SOURCE')
    state_path = write_state(tmp_path / 'records', 'selected', 'nested/hidden.py')
    recent = deque()
    with activate_task_files(files, host_root=source):
        assert prepopulate_from_trace(source, recent, 1000, state_path=state_path) == 0
        assert _file_revision(source, 'nested/hidden.py') == 'missing'
    assert not recent


def test_resume_does_not_reinterpret_parent_traversal_as_a_task_file(tmp_path):
    task = tmp_path / 'task'
    task.mkdir()
    (task / 'collision.py').write_text('WRONG_REINTERPRETED_PATH')
    state = write_state(tmp_path / 'records', 'selected', '../collision.py')
    recent = deque()
    assert prepopulate_from_trace(task, recent, 1000, state_path=state) == 0
    assert not recent and _file_revision(task, '../collision.py') == ''


@pytest.mark.parametrize('pretest', [False, True])
def test_component_selection_uses_native_candidates_and_declarations(bwrap, tmp_path, pretest):
    from scripts.llm_solver.harness._guardrails.state import GuardrailState
    from scripts.llm_solver.harness._guardrails.verification import resolve_component_verification_target
    overlay = tmp_path / 'overlay'
    (overlay / 'visible').mkdir(parents=True)
    (overlay / 'core.py').write_text('native source')
    (overlay / 'visible/test_core.py').write_text('native candidate')
    source, files = namespace_files(bwrap, tmp_path, overlay=overlay)
    (source / 'pytest.ini').write_text('[pytest]\n')
    (source / 'nested/host').mkdir()
    (source / 'nested/core.py').write_text('hidden source')
    (source / 'nested/host/test_core.py').write_text('hidden candidate')
    state = GuardrailState(post_mutation_source_paths=(str(source / 'nested/core.py'),))
    if pretest:
        state.pretest_failing_tests = {'nested/host/test_core.py::test_old'}
    with activate_task_files(files, host_root=source):
        target = resolve_component_verification_target(state, source)
        assert target is not None
        assert target.path == ''  # Collection is deferred to the selected native runner.
        assert target.source_path == 'nested/core.py' and target.runner == 'pytest'


def test_permission_denied_native_read_does_not_become_a_host_revision(bwrap, tmp_path):
    overlay = tmp_path / 'overlay'
    overlay.mkdir()
    denied = overlay / 'source.py'
    denied.write_text('native denied')
    source, files = namespace_files(bwrap, tmp_path, overlay=overlay)
    (source / 'nested/source.py').write_text('readable hidden host')
    state = write_state(tmp_path / 'records', 'selected', 'nested/source.py')
    denied.chmod(0)
    try:
        recent = deque()
        with activate_task_files(files, host_root=source):
            assert _file_revision(source, 'nested/source.py') == ''
            assert prepopulate_from_trace(source, recent, 1000, state_path=state) == 0
        assert not recent
    finally:
        denied.chmod(0o600)


def test_context_and_revision_readers_apply_the_active_visibility_policy(bwrap, tmp_path):
    from scripts.llm_solver.harness.sandbox.ignore_policy import load_ignore_policy
    from scripts.llm_solver.harness.task_file_runtime import task_file_scope
    source, files = namespace_files(bwrap, tmp_path)
    (source / 'blocked.py').write_text('not admitted to the model')
    (source / '.yujignore').write_text('blocked.py\n')
    state = write_state(tmp_path / 'records', 'selected', 'blocked.py')
    recent = deque()
    with activate_task_files(files, host_root=source):
        policy = load_ignore_policy(source)
        with task_file_scope(source, make_config(sandbox_bash=True), ignore_policy=policy):
            assert _file_revision(source, 'blocked.py') == 'missing'
            assert prepopulate_from_trace(source, recent, 1000, state_path=state) == 0
    assert not recent


@pytest.mark.parametrize('selected', [False, True])
@pytest.mark.parametrize('change', ['missing', 'permission', 'policy'])
def test_current_file_projection_does_not_restore_unavailable_snapshot(
    bwrap, tmp_path, selected, change,
):
    from scripts.llm_solver.harness.context_strategies._working_set import WorkingSet
    from scripts.llm_solver.harness.sandbox.ignore_policy import load_ignore_policy
    from scripts.llm_solver.harness.task_file_runtime import task_file_scope
    source, files = namespace_files(bwrap, tmp_path)
    native_file = source / 'source.py'
    native_file.write_text('PREVIOUS_NATIVE_CONTENT')
    working = WorkingSet(source)

    def project():
        if selected:
            return working.project_selected_files(['source.py'], 10000)[0]
        return working.project_files(10000)[0]

    with activate_task_files(files, host_root=source):
        working.record_read('source.py', 'PREVIOUS_NATIVE_CONTENT', 1)
        assert 'PREVIOUS_NATIVE_CONTENT' in project()
        if change == 'missing':
            native_file.unlink()
        elif change == 'permission':
            native_file.chmod(0)
        else:
            (source / '.yujignore').write_text('source.py\n')
        policy = load_ignore_policy(source)
        try:
            with task_file_scope(source, make_config(sandbox_bash=True), ignore_policy=policy):
                rendered = project()
                assert 'PREVIOUS_NATIVE_CONTENT' not in rendered
                assert 'Current file content unavailable' in rendered
                assert 'source.py' in rendered
        finally:
            if change == 'permission':
                native_file.chmod(0o600)
        native_file.write_text('NEW_NATIVE_CONTENT')
        assert 'NEW_NATIVE_CONTENT' in project()


def test_file_slot_keys_preserve_distinct_path_spellings(tmp_path):
    from scripts.llm_solver.harness.context_strategies._working_set import WorkingSet
    working = WorkingSet(tmp_path)
    for path in ('source.py', '.source.py', '../source.py', '/source.py'):
        working.record_read(path, path, 1)
    working.record_read('./source.py', 'second read', 2)
    assert len(working.files) == 4
    assert working.files['source.py'].content == 'second read'
    working.forget_file('../source.py')
    assert len(working.files) == 3
    assert 'source.py' in working.files and '.source.py' in working.files


def test_slot_candidates_use_native_entries_and_admission(bwrap, tmp_path):
    from types import SimpleNamespace
    from scripts.llm_solver.harness.context_strategies._working_set_baseline_focus import is_repo_file_candidate
    from scripts.llm_solver.harness.sandbox.ignore_policy import load_ignore_policy
    from scripts.llm_solver.harness.task_file_runtime import task_file_scope
    overlay = tmp_path / 'overlay'
    overlay.mkdir()
    (overlay / 'directory.py').mkdir()
    (overlay / 'source.py').write_text('native file')
    source, files = namespace_files(bwrap, tmp_path, overlay=overlay)
    (source / 'nested/directory.py').write_text('hidden host file')
    (source / 'nested/source.py').mkdir()
    (source / '.yujignore').write_text('nested/blocked.py\n')
    (overlay / 'blocked.py').write_text('not admitted')
    ctx = SimpleNamespace(_cwd=source)
    with activate_task_files(files, host_root=source):
        policy = load_ignore_policy(source)
        with task_file_scope(source, make_config(sandbox_bash=True), ignore_policy=policy):
            assert not is_repo_file_candidate(ctx, 'nested/directory.py')
            for path in ('nested/source.py', str(source / 'nested/source.py'),
                         str(files.root / 'nested/source.py')):
                assert is_repo_file_candidate(ctx, path)
            assert not is_repo_file_candidate(ctx, 'nested/blocked.py')
            assert not is_repo_file_candidate(ctx, '../outside.py')
            # Retain the existing hint for an admitted, not-yet-created file.
            assert is_repo_file_candidate(ctx, 'nested/new.py')
