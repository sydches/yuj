"""Verification advice compares delivered excerpts with native task bytes."""
from dataclasses import replace
import hashlib
from pathlib import Path
from unittest.mock import patch

import pytest

from tests._config_helpers import make_config
from tests.test_task_files import bwrap, namespace_files
from scripts.llm_solver.harness._guardrails.test_inspection import (
    observe_test_file_read, _workspace_coverage,
)
from scripts.llm_solver.harness.guardrails import init_guardrail_state
from scripts.llm_solver.harness.runner_invocations import bind_runner_workspace
from scripts.llm_solver.harness.task_file_runtime import task_file_scope
from scripts.llm_solver.harness.task_path import activate_task_files
from scripts.llm_solver.harness.tools import dispatch


@pytest.fixture
def rig(bwrap, tmp_path):
    overlay = tmp_path / 'overlay'
    overlay.mkdir()
    (overlay / 'cases.py').write_text('native first\nnative second\n')
    source, files = namespace_files(bwrap, tmp_path, overlay=overlay)
    (source / 'nested/cases.py').write_text('HIDDEN_HOST_COPY\n')
    cfg = make_config(test_read_warn_after=1, sandbox_bash=True, bwrap_bin=bwrap,
                      test_read_nudge='{runner}|{target}|{coverage}')
    state = init_guardrail_state(cfg)
    return source, files, overlay, cfg, state


def observe(root, cfg, state, **args):
    metadata = {}
    result = dispatch('read', args, cwd=str(root), cfg=cfg, execution_metadata=metadata)
    observe_test_file_read(state, cfg, gate_blocked=False, result=result,
                           execution_metadata=metadata)
    return result, metadata


def request(root):
    # A literal command submission receipt; no runner or benchmark is executed.
    return bind_runner_workspace({'family': 'pytest'}, 'pytest nested/cases.py', str(root))


def test_verification_batches_all_tracked_files_and_checks_canonical_ignore_rules(bwrap, tmp_path):
    import os
    from scripts.llm_solver.harness._guardrails.state import GuardrailState
    from scripts.llm_solver.harness._guardrails.verification import record_verification_mutation, verification_tree_matches
    from scripts.llm_solver.harness.sandbox.ignore_policy import activate_ignore_policy, load_ignore_policy
    source, files = namespace_files(bwrap, tmp_path)
    (source / '__pycache__').mkdir()
    names = [f'__pycache__/module_{i}.pyc' for i in range(23)]
    names[0] = '__pycache__/literal\nmodule.pyc'
    for name in names:
        (source / name).write_bytes(name.encode())
    (source / '.yujignore').write_text('hidden.py\n')
    (source / 'hidden.py').write_bytes(b'not observable')
    (source / 'alias.py').symlink_to('hidden.py')
    state = GuardrailState()
    operations = []
    original_run = files.run
    def observed(script, args, data):
        operations.append(args[3] if len(args) > 3 else 'discovery')
        return original_run(script, args, data)
    files.run = observed
    with activate_task_files(files, host_root=source):
        policy = load_ignore_policy(source)
        with activate_ignore_policy(policy):
            operations.clear()
            record_verification_mutation(state, tc_name='edit', result='OK',
                source_write_paths=tuple([*names, 'alias.py']), cwd=str(source))
            assert len(state.verification_file_revisions) == 24
            assert state.verification_file_revisions['alias.py'] == 'missing'
            assert operations.count('resolve_files') == 1
            assert operations.count('digest_batch') == 1
            assert 'resolve' not in operations and 'stat' not in operations
            operations.clear()
            assert verification_tree_matches(state, str(source))
            assert operations == ['resolve_files', 'digest_batch']
            changed = source / names[-1]
            info = changed.stat()
            changed.write_bytes(b'X' * info.st_size)
            os.utime(changed, ns=(info.st_atime_ns, info.st_mtime_ns))
            assert not verification_tree_matches(state, str(source))


def test_batched_verification_preserves_scalar_missing_and_unknown_results(bwrap, tmp_path):
    from scripts.llm_solver.harness._guardrails.verification import _file_revision, _file_revisions
    source, files = namespace_files(bwrap, tmp_path)
    (source / 'visible').write_bytes(b'bytes')
    (source / 'directory').mkdir()
    (source / 'denied').write_bytes(b'denied')
    (source / 'denied').chmod(0)
    (source / 'escape').symlink_to(tmp_path / 'outside')
    (tmp_path / 'outside').write_bytes(b'outside')
    names = ['visible', 'missing', 'directory', 'denied', 'escape']
    with activate_task_files(files, host_root=source):
        expected = {name: _file_revision(str(source), name) for name in names}
        assert expected == {'visible': hashlib.sha256(b'bytes').hexdigest(),
                            'missing': 'missing', 'directory': '', 'denied': '', 'escape': ''}
        assert _file_revisions(str(source), names) == expected


def test_bulk_removed_files_and_directory_links_do_not_use_scalar_reads(bwrap, tmp_path):
    from scripts.llm_solver.harness._guardrails import verification
    source, files = namespace_files(bwrap, tmp_path)
    (source / 'package').mkdir()
    (source / 'package/index.js').write_bytes(b'new package bytes')
    names = [f'removed/file_{i}.js' for i in range(1000)]
    links = [f'package-link-{i}' for i in range(100)]
    for name in links:
        (source / name).symlink_to('package', target_is_directory=True)
    operations = []
    original_run = files.run
    def observed(script, args, data):
        operations.append(args[3] if len(args) > 3 else 'discovery')
        return original_run(script, args, data)
    files.run = observed
    with activate_task_files(files, host_root=source), patch.object(
        verification, '_file_revision', side_effect=AssertionError('scalar fallback')):
        result = verification._file_revisions(str(source), [*names, *links, 'package/index.js'])
    assert result == {**dict.fromkeys(names, 'missing'), **dict.fromkeys(links, ''),
                      'package/index.js': hashlib.sha256(b'new package bytes').hexdigest()}
    assert operations.count('resolve_files') == operations.count('digest_batch') == 1
    assert 'resolve' not in operations and 'stat' not in operations


def test_component_sources_batch_regular_hidden_missing_and_alias_paths(bwrap, tmp_path):
    from scripts.llm_solver.harness.operation_timing import operation_profile
    from scripts.llm_solver.harness._guardrails.verification import _safe_source_paths
    from scripts.llm_solver.harness.sandbox.ignore_policy import activate_ignore_policy, load_ignore_policy
    from scripts.llm_solver.harness.task_path import TaskPath
    source, files = namespace_files(bwrap, tmp_path)
    names = [f'generated-{i}.js' for i in range(1000)]
    for name in names:
        (source / name).write_bytes(b'source')
    (source / 'hidden.js').write_bytes(b'hidden')
    (source / '.yujignore').write_text('hidden.js\n')
    (source / 'alias.js').symlink_to(names[0])
    (source / 'escape.js').symlink_to(tmp_path / 'outside.js')
    (tmp_path / 'outside.js').write_bytes(b'outside')
    operations = []
    run = files.run
    files.run = lambda script, args, data: (operations.append(args[3] if len(args) > 3 else ''),
                                           run(script, args, data))[1]
    with activate_task_files(files, host_root=source):
        policy = load_ignore_policy(source)
        with activate_ignore_policy(policy), operation_profile() as profile:
            operations.clear()
            paths = _safe_source_paths(TaskPath(files, files.root), tuple([
                *names, 'missing.js', 'alias.js', 'escape.js', 'hidden.js',
                str(tmp_path / 'outside.js'), str(source / names[0]),
            ]))
            timings = profile.take()
    assert [p.relative_to(files.root).as_posix() for p in paths] == names
    # The first large call also discovers optional bulk tools in this namespace.
    assert operations == ['', 'resolve_files']
    resolution, = [row for row in timings if row['operation'][-1] == 'files.resolve_files']
    assert resolution['calls'] == 1
    assert resolution['operation'] == ['verification.source_paths', 'files.resolve_files']


def test_file_state_batch_keeps_denials_loops_and_hidden_entries_distinct(bwrap, tmp_path):
    from scripts.llm_solver.harness._guardrails import verification
    from scripts.llm_solver.harness.sandbox.ignore_policy import activate_ignore_policy, load_ignore_policy
    source, files = namespace_files(bwrap, tmp_path)
    (source / 'file').write_bytes(b'native bytes')
    (source / 'unreadable').write_bytes(b'denied')
    (source / 'unreadable').chmod(0)
    (source / 'closed').mkdir(mode=0)
    (source / 'loop').symlink_to('loop')
    (source / 'dangling').symlink_to('missing')
    (source / 'outside').symlink_to(tmp_path / 'outside')
    (tmp_path / 'outside').write_bytes(b'outside bytes')
    (source / 'hidden').write_bytes(b'hidden bytes')
    (source / 'hidden').chmod(0)
    (source / 'hidden-directory').mkdir()
    (source / '.yujignore').write_text('hidden\nhidden-directory/\n')
    names = ['file', 'unreadable', 'closed/missing', 'loop', 'dangling', 'outside',
             'hidden', 'hidden-directory']
    try:
        assert len(files.resolve_file_states(names)) == len(names)
        with activate_task_files(files, host_root=source):
            policy = load_ignore_policy(source)
            with activate_ignore_policy(policy):
                expected = {name: verification._file_revision(str(source), name) for name in names}
                with patch.object(verification, '_file_revision', side_effect=AssertionError('scalar fallback')):
                    assert verification._file_revisions(str(source), names) == expected
    finally:
        (source / 'closed').chmod(0o700)
    # The hidden-directory mask walk encounters the denied parent. Its result
    # stays unknown without discarding unrelated entries from the batch.
    assert expected == {'file': hashlib.sha256(b'native bytes').hexdigest(),
        'unreadable': '', 'closed/missing': '', 'loop': '', 'dangling': 'missing',
        'outside': '', 'hidden': 'missing', 'hidden-directory': ''}


def test_one_file_removed_during_hashing_does_not_discard_the_batch(bwrap, tmp_path):
    from scripts.llm_solver.harness._guardrails import verification
    source, files = namespace_files(bwrap, tmp_path)
    names = [f'file-{i}.py' for i in range(128)]
    for name in names:
        (source / name).write_bytes(name.encode())
    original_digest = files._sha256_resolved_many
    removed = False
    def digest(paths):
        nonlocal removed
        if not removed:
            (source / names[57]).unlink()
            removed = True
        return original_digest(paths)
    with activate_task_files(files, host_root=source), patch.object(
        files, '_sha256_resolved_many', digest,
    ), patch.object(
        verification, '_file_revision', wraps=verification._file_revision) as scalar:
        result = verification._file_revisions(str(source), names)
    assert result == {name: ('missing' if name == names[57] else hashlib.sha256(name.encode()).hexdigest())
                      for name in names}
    assert scalar.call_count == 1


def test_failed_resolved_digest_retries_established_canonical_hashing(bwrap, tmp_path):
    from scripts.llm_solver.harness._guardrails import verification
    from scripts.llm_solver.harness.task_files import TaskFileError

    source, files = namespace_files(bwrap, tmp_path)
    names = ['first.py', 'second.py']
    for name in names:
        (source / name).write_bytes(name.encode())
    with activate_task_files(files, host_root=source), patch.object(
        files, '_sha256_resolved_many', side_effect=TaskFileError('candidate failed'),
    ), patch.object(files, 'sha256_many', wraps=files.sha256_many) as established:
        result = verification._file_revisions(str(source), names)
    assert result == {
        name: hashlib.sha256(name.encode()).hexdigest() for name in names
    }
    established.assert_called_once()


def test_partial_resolved_digest_output_is_discarded_before_canonical_retry(bwrap, tmp_path):
    from scripts.llm_solver.harness._guardrails import verification

    source, files = namespace_files(bwrap, tmp_path)
    names = [f'file-{index}.py' for index in range(270)]
    for name in names:
        (source / name).write_bytes(name.encode())
    native = files.run
    candidate_calls = 0
    marker = '        (( ${#batch[@]} )) || break\n'

    def fail_second_candidate_batch(script, args, data):
        nonlocal candidate_calls
        if len(args) > 7 and args[3] == 'digest_batch' and args[-1] == 'resolved':
            candidate_calls += 1
            assert script.count(marker) == 1
            script = 'yuj_test_batch=0\n' + script.replace(
                marker,
                marker + '        (( ++yuj_test_batch == 2 )) && return 74\n',
            )
        return native(script, args, data)

    files.run = fail_second_candidate_batch
    with activate_task_files(files, host_root=source), patch.object(
        files, 'sha256_many', wraps=files.sha256_many,
    ) as established:
        result = verification._file_revisions(str(source), names)
    assert result == {
        name: hashlib.sha256(name.encode()).hexdigest() for name in names
    }
    assert candidate_calls == 1
    established.assert_called_once()


@pytest.mark.parametrize('spelling', ['relative', 'host', 'native'])
def test_partial_complete_and_changed_native_revision(rig, spelling):
    root, files, overlay, cfg, state = rig
    target = {'relative': 'nested/cases.py', 'host': str(root / 'nested/cases.py'),
              'native': str(files.root / 'nested/cases.py')}[spelling]
    with activate_task_files(files, host_root=root), task_file_scope(root, cfg):
        result, metadata = observe(root, cfg, state, path=target, limit=1)
        assert 'native first' in result and 'HIDDEN_HOST_COPY' not in result
        assert metadata['inspection_evidence']['namespace'] == 'task_execution'
        receipt = request(root)
        assert receipt['workspace_namespace'] == 'task_execution'
        coverage = _workspace_coverage(state, cfg, str(root), receipt, target)
        assert 'partial recorded excerpt: lines 1 of 2' in coverage[1]
        assert not coverage[3]
        observe(root, cfg, state, path=target, offset=1)
        assert _workspace_coverage(state, cfg, str(root), receipt, target)[3]
        (overlay / 'cases.py').write_text('changed native file\n')
        coverage = _workspace_coverage(state, cfg, str(root), receipt, target)
        assert 'different file revision' in coverage[1]
        assert coverage[2] == hashlib.sha256(b'changed native file\n').hexdigest()


@pytest.mark.parametrize('change', ['missing', 'denied', 'policy'])
def test_unavailable_native_file_never_hashes_host_copy(rig, change, monkeypatch):
    root, files, overlay, cfg, state = rig
    original = Path.open

    def refuse_host(path, *args, **kwargs):
        if path == root / 'nested/cases.py':
            raise AssertionError('inspection used a hidden host file')
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', refuse_host)
    with activate_task_files(files, host_root=root), task_file_scope(root, cfg):
        observe(root, cfg, state, path='nested/cases.py')
        receipt = request(root)
        if change == 'missing':
            (overlay / 'cases.py').unlink()
        elif change == 'denied':
            (overlay / 'cases.py').chmod(0)
        else:
            cfg = replace(cfg, unreadable_paths=(str(files.root / 'nested/cases.py'),))
        coverage = _workspace_coverage(state, cfg, str(root), receipt, 'nested/cases.py')
        assert coverage[0] == 'unknown' and not coverage[3]
        assert not coverage[2]


def test_native_receipt_cannot_be_used_after_leaving_its_view(rig):
    root, files, _, cfg, state = rig
    with activate_task_files(files, host_root=root), task_file_scope(root, cfg):
        observe(root, cfg, state, path='nested/cases.py')
        receipt = request(root)
    assert 'namespace differs' in _workspace_coverage(
        state, cfg, str(root), receipt, 'nested/cases.py')[1]


@pytest.mark.parametrize('retarget_at', ['before_request', 'after_request'])
def test_inspection_receipt_keeps_captured_root_after_alias_retargets(rig, tmp_path, retarget_at):
    root, files, _, cfg, state = rig
    alias = tmp_path / 'alias'
    alias.symlink_to(root, target_is_directory=True)
    other = tmp_path / 'other'
    other.mkdir()
    with activate_task_files(files, host_root=alias):
        observe(alias, cfg, state, path='nested/cases.py')
        receipt = request(alias) if retarget_at == 'after_request' else None
        alias.unlink()
        alias.symlink_to(other, target_is_directory=True)
        if receipt is None:
            receipt = request(alias)
        assert receipt['workspace_cwd'] == str(root)
        coverage = _workspace_coverage(state, cfg, str(alias), receipt, 'nested/cases.py')
        assert coverage[3], coverage


@pytest.mark.parametrize('retarget_at', ['before_request', 'after_request'])
def test_custom_command_receipt_keeps_captured_root(rig, tmp_path, retarget_at):
    import shlex
    import sys
    from types import SimpleNamespace
    from scripts.llm_solver.harness.runner_invocations import describe_shell_submission
    from scripts.llm_solver.harness._guardrails.custom_execution import (
        completed_custom_execution, observed_component_runner_base_cmd,
    )

    root, files, _, cfg, _ = rig
    alias = tmp_path / 'alias'
    alias.symlink_to(root, target_is_directory=True)
    other = tmp_path / 'other'
    other.mkdir()
    command = shlex.join([sys.executable, '-c', 'assert True'])
    with activate_task_files(files, host_root=alias):
        submission = (describe_shell_submission(command, alias)
                      if retarget_at == 'after_request' else None)
        alias.unlink()
        alias.symlink_to(other, target_is_directory=True)
        if submission is None:
            submission = describe_shell_submission(command, alias)
        assert submission['task_cwd'] == str(root)
        metadata = dict(executed=True, exit_status_known=True, exit_status=0,
                        verification_status='custom_passed', shell_submission=submission)
        assert completed_custom_execution(metadata, alias)
        state = SimpleNamespace(post_mutation_observed_runtime_executable=sys.executable,
                                post_mutation_observed_runtime_family='python',
                                post_mutation_observed_runtime_binding=submission)
        assert observed_component_runner_base_cmd(state, 'pytest', cwd=alias, cfg=cfg).startswith(
            shlex.quote(sys.executable))


@pytest.mark.parametrize('selection_root', ['original', 'replacement'])
def test_runner_selection_is_bound_to_native_task_after_alias_retargets(rig, tmp_path, selection_root):
    root, files, _, cfg, _ = rig
    alias = tmp_path / 'alias'
    alias.symlink_to(root, target_is_directory=True)
    other = tmp_path / 'other'
    other.mkdir()
    cfg = replace(cfg, tools_run_tests_enabled=True, analysis_task_format='pytest',
                  runtime_test_selection={
                      'status': 'selected',
                      'task_root': str(root if selection_root == 'original' else other),
                      'selected': {'runner': 'pytest', 'base_cmd': 'observed-check'},
                  })
    with activate_task_files(files, host_root=alias):
        alias.unlink()
        alias.symlink_to(other, target_is_directory=True)
        with patch('scripts.llm_solver.harness.tools._run_in_sandbox', return_value=('ok', 0, False)) as execute:
            result = dispatch('run_tests', {'path': 'nested/cases.py'}, cwd=str(alias), cfg=cfg,
                              effective_env={'PATH': '/usr/bin:/bin'})
        if selection_root == 'original':
            assert execute.call_count == 1, result
            assert execute.call_args.args[0] == 'observed-check nested/cases.py'
        else:
            execute.assert_not_called()
            assert 'selection_unresolved' in result


def test_startup_runner_selection_records_captured_root(rig, tmp_path):
    from scripts.llm_solver.harness._loop._driver_setup import resolve_task_format

    root, files, _, cfg, _ = rig
    alias = tmp_path / 'alias'
    alias.symlink_to(root, target_is_directory=True)
    other = tmp_path / 'other'
    other.mkdir()
    selection = {'status': 'selected',
                 'selected': {'runner': 'pytest', 'base_cmd': 'observed-check'}}
    with activate_task_files(files, host_root=alias):
        alias.unlink()
        alias.symlink_to(other, target_is_directory=True)
        resolved = resolve_task_format(
            replace(cfg, analysis_task_format='pytest'), alias,
            runtime_observations={'runner_selection': selection})
        assert resolved.runtime_test_selection == {**selection, 'task_root': str(root)}


def test_same_bytes_do_not_merge_excerpts_from_different_bindings(rig):
    root, files, _, cfg, state = rig
    with activate_task_files(files, host_root=root), task_file_scope(root, cfg):
        observe(root, cfg, state, path='nested/cases.py', limit=1)
        old_request = request(root)
        files.binding = {**files.binding, 'fixture_selection': 'another'}
        assert 'namespace differs' in _workspace_coverage(
            state, cfg, str(root), old_request, 'nested/cases.py')[1]
        observe(root, cfg, state, path='nested/cases.py', offset=1)
        coverage = _workspace_coverage(state, cfg, str(root), request(root), 'nested/cases.py')
        assert 'partial recorded excerpt: lines 2 of 2' in coverage[1]
        assert not coverage[3]


def test_direct_and_structured_runner_tools_record_the_native_dispatch_view(rig):
    root, files, _, cfg, _ = rig
    cfg = replace(cfg, tools_run_tests_enabled=True, analysis_task_format='pytest')
    with activate_task_files(files, host_root=root), task_file_scope(root, cfg):
        for tool, args in [('bash', {'cmd': 'pytest nested/cases.py'}),
                           ('run_tests', {'path': 'nested/cases.py'})]:
            metadata = {}
            with patch('scripts.llm_solver.harness.tools._run_in_sandbox', return_value=('ok', 0, False)):
                dispatch(tool, args, cwd=str(root), cfg=cfg, execution_metadata=metadata)
            assert metadata['runner_request']['workspace_namespace'] == 'task_execution'
            assert metadata['runner_request']['workspace_task_view'] == files.binding


def test_native_partial_excerpt_advice_reaches_the_next_scripted_request(rig, monkeypatch):
    import io
    import json
    from unittest.mock import MagicMock
    from scripts.llm_solver.harness import task_file_runtime
    from scripts.llm_solver.harness.loop import Session
    from scripts.llm_solver.server.types import ToolCall, TurnResult, Usage
    root, files, _, cfg, _ = rig
    cfg = replace(cfg, max_turns=3, loop_detect_enabled=False, duplicate_guard_enabled=False)
    monkeypatch.setattr(task_file_runtime, 'make_task_files', lambda *args, **kwargs: files)
    client = MagicMock()
    calls = [[ToolCall('read', 'read', {'path': 'nested/cases.py', 'limit': 1})],
             [ToolCall('check', 'bash', {'cmd': 'pytest nested/cases.py'})], []]
    client.chat.side_effect = [TurnResult(content=None, tool_calls=tools,
        finish_reason='tool_calls' if tools else 'stop',
        usage=Usage(prompt_tokens=10, completion_tokens=5)) for tools in calls]
    client.build_assistant_message.return_value = {'role': 'assistant', 'content': None}
    trace = io.StringIO()
    with patch('scripts.llm_solver.harness.tools._run_in_sandbox', return_value=('ok', 0, False)):
        Session(cfg, client, 'system', 'task', str(root), trace_file=trace).run()
    request_text = str(client.chat.call_args_list[-1])
    assert 'partial recorded excerpt: lines 1 of 2' in request_text, json.dumps(
        client.chat.call_args_list[-1].args[0], ensure_ascii=False)
    assert 'HIDDEN_HOST_COPY' not in request_text
    rows = [json.loads(line) for line in trace.getvalue().splitlines()]
    assert any(row.get('inspection_evidence', {}).get('namespace') == 'task_execution'
               for row in rows)
