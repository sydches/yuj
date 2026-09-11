"""Streamed rule matching uses the task's selected filenames."""
import json
from types import SimpleNamespace

import pytest

from tests.test_task_files import bwrap, namespace_files
from tests.test_stream_rules import _finish, _rule
from scripts.llm_solver.harness.stream_rules import StreamRuleRuntime
from scripts.llm_solver.harness.task_path import activate_task_files


@pytest.mark.parametrize('spelling', ['host', 'native'])
@pytest.mark.parametrize('retarget_alias', [False, True])
def test_stream_rule_matches_native_document_despite_hidden_host_symlink(
        bwrap, tmp_path, spelling, retarget_alias):
    source, files = namespace_files(bwrap, tmp_path)
    (source / 'app.py').write_text('native document')
    host = tmp_path / 'host'
    host.mkdir()
    outside = tmp_path / 'outside'
    outside.write_text('hidden host contents')
    (host / 'app.py').symlink_to(outside)
    alias = tmp_path / 'alias'
    alias.symlink_to(host, target_is_directory=True)
    rule = _rule('condition = "forbidden"\nscope = "tool:write(**/*.py)"\n'
                 'interruptMode = "never"')
    runtime = StreamRuleRuntime([rule], repeat_gap=10, cwd=alias)
    requested = str((alias if spelling == 'host' else files.root) / 'app.py')
    with activate_task_files(files, host_root=alias):
        if retarget_alias:
            alias.unlink()
            alias.symlink_to(source, target_is_directory=True)
        runtime.begin_attempt()
        runtime.observe(SimpleNamespace(
            source='tool', delta='', tool_index=0, tool_name='write',
            tool_arguments=json.dumps({'path': requested, 'content': 'forbidden'}),
        ), turn=0)
        records = _finish(runtime)
    assert len(records) == 1
    assert records[0]['path'] == 'app.py'
    assert records[0]['scope'] == 'tool:write(**/*.py)'


@pytest.mark.parametrize('failure', ['escape', 'unavailable', 'budget'])
def test_stream_path_does_not_fall_back_after_native_resolution_failure(
        bwrap, tmp_path, monkeypatch, failure):
    from scripts.llm_solver.harness._stream_rule_runtime import _normalize_path
    from scripts.llm_solver.harness.task_files import TaskFileError
    from scripts.llm_solver.harness.time_budget import BudgetExhausted

    source, files = namespace_files(bwrap, tmp_path)
    host = tmp_path / 'host'
    host.mkdir()
    (host / 'app.py').write_text('hidden host file')
    outside = tmp_path / 'outside'
    outside.write_text('outside')
    (source / 'app.py').symlink_to(outside)
    if failure != 'escape':
        def refuse(path):
            error = BudgetExhausted if failure == 'budget' else TaskFileError
            raise error('native resolution unavailable')
        monkeypatch.setattr(files, 'resolve', refuse)
    with activate_task_files(files, host_root=host):
        if failure == 'budget':
            with pytest.raises(BudgetExhausted):
                _normalize_path(str(host / 'app.py'), host)
        else:
            assert _normalize_path(str(host / 'app.py'), host) == ''
