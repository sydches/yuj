"""Repository maps describe the selected task filesystem."""
import os
from pathlib import Path

import pytest

from tests.test_task_files import bwrap, namespace_files
from scripts.llm_solver.harness.repo_map import build_repo_map
from scripts.llm_solver.harness.task_path import TaskPath, activate_task_files


@pytest.mark.parametrize('explicit_root', [False, True])
def test_map_reads_nested_readonly_overlay(bwrap, tmp_path, explicit_root):
    overlay = tmp_path / 'overlay'
    overlay.mkdir()
    (overlay / 'module.py').write_text('def selected_definition():\n    pass\n')
    source, files = namespace_files(bwrap, tmp_path, overlay=overlay, readonly=True)
    (source / 'nested/module.py').write_text('def hidden_host_definition():\n    pass\n')
    (source / 'nested/host_only.py').write_text('def hidden_host_only():\n    pass\n')
    with activate_task_files(files, host_root=source):
        result = build_repo_map(
            TaskPath(files, files.root) if explicit_root else source,
            task_message='Inspect nested/module.py', token_budget=1000,
            refresh='always', cache_dir=tmp_path / 'private_cache',
        )
    assert 'selected_definition' in result.content
    assert 'hidden_host' not in result.content
    assert result.files == 1
    assert (tmp_path / 'private_cache/symbols.v1.json').is_file()
    assert not (source / 'private_cache').exists()
    assert 'hidden_host_definition' in (source / 'nested/module.py').read_text()


def test_disabled_map_does_not_read_the_task(tmp_path, monkeypatch):
    from scripts.llm_solver.harness import repo_map

    def forbidden(*args, **kwargs):
        raise AssertionError('disabled map must not open the task')

    monkeypatch.setattr(repo_map, 'StructuralIndex', forbidden)
    assert not build_repo_map(tmp_path / 'absent', task_message='', token_budget=0).content


@pytest.mark.parametrize('refresh', ['auto', 'manual'])
def test_native_map_rejects_host_cache_with_the_same_root_name(bwrap, tmp_path, refresh):
    overlay = tmp_path / 'overlay'
    overlay.mkdir()
    selected = overlay / 'module.py'
    selected.write_text('def native_symbol():\n    pass\n')
    source, files = namespace_files(bwrap, tmp_path, overlay=overlay)
    host_root = Path(str(files.root))
    host = host_root / 'nested/module.py'
    host.parent.mkdir()
    host.write_text('def hidden_symbol():\n    pass\n')
    stamp = host.stat()
    os.utime(selected, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    arguments = dict(task_message='Inspect nested/module.py', token_budget=1000,
                     refresh=refresh, cache_dir=tmp_path / 'private_cache')
    assert 'hidden_symbol' in build_repo_map(host_root, **arguments).content
    with activate_task_files(files, host_root=source):
        result = build_repo_map(source, **arguments)
    assert not result.cache_hit
    assert 'native_symbol' in result.content and 'hidden_symbol' not in result.content


@pytest.mark.parametrize('refresh', ['auto', 'files', 'manual'])
def test_cache_rejects_replaced_native_mount(bwrap, tmp_path, refresh):
    overlay = tmp_path / 'overlay'
    overlay.mkdir()
    selected = overlay / 'module.py'
    selected.write_text('def former_symbol():\n    pass\n')
    source, files = namespace_files(bwrap, tmp_path, overlay=overlay)
    arguments = dict(task_message='Inspect nested/module.py', token_budget=1000,
                     refresh=refresh, cache_dir=tmp_path / 'private_cache')
    with activate_task_files(files, host_root=source):
        assert 'former_symbol' in build_repo_map(source, **arguments).content
        assert build_repo_map(source, **arguments).cache_hit
        stamp = selected.stat()
        overlay.rename(tmp_path / 'former-overlay')
        overlay.mkdir()
        selected.write_text('def native_symbol():\n    pass\n')
        os.utime(selected, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        result = build_repo_map(source, **arguments)
    assert not result.cache_hit
    assert 'native_symbol' in result.content and 'former_symbol' not in result.content


def test_manual_cache_preserves_edits_in_the_same_native_view(bwrap, tmp_path):
    from scripts.llm_solver.harness.task_files import NamespaceFiles
    source, files = namespace_files(bwrap, tmp_path)
    selected = source / 'module.py'
    selected.write_text('def former_symbol():\n    pass\n')
    arguments = dict(task_message='Inspect module.py', token_budget=1000,
                     refresh='manual', cache_dir=tmp_path / 'private_cache')
    with activate_task_files(files, host_root=source):
        assert not build_repo_map(source, **arguments).cache_hit
    selected.write_text('def edited_symbol():\n    pass\n')
    (source / 'added.py').write_text('def added_symbol():\n    pass\n')
    # A new access object and a new bwrap process still observe the same view.
    resumed = NamespaceFiles(str(files.root), files.run, binding=files.binding,
                             shares_host_kernel=True)
    with activate_task_files(resumed, host_root=source):
        result = build_repo_map(source, **arguments)
    assert result.cache_hit and 'former_symbol' in result.content
    assert 'edited_symbol' not in result.content
    assert 'added_symbol' not in result.content


def test_view_cache_handles_literal_unicode_and_newline_mount_paths(bwrap, tmp_path):
    base = tmp_path / 'literal space Ω\nroot'
    base.mkdir()
    overlay = base / 'overlay Ω\n'
    overlay.mkdir()
    (overlay / 'module.py').write_text('def native_symbol():\n    pass\n')
    source, files = namespace_files(bwrap, base, overlay=overlay)
    arguments = dict(task_message='Inspect nested/module.py', token_budget=1000,
                     refresh='manual', cache_dir=tmp_path / 'private_cache')
    with activate_task_files(files, host_root=source):
        assert 'native_symbol' in build_repo_map(source, **arguments).content
        assert build_repo_map(source, **arguments).cache_hit


def test_manual_cache_reuses_the_production_bwrap_view(bwrap, tmp_path, monkeypatch):
    from tests._config_helpers import make_config
    from scripts.llm_solver.harness.task_file_runtime import task_file_scope
    monkeypatch.delenv('YUJ_CONTAINER', raising=False)
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'module.py').write_text('def native_symbol():\n    pass\n')
    cfg = make_config(sandbox_bash=True, bwrap_bin=bwrap)
    arguments = dict(task_message='Inspect module.py', token_budget=1000,
                     refresh='manual', cache_dir=tmp_path / 'private_cache')
    environment = {'PATH': os.defpath, 'HOME': str(source)}
    with task_file_scope(source, cfg, environment=environment):
        assert not build_repo_map(source, **arguments).cache_hit
    with task_file_scope(source, cfg, environment=environment):
        result = build_repo_map(source, **arguments)
    assert result.cache_hit and 'native_symbol' in result.content


@pytest.mark.parametrize('refresh', ['auto', 'manual'])
def test_cache_cannot_restore_a_now_unreadable_symbol(bwrap, tmp_path, refresh):
    source, files = namespace_files(bwrap, tmp_path)
    selected = source / 'module.py'
    selected.write_text('def denied_symbol():\n    pass\n')
    arguments = dict(task_message='Inspect module.py', token_budget=1000,
                     refresh=refresh, cache_dir=tmp_path / 'private_cache')
    with activate_task_files(files, host_root=source):
        assert 'denied_symbol' in build_repo_map(source, **arguments).content
        selected.chmod(0)
        try:
            result = build_repo_map(source, **arguments)
        finally:
            selected.chmod(0o600)
    assert not result.cache_hit and 'denied_symbol' not in result.content


def test_cache_rechecks_policy_without_an_os_denial_mount(bwrap, tmp_path):
    from scripts.llm_solver.harness.sandbox.ignore_policy import (
        activate_ignore_policy, load_ignore_policy,
    )
    source, files = namespace_files(bwrap, tmp_path)
    (source / 'module.py').write_text('def denied_symbol():\n    pass\n')
    arguments = dict(task_message='Inspect module.py', token_budget=1000,
                     refresh='manual', cache_dir=tmp_path / 'private_cache')
    with activate_task_files(files, host_root=source):
        assert 'denied_symbol' in build_repo_map(source, **arguments).content
        (source / '.yujignore').write_text('module.py\n')
        with activate_ignore_policy(load_ignore_policy(source)):
            result = build_repo_map(source, **arguments)
    assert not result.cache_hit and 'denied_symbol' not in result.content


def test_changed_view_during_scan_does_not_publish_or_cache_rows(bwrap, tmp_path):
    from scripts.llm_solver.harness.task_files import TaskFileError
    from tests.test_structural_index import _RecordingExtractor
    overlay = tmp_path / 'overlay'
    overlay.mkdir()
    (overlay / 'module.fixture').write_text('DEF former_symbol\n')
    source, files = namespace_files(bwrap, tmp_path, overlay=overlay)

    class ReplacingExtractor(_RecordingExtractor):
        def extract(self, source, **kwargs):
            result = super().extract(source, **kwargs)
            overlay.rename(tmp_path / 'former-overlay')
            overlay.mkdir()
            (overlay / 'module.fixture').write_text('DEF native_symbol\n')
            return result

    cache = tmp_path / 'private_cache'
    with activate_task_files(files, host_root=source):
        with pytest.raises(TaskFileError, match='mount view changed'):
            build_repo_map(source, task_message='Inspect module', token_budget=1000,
                           refresh='always', extractor=ReplacingExtractor(), cache_dir=cache)
    assert not (cache / 'symbols.v1.json').exists()


def test_driver_builds_map_inside_task_scope(bwrap, tmp_path, monkeypatch):
    from contextlib import contextmanager
    from unittest.mock import MagicMock
    from tests._config_helpers import make_config
    from tests.test_repo_map import _done_turn
    from scripts.llm_solver.harness import loop, task_file_runtime
    from scripts.llm_solver.harness.context import FullTranscript

    overlay = tmp_path / 'overlay'
    overlay.mkdir()
    (overlay / 'module.py').write_text('def selected_definition():\n    pass\n')
    source, files = namespace_files(bwrap, tmp_path, overlay=overlay, readonly=True)
    (source / 'nested/module.py').write_text('def hidden_host_definition():\n    pass\n')
    entered = []
    original_scope = task_file_runtime.task_file_scope

    @contextmanager
    def scope(cwd, cfg, **options):
        assert str(cwd) == str(source)
        assert options['environment'] is not None
        if options.get('ignore_policy') is None:
            with original_scope(cwd, cfg, **options) as result:
                yield result
            return
        entered.append(True)
        with activate_task_files(files, host_root=source):
            yield files

    monkeypatch.setattr(task_file_runtime, 'task_file_scope', scope)
    monkeypatch.setattr(loop, '_auto_commit', lambda *args, **kwargs: None)
    monkeypatch.setattr(loop.Session, '_get_server_ctx', lambda self: 0)
    client = MagicMock()
    client.chat.return_value = _done_turn()
    client.build_assistant_message.return_value = {'role': 'assistant', 'content': 'Done.'}
    cfg = make_config(max_sessions=1, max_turns=1, repo_map_tokens=1000,
                      repo_map_refresh='always')
    assert loop.solve_task(
        source, cfg, client, context_class=FullTranscript,
        initial_prompt='Inspect nested/module.py', artifacts_dir=tmp_path / 'records',
    )
    assert entered
    messages = client.chat.call_args.args[0]
    task = next(message['content'] for message in messages if message['role'] == 'user')
    assert 'selected_definition' in task and 'hidden_host_definition' not in task
