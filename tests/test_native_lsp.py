"""Language servers and document sync must see the same mounted task."""
import json
import os
import shutil
import sys
import subprocess
from contextlib import nullcontext

import pytest

from tests.test_task_files import bwrap, namespace_files
from scripts.llm_solver.harness.lsp_support import LspManager, LspServerSpec
from scripts.llm_solver.harness.task_path import activate_task_files


SERVER = r'''
import json
from pathlib import Path
import sys
from urllib.parse import unquote, urlparse
root = None
document = None
while True:
    headers = {}
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            raise SystemExit
        if line in (b'\r\n', b'\n'):
            break
        name, value = line.decode().split(':', 1)
        headers[name.lower()] = value.strip()
    request = json.loads(sys.stdin.buffer.read(int(headers['content-length'])))
    method = request['method']
    result = None
    if method == 'initialize':
        root = request['params']
        result = {'capabilities': {}}
    elif method == 'textDocument/didOpen':
        document = request['params']['textDocument']
    elif method == 'textDocument/didChange':
        document['text'] = request['params']['contentChanges'][0]['text']
    elif method == 'textDocument/documentSymbol':
        uri = request['params']['textDocument']['uri']
        result = {'root': root['rootUri'], 'process_id': root['processId'], 'uri': uri,
                  'synced': document['text'], 'native': Path(unquote(urlparse(uri).path)).read_text()}
    elif method == 'exit':
        raise SystemExit
    if 'id' in request:
        payload = json.dumps({'jsonrpc': '2.0', 'id': request['id'], 'result': result}).encode()
        sys.stdout.buffer.write(b'Content-Length: ' + str(len(payload)).encode() + b'\r\n\r\n' + payload)
        sys.stdout.buffer.flush()
'''


@pytest.mark.parametrize('later_scope', ['original', 'none', 'different', 'retargeted_alias'])
def test_lsp_root_uri_and_document_sync_use_nested_native_overlay(bwrap, tmp_path, later_scope):
    overlay = tmp_path / 'overlay'
    overlay.mkdir()
    name = 'native # document.py'
    (overlay / name).write_text('native selected contents')
    (overlay / 'pyproject.toml').write_text('[project]\n')
    source, files = namespace_files(bwrap, tmp_path, overlay=overlay)
    alias = tmp_path / 'host-alias'
    alias.symlink_to(source, target_is_directory=True)
    (source / 'nested' / name).write_text('hidden host contents')
    server = tmp_path / 'server.py'
    server.write_text(SERVER)
    spec = LspServerSpec('python', (sys.executable, '-u', str(server)), ('.py',), ('pyproject.toml',))
    diagnostics = tmp_path / 'server.stderr'
    diagnostic_stream = diagnostics.open('wb')
    with activate_task_files(files, host_root=alias):
        manager = LspManager(
            cwd=alias, servers=(spec,), tool_enabled=True,
            diagnostics_timeout_s=0,
            popen_factory=lambda *args, **kwargs: subprocess.Popen(
                *args, **{**kwargs, 'stderr': diagnostic_stream}),
            argv_builder=lambda spec, root: [
                bwrap, '--ro-bind', '/', '/', '--dev', '/dev', '--unshare-net',
                '--ro-bind', str(source), str(files.root),
                '--ro-bind', str(overlay), str(files.root / 'nested'), '--', *spec.command],
        )
    other_dir = tmp_path / 'other'
    other_dir.mkdir()
    other_source, other_files = namespace_files(bwrap, other_dir)
    (other_source / 'nested').mkdir()
    (other_source / 'nested' / name).write_text('different selected contents')
    if later_scope == 'retargeted_alias':
        alias.unlink()
        alias.symlink_to(other_source, target_is_directory=True)
    later = (nullcontext() if later_scope in ('none', 'retargeted_alias') else activate_task_files(
        files if later_scope == 'original' else other_files, host_root=source))
    try:
        with later:
            result = manager.query('symbols', path='nested/' + name)
            assert result.status == 'ok', diagnostics.read_text()
            value = json.loads(result.result)
            assert value == {
                'root': (files.root / 'nested').as_uri(), 'process_id': None,
                'uri': (files.root / 'nested' / name).as_uri(),
                'synced': 'native selected contents', 'native': 'native selected contents',
            }
            for requested in (str(alias / 'nested' / name), str(source / 'nested' / name),
                              str(files.root / 'nested' / name)):
                assert json.loads(manager.query('symbols', path=requested).result) == value
            (overlay / name).write_text('edited native contents')
            manager.after_edit(str(alias / 'nested' / name))
            value = json.loads(manager.query('symbols', path='nested/' + name).result)
            assert value['synced'] == value['native'] == 'edited native contents'
    finally:
        manager.close()
        diagnostic_stream.close()
    assert (source / 'nested' / name).read_text() == 'hidden host contents'


def test_standalone_sandboxed_lsp_finds_root_through_its_masks(bwrap, tmp_path):
    task = tmp_path / 'task'
    nested = task / 'nested'
    nested.mkdir(parents=True)
    hidden = nested / '.project'
    hidden.mkdir()
    marker = hidden / 'pyproject.toml'
    marker.write_text('[project]\n')
    document = nested / 'app.py'
    document.write_text('permitted native contents')
    secret = nested / 'secret.py'
    secret.write_text('host bytes must not be synced')
    python = shutil.which('python3', path=os.defpath)
    if python is None:
        pytest.skip('system Python is unavailable')
    spec = LspServerSpec('python', (python, '-u', '-c', SERVER),
                         ('.py',), ('.project/pyproject.toml',))
    manager = LspManager.sandboxed(
        cwd=task, servers=(spec,), bwrap_bin=bwrap, tool_enabled=True,
        unreadable_paths=(str(hidden), str(secret)),
        effective_env={'PATH': os.environ['PATH']},
    )
    try:
        result = manager.query('symbols', path='nested/app.py')
        assert result.status == 'ok'
        value = json.loads(result.result)
        assert value['synced'] == value['native'] == 'permitted native contents'
        assert value['root'] == task.as_uri()
        with pytest.raises(OSError):
            manager.query('symbols', path='nested/secret.py')
        assert all(secret.as_uri() not in state.versions for state in manager._states.values())
    finally:
        manager.close()


def test_lsp_locations_keep_visible_dependencies_and_reject_hidden_host_files(bwrap, tmp_path):
    task = tmp_path / 'task'
    task.mkdir()
    local = task / 'local.py'
    local.write_text('task')
    hidden = task / 'hidden.py'
    hidden.write_text('hidden')
    outside = tmp_path / 'unrelated.py'
    outside.write_text('unrelated host project')
    dependency = tmp_path / 'dependency' / 'installed.py'
    dependency.parent.mkdir()
    dependency.write_text('explicitly admitted runtime resource')
    manager = LspManager.sandboxed(
        cwd=task, servers=(), bwrap_bin=bwrap,
        unreadable_paths=(str(hidden),), readable_paths=(str(dependency.parent),),
        effective_env={'PATH': os.environ['PATH']},
    )
    manager._task_files._utility('realpath')
    operations = []
    run = manager._task_files.run
    manager._task_files.run = lambda script, args, data: (operations.append((args, data)), run(script, args, data))[1]
    valid = [{'uri': local.as_uri()}, {'uri': dependency.as_uri()}, {'uri': local.as_uri()}]
    response, omitted = manager._visible_locations(
        [*valid, {'uri': hidden.as_uri()}, {'uri': outside.as_uri()}], None)
    assert omitted and response == valid
    assert len(operations) == 1
    assert len(operations[0][0]) == 1  # Only the utility path is an exec argument.
    assert len(operations[0][1].split(b'\0')) == 5  # Duplicate URIs are checked once.


def test_native_lsp_ignores_canonical_targets_without_requiring_mount_masks(bwrap, tmp_path):
    from scripts.llm_solver.harness.sandbox.ignore_policy import load_ignore_policy
    (tmp_path / '.yujignore').write_text('hidden.py\n')
    (tmp_path / 'hidden.py').write_text('hidden')
    (tmp_path / 'safe').mkdir()
    (tmp_path / 'alias.py').symlink_to('hidden.py')
    manager = LspManager.sandboxed(cwd=tmp_path, servers=(), bwrap_bin=bwrap,
                                  effective_env={'PATH': '/usr/bin:/bin'})
    paths = [str(tmp_path) + '/safe/../hidden.py', str(tmp_path / 'alias.py')]
    for path in paths:
        assert manager._visible_locations({'uri': 'file://' + path}, load_ignore_policy(tmp_path)) == (None, True)


def test_lsp_checks_reuse_the_existing_task_shell_namespace(bwrap, tmp_path):
    from scripts.llm_solver.harness.sandbox._persistent import PersistentBashSession, set_persistent_runner
    environment = {'PATH': '/usr/bin:/bin', 'HOME': '/home/task'}
    runner = PersistentBashSession(cwd=str(tmp_path), bwrap_bin=bwrap,
                                  sandbox_required=True, effective_env=environment)
    set_persistent_runner(runner)
    try:
        assert runner.run('printf dependency > /tmp/lsp-private-dependency.py',
                          cwd=str(tmp_path), timeout=10) == ('', 0, False)
        manager = LspManager.sandboxed(cwd=tmp_path, servers=(), bwrap_bin=bwrap,
                                       effective_env=environment)
        location = {'uri': 'file:///tmp/lsp-private-dependency.py'}
        assert manager._visible_locations(location, None) == (location, False)
    finally:
        set_persistent_runner(None)
        runner.close()
