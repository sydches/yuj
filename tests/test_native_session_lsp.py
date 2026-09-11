"""Session navigation applies visibility rules to the manager's native path."""
from unittest.mock import MagicMock

import pytest

from tests._config_helpers import make_config
from tests.test_task_files import bwrap, namespace_files
from scripts.llm_solver.harness.loop import Session
from scripts.llm_solver.harness.lsp_support import LspManager
from scripts.llm_solver.harness.sandbox.ignore_policy import load_ignore_policy


@pytest.mark.parametrize('native_directory', [False, True])
@pytest.mark.parametrize('spelling', ['relative', 'host', 'native'])
def test_session_lsp_checks_native_file_type_and_path(
        bwrap, tmp_path, native_directory, spelling):
    source, files = namespace_files(bwrap, tmp_path, readonly=True)
    host = tmp_path / 'host'
    host.mkdir()
    (host / '.yujignore').write_text('app.py/\n')
    if native_directory:
        (source / 'app.py').mkdir()
        (host / 'app.py').write_text('hidden host file')
    else:
        (source / 'app.py').write_text('native document')
        (host / 'app.py').mkdir()
    policy = load_ignore_policy(host)
    launcher = MagicMock(side_effect=AssertionError('no server should launch'))
    manager = LspManager(cwd=host, servers=(), tool_enabled=True, task_files=files,
                         argv_builder=launcher)
    client = MagicMock()
    cfg = make_config(lsp_tool_enabled=True)
    session = Session(cfg, client, 'system', 'fixture task', str(host),
                      effective_env={'PATH': '/usr/bin:/bin'},
                      ignore_policy=policy, lsp_manager=manager)
    requested = {'relative': 'app.py', 'host': str(host / 'app.py'),
                 'native': str(files.root / 'app.py')}[spelling]
    try:
        result = session._tool_registry.handlers['lsp'](
            {'kind': 'symbols', 'path': requested}, str(host), cfg)
        if native_directory:
            assert result == f'ERROR: lsp query failed: {files.root / "app.py"}'
        else:
            assert result == 'LSP symbols app.py status=unmatched\n[]'
        client.chat.assert_not_called()
        launcher.assert_not_called()
    finally:
        manager.close()
