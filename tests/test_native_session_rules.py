"""Direct session startup must obtain rule files from its task namespace."""
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tests._config_helpers import make_config
from tests.test_task_files import bwrap, namespace_files
from scripts.llm_solver.harness import task_file_runtime
from scripts.llm_solver.harness.loop import Session


@pytest.mark.parametrize('kind', ['injections', 'stream_rules'])
@pytest.mark.parametrize('native_present', [False, True])
def test_direct_session_loads_only_native_rule_documents(
        bwrap, tmp_path, monkeypatch, kind, native_present):
    source, files = namespace_files(bwrap, tmp_path, readonly=True)
    workspace = Path(str(files.root))
    for root, name in ((source, 'native'), (workspace, 'hidden-host')):
        (root / 'rules').mkdir()
        if root == source and not native_present:
            continue
        condition = 'condition = "fixture"\n' if kind == 'stream_rules' else ''
        (root / 'rules' / 'rule.md').write_text(
            f'+++\nname = "{name}"\n{condition}+++\n{name} guidance\n')
    environment = {'PATH': '/usr/bin:/bin'}
    cfg = make_config(sandbox_bash=True, bwrap_bin=bwrap, **{
        kind + '_enabled': True, kind + '_dir': 'rules',
    })
    bindings = []

    def bind(cwd, config, **options):
        assert str(cwd) == str(workspace)
        assert dict(options['environment']) == environment
        bindings.append(True)
        return files

    monkeypatch.setattr(task_file_runtime, 'make_task_files', bind)
    client = MagicMock()
    session = Session(cfg, client, 'system', 'fixture task', str(workspace),
                      effective_env=environment, allow_login_shell=False)
    rules = session._injections if kind == 'injections' else session._stream_rule_runtime.rules
    assert [rule.name for rule in rules] == (['native'] if native_present else [])
    assert bindings
    client.chat.assert_not_called()
