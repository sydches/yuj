"""Advisor visibility is derived from the primary task's native reader."""
from types import SimpleNamespace

import pytest

from tests._config_helpers import make_config
from tests.test_task_files import bwrap, namespace_files
from scripts.llm_solver.harness._advisor_support import advisor_ignore_policy
from scripts.llm_solver.harness.sandbox.ignore_policy import load_ignore_policy
from scripts.llm_solver.harness.task_path import activate_task_files
from scripts.llm_solver.harness.tools import dispatch


@pytest.mark.parametrize('retarget_alias', [False, True])
def test_advisor_policy_hides_native_evidence_and_preserves_negation(
        bwrap, tmp_path, retarget_alias):
    source, files = namespace_files(bwrap, tmp_path, readonly=True)
    (source / '.yujignore').write_text('hidden/\n!hidden/keep.txt\n')
    (source / 'hidden').mkdir()
    (source / 'audit-notes').mkdir()
    documents = {'prompt.txt': 'reserved prompt', 'hidden/drop.txt': 'hidden task data',
                 'hidden/keep.txt': 'permitted exception',
                 'audit-notes/internal.txt': 'reserved artifact',
                 'private-trace': 'reserved trace'}
    for name, body in documents.items():
        (source / name).write_text(body)
    host = tmp_path / 'host'
    host.mkdir()
    alias = tmp_path / 'alias'
    alias.symlink_to(host, target_is_directory=True)
    other = tmp_path / 'other'
    other.mkdir()
    cfg = make_config(sandbox_bash=True, bwrap_bin=bwrap)
    with activate_task_files(files, host_root=alias):
        primary_policy = load_ignore_policy(alias)
        session = SimpleNamespace(cwd=str(alias), _ignore_policy=primary_policy,
                                  client=SimpleNamespace(), _trace_path=alias / 'private-trace')
        if retarget_alias:
            alias.unlink()
            alias.symlink_to(other, target_is_directory=True)
        policy = advisor_ignore_policy(session, alias / 'audit-notes')
        for name, body in documents.items():
            result = dispatch('read', {'path': name}, cwd=str(alias), cfg=cfg,
                              effective_env={'PATH': '/usr/bin:/bin'}, ignore_policy=policy)
            if name == 'hidden/keep.txt':
                assert body in result, result
            else:
                assert body not in result, (name, result)
        assert policy.root == host
        masks = set(policy.existing_ignored_paths())
        assert str(files.root / 'hidden/drop.txt') in masks
        assert str(files.root / 'hidden') not in masks
        assert str(files.root / 'prompt.txt') in masks
        assert str(files.root / 'audit-notes') in masks
