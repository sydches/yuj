"""Path rules preserve native symlink names across captured task aliases."""
import pytest

from tests.test_task_files import bwrap, namespace_files
from tests.test_injections import _path_rule
from scripts.llm_solver.harness.injections import InjectionState, fire_path_candidates
from scripts.llm_solver.harness.task_path import activate_task_files


@pytest.mark.parametrize('spelling', ['relative', 'host_alias', 'host_root', 'native'])
@pytest.mark.parametrize('retarget_alias', [False, True])
def test_native_path_rule_matches_lexical_symlink_through_each_task_alias(
        bwrap, tmp_path, spelling, retarget_alias):
    source, files = namespace_files(bwrap, tmp_path)
    (source / 'actual.py').write_text('native document')
    (source / 'alias.py').symlink_to('actual.py')
    host = tmp_path / 'host'
    host.mkdir()
    outside = tmp_path / 'outside'
    outside.write_text('hidden host document')
    (host / 'alias.py').symlink_to(outside)
    alias = tmp_path / 'alias'
    alias.symlink_to(host, target_is_directory=True)
    requested = {
        'relative': 'alias.py', 'host_alias': str(alias / 'alias.py'),
        'host_root': str(host / 'alias.py'), 'native': str(files.root / 'alias.py'),
    }[spelling]
    with activate_task_files(files, host_root=alias):
        if retarget_alias:
            alias.unlink()
            alias.symlink_to(source, target_is_directory=True)
        fired = fire_path_candidates(
            [_path_rule(pattern='alias.py')], tool_name='read',
            arguments={'path': requested}, cwd=str(alias), state=InjectionState(),
        )
    assert len(fired) == 1
    assert fired[0].path == 'actual.py'
