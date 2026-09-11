"""Branch snapshots consume the task view without running a branch attempt."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests._config_helpers import make_config
from tests.test_task_files import bwrap, namespace_files
from tests._branch_fixture import _cfg
from scripts.llm_solver.harness.adaptive_control import branch_bundle
from scripts.llm_solver.harness.context import FullTranscript
from scripts.llm_solver.harness import task_file_runtime


@pytest.mark.parametrize('link_case', ['permitted', 'escape', 'cycle'])
def test_bundle_capture_reads_native_tree_and_sunk_outputs(bwrap, tmp_path, monkeypatch, link_case):
    monkeypatch.delenv('YUJ_CONTAINER', raising=False)
    overlay = tmp_path / 'overlay'
    overlay.mkdir()
    (overlay / 'module.py').write_bytes(b'native\x00\xff\r\n')
    source, files = namespace_files(bwrap, tmp_path, overlay=overlay, readonly=True)
    workspace = Path(str(files.root))
    (workspace / 'nested').mkdir()
    (workspace / 'nested/module.py').write_text('HIDDEN_HOST_MODULE')
    (source / 'alias.py').symlink_to('nested/module.py')
    (source / '__pycache__').mkdir()
    (source / '__pycache__/old.pyc').write_bytes(b'omitted')
    retained = {'.tox/bin/python': b'runtime state',
                '.pytest_cache/v/cache/lastfailed': b'{"failed": true}',
                'fixtures/.pytest_cache/expected.txt': b'ordinary fixture',
                'fixtures/example.pyc': b'ordinary bytes'}
    for relative, data in retained.items():
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    for root, body in ((source, 'native output'), (workspace, 'HIDDEN_HOST_OUTPUT')):
        (root / '.tool_output').mkdir()
        (root / '.tool_output/log').write_text(body)
    if link_case == 'escape':
        outside = tmp_path / 'outside'
        outside.write_text('UNADMITTED_OUTSIDE_SOURCE')
        (source / 'escape.py').symlink_to(outside)
    elif link_case == 'cycle':
        (source / 'recursive').symlink_to('.')
    baseline = tmp_path / 'baseline.toml'
    baseline.write_text('[loop]\nmax_turns = 3\n')
    artifacts = tmp_path / 'records'
    artifacts.mkdir()
    state = artifacts / 'state.json'
    state.write_text('{"private_state":true}\n')
    cfg = make_config(**vars(_cfg(tmp_path / 'bundles', baseline)),
                      sandbox_bash=True, bwrap_bin=bwrap)
    context = FullTranscript()
    context.add_user('Synthetic task')
    environment = {'PATH': '/usr/bin:/bin', 'HOME': str(workspace)}
    session = SimpleNamespace(
        cfg=cfg, cwd=str(workspace), context=context, instance_id='fixture',
        attempt_id='fixture-attempt', _state_path=state,
        _effective_env=environment, _allow_login_shell=False, _ignore_policy=None,
    )
    calls = []

    def bind(cwd, config, **options):
        assert str(cwd) == str(workspace)
        assert options['environment'] is environment
        calls.append(True)
        return files

    monkeypatch.setattr(task_file_runtime, 'make_task_files', bind)
    decision = SimpleNamespace(diagnosis_status='active_confirmed',
                               active_hurdle_mode='fixture', detector_status='active_confirmed',
                               basis_refs=['turn=1'])
    result = branch_bundle.maybe_capture(session, decision, 1, 'fixture-boundary')
    assert calls
    bundle = tmp_path / 'bundles' / result['branch_point_id']
    if link_case != 'permitted':
        assert result['status'] == 'blocked'
        assert result['reason'].startswith('bundle_write_failed:')
        assert not bundle.exists()
        assert not bundle.with_name('.' + bundle.name + '.tmp').exists()
    else:
        assert result['status'] == 'created', result
        assert (bundle / 'repo_snapshot/nested/module.py').read_bytes() == b'native\x00\xff\r\n'
        assert (bundle / 'repo_snapshot/alias.py').read_bytes() == b'native\x00\xff\r\n'
        assert not (bundle / 'repo_snapshot/alias.py').is_symlink()
        assert (bundle / 'sunk_outputs/log').read_text() == 'native output'
        assert json.loads((bundle / 'solver_state.json').read_text()) == {'private_state': True}
        exclusions = json.loads((bundle / 'snapshot_exclusions.json').read_text())
        assert exclusions['exclusions'] == []
        assert exclusions['policy'] == 'preserve_permitted_entries_v1'
        for relative, data in retained.items():
            assert (bundle / 'repo_snapshot' / relative).read_bytes() == data
        assert (bundle / 'repo_snapshot/__pycache__/old.pyc').read_bytes() == b'omitted'
        assert exclusions['retention_status'] == 'unverified'
    assert (workspace / 'nested/module.py').read_text() == 'HIDDEN_HOST_MODULE'


def test_bundle_capture_keeps_native_task_after_host_alias_retargets(bwrap, tmp_path):
    from scripts.llm_solver.harness.task_path import activate_task_files

    source, files = namespace_files(bwrap, tmp_path)
    (source / 'module.py').write_text('selected native module')
    host = tmp_path / 'host'
    host.mkdir()
    (host / 'module.py').write_text('hidden host module')
    other = tmp_path / 'other'
    other.mkdir()
    (other / 'module.py').write_text('replacement host module')
    alias = tmp_path / 'alias'
    alias.symlink_to(host, target_is_directory=True)
    baseline = tmp_path / 'baseline.toml'
    baseline.write_text('[loop]\nmax_turns = 3\n')
    cfg = make_config(**vars(_cfg(tmp_path / 'bundles', baseline)),
                      sandbox_bash=True, bwrap_bin=bwrap)
    session = SimpleNamespace(
        cfg=cfg, cwd=str(alias), context=FullTranscript(),
        instance_id='fixture', attempt_id='fixture-attempt',
        _effective_env={'PATH': '/usr/bin:/bin'}, _allow_login_shell=False,
    )
    decision = SimpleNamespace(diagnosis_status='active_confirmed',
                               active_hurdle_mode='fixture', detector_status='active_confirmed',
                               basis_refs=['turn=1'])
    with activate_task_files(files, host_root=alias):
        alias.unlink()
        alias.symlink_to(other, target_is_directory=True)
        result = branch_bundle.maybe_capture(session, decision, 1, 'fixture-boundary')
    assert result['status'] == 'created', result
    bundle = tmp_path / 'bundles' / result['branch_point_id']
    assert (bundle / 'repo_snapshot/module.py').read_text() == 'selected native module'
