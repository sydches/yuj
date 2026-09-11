"""Relocated installed tools remain usable without widening filesystem access."""
from pathlib import Path
import shlex

import pytest

from test_native_toolchain_selection import managed_task, freeze, command
from scripts.llm_solver.harness.task_environment import task_environment_scope


@pytest.mark.parametrize('under_home', [True, False])
def test_relocated_manager_compiles_with_private_state(managed_task, under_home):
    task, home, manager_home, settings, env = managed_task
    relocated = (home if under_home else task.parent) / 'relocated manager'
    manager_home.rename(relocated)
    settings = relocated / settings.name
    before = settings.read_bytes()
    (relocated / 'private-data').write_text('private manager data')
    env.update(RUSTUP_HOME=str(relocated),
               RUSTUP_TOOLCHAIN=str(relocated / 'toolchains/fixture-second'))
    (task / 'main.rs').write_text('fn main() { println!("42"); }\n')

    @task_environment_scope
    def session():
        view = freeze(task, env)
        assert view.native_selections[0]['status'] == 'selected'
        result = command(task, env, 'rustc main.rs -o check && ./check && cargo --version')
        assert result.returncode == 0, result.stderr
        assert result.stdout.startswith('42\ncargo ')
        excluded = [home / 'private.txt', task.parent / 'sibling.txt',
                    relocated / 'private-data', relocated / 'toolchains/fixture-first']
        result = command(task, env, ' && '.join('test ! -e ' + shlex.quote(str(p)) for p in excluded))
        assert result.returncode == 0, result.stderr
    session()
    assert settings.read_bytes() == before


@pytest.mark.parametrize('admitted', [False, True])
def test_project_alias_read_respects_admitted_resources(managed_task, monkeypatch, admitted):
    task, home, manager_home, settings, env = managed_task
    resource = home / 'selection-resource'
    resource.mkdir()
    external = resource / 'private-selection.toml'
    external.write_text('[toolchain]\nchannel="fixture-second"\n')
    alias = task / 'rust-toolchain.toml'
    alias.symlink_to(external)
    opened = []
    original_open = Path.open

    def watch(path, *args, **kwargs):
        if path.resolve() == external:
            opened.append(path)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', watch)
    view = freeze(task, env, readable_paths=(str(resource),) if admitted else ())
    record, = view.native_selections
    assert record['status'] == 'selected', record
    assert bool(opened) is admitted
    assert any(item['path'] == str(alias) for item in record['inputs']) is admitted
    assert record['toolchain'].endswith('/fixture-second' if admitted else '/fixture-first')
    if not admitted:
        assert record['unavailable_inputs'] == [{'path': str(alias), 'reason': 'outside_permitted_view'}]
