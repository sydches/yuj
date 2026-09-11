"""Post-edit validators receive native paths and the selected environment."""
import pytest

from tests._config_helpers import make_config
from tests.test_task_files import bwrap, namespace_files
from scripts.llm_solver.harness.post_edit import run_post_edit_checks
from scripts.llm_solver.harness.sandbox.env_policy import activate_environment
from scripts.llm_solver.harness.task_file_runtime import task_file_scope
from scripts.llm_solver.harness.task_path import activate_task_files
from scripts.llm_solver.harness.tools import dispatch


@pytest.mark.parametrize('spelling', ['relative', 'host', 'native'])
@pytest.mark.parametrize('expected', ['EDITED', 'DIFFERENT'])
def test_edit_validator_uses_native_relative_alias_and_environment(
    bwrap, tmp_path, monkeypatch, spelling, expected,
):
    from scripts.llm_solver.harness import tools
    root, files = namespace_files(bwrap, tmp_path)
    (root / 'nested').mkdir()
    name = "nested/source with 'quote'.txt"
    (root / name).write_text('NATIVE')
    host = tmp_path / 'host_alias'
    (host / 'nested').mkdir(parents=True)
    (host / name).write_text('HIDDEN_HOST')
    environment = {'PATH': '/usr/bin:/bin', 'EXPECTED': expected}
    cfg = make_config(sandbox_bash=True, bwrap_bin=bwrap,
        post_edit_check_enabled=True, post_edit_check_timeout=0,
        post_edit_checks=[{'name': 'native-content', 'trigger': 'edit',
                          'when': "path.startswith('nested/') and ext == '.txt'",
                          'cmd': 'test "$(cat {path})" = "$EXPECTED"', 'on_fail': 'append'}])
    submitted = []

    def execute(command, **options):
        assert dict(options['effective_env']) == environment
        assert options['sandbox'] and options['sandbox_backend'] == 'bwrap'
        submitted.append(command)
        # The fixture owns a real bwrap view. Execute the declared validator
        # there with the same explicit environment, without another backend.
        import shlex
        script = shlex.join(['env', '-i', *(f'{k}={v}' for k, v in environment.items()),
                             'bash', '--noprofile', '--norc', '-c', command])
        result = files.run(script, [], None)
        return (result.stdout + result.stderr).decode(), result.returncode, False

    monkeypatch.setattr(tools, '_run_in_sandbox', execute)
    path = {'relative': name, 'host': str(host / name), 'native': str(files.root / name)}[spelling]
    with activate_task_files(files, host_root=host), activate_environment(environment, allow_login_shell=False):
        with task_file_scope(host, cfg, environment=environment, allow_login_shell=False):
            result = dispatch('edit', {'path': path, 'old_str': 'NATIVE', 'new_str': 'EDITED'},
                              cwd=str(host), cfg=cfg, effective_env=environment,
                              allow_login_shell=False)
    assert (root / name).read_text() == 'EDITED'
    assert (host / name).read_text() == 'HIDDEN_HOST'
    assert len(submitted) == 1
    assert str(host) not in submitted[0] and str(files.root) not in submitted[0]
    assert ('post-edit check' in result) is (expected == 'DIFFERENT')


def test_post_edit_validator_reads_read_only_nested_mount(bwrap, tmp_path, monkeypatch):
    from scripts.llm_solver.harness import tools
    overlay = tmp_path / 'overlay'
    overlay.mkdir()
    (overlay / 'source.txt').write_text('NATIVE')
    root, files = namespace_files(bwrap, tmp_path, overlay=overlay)
    (root / 'nested/source.txt').write_text('HOST')
    cfg = make_config(sandbox_bash=True, bwrap_bin=bwrap, post_edit_check_enabled=True,
        post_edit_checks=[{'name': 'read-only-check', 'trigger': 'edit', 'when': '',
                          'cmd': 'test "$(cat {path})" = NATIVE', 'on_fail': 'block'}])

    def execute(command, **options):
        result = files.run(command, [], None)
        return result.stdout.decode(), result.returncode, False

    monkeypatch.setattr(tools, '_run_in_sandbox', execute)
    with activate_task_files(files, host_root=root):
        result = run_post_edit_checks(str(root / 'nested/source.txt'), cwd=str(root), cfg=cfg, trigger='edit')
    assert result.action == 'ok'
    assert (root / 'nested/source.txt').read_text() == 'HOST'
