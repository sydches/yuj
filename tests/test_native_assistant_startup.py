"""Assistant preflight reads declarations and guidance in the task view."""
from pathlib import Path
import pytest

from tests.test_task_files import bwrap, namespace_files
from scripts.llm_assist import startup
from scripts.llm_solver.harness import task_file_runtime


@pytest.mark.parametrize('declaration_denied', [False, True])
def test_preflight_shares_native_rules_skills_and_runner_reads(bwrap, tmp_path, monkeypatch, declaration_denied):
    source, files = namespace_files(bwrap, tmp_path, readonly=True)
    target = Path(str(files.root))
    (source / '.git').mkdir()
    (source / 'pytest.ini').write_text('[pytest]\n')
    if declaration_denied:
        (source / 'pytest.ini').chmod(0)
    (source / 'AGENTS.md').write_text('NATIVE_GUIDANCE\n')
    (source / '.yujignore').write_text('hidden.md\n')
    (target / 'go.mod').write_text('module example.org/hidden\n')
    (target / 'AGENTS.md').write_text('HIDDEN_HOST_GUIDANCE\n')
    (target / '.yujignore').write_text('skills/native\npytest.ini\n')
    outside_prompt = tmp_path / 'host-prompt.md'
    outside_prompt.write_text('HOST_LINK_TARGET\n')
    (target / 'arm.md').symlink_to(outside_prompt)
    (source / 'arm.md').write_text('NATIVE_SYSTEM_PROMPT\n')
    for root, name in ((source, 'native'), (target, 'host')):
        skill = root / 'skills' / name
        skill.mkdir(parents=True)
        (skill / 'SKILL.md').write_text(
            f'---\nname: {name}\ndescription: {name} task guidance\n---\nUse {name}.\n')
    environment = {'PATH': '/usr/bin:/bin', 'HOME': str(target)}
    discovered = []
    bound_environments = []

    def discover(cfg, *, cwd):
        assert cwd == target
        discovered.append(True)
        return environment, False

    def bind(cwd, cfg, **options):
        assert str(cwd) == str(target)
        bound_environments.append(options['environment'])
        return files

    monkeypatch.setattr(startup, '_effective_command_environment', discover)
    monkeypatch.setattr(task_file_runtime, 'make_task_files', bind)
    original_prompt = startup.load_system_prompt_and_provenance
    prompts = []

    def assemble(*args, **kwargs):
        result = original_prompt(*args, **kwargs)
        prompts.append(result[0])
        return result

    monkeypatch.setattr(startup, 'load_system_prompt_and_provenance', assemble)
    report = startup.preflight_assistant_startup(
        config_paths=(), cwd=target, context_mode='full', system_prompt_file=target / 'arm.md', config_overrides={
            'sandbox_bash': True, 'bwrap_bin': bwrap,
            'project_docs_enabled': True, 'project_doc_global_dir': '',
            'skills_enabled': True, 'skills_dirs': ('skills',), 'skill_paths': (),
        },
    )
    assert discovered == [True]
    assert bound_environments and all(item is bound_environments[0] for item in bound_environments)
    assert dict(bound_environments[0]) == environment
    assert report.detected_runner == ('generic' if declaration_denied else 'pytest')
    assert report.skill_count == 1 and report.project_instruction_count == 1
    assert not report.network_contacted
    assert 'NATIVE_GUIDANCE' in prompts[0] and 'native task guidance' in prompts[0]
    assert 'HIDDEN_HOST_GUIDANCE' not in prompts[0] and 'host task guidance' not in prompts[0]
    assert 'NATIVE_SYSTEM_PROMPT' in prompts[0] and 'HOST_LINK_TARGET' not in prompts[0]


@pytest.mark.parametrize('pattern,blocks_nested', [('**/secret.py', True), ('*/secret.py', False)])
def test_native_masks_expand_ancestor_globs_only_within_the_task(bwrap, tmp_path, pattern, blocks_nested):
    from scripts.llm_solver.harness.task_path import NativeUnreadableMatcher, TaskPath
    source, files = namespace_files(bwrap, tmp_path)
    (source / 'deep').mkdir()
    (source / 'secret.py').write_text('root secret')
    (source / 'deep/secret.py').write_text('nested secret')
    (source / 'visible.py').write_text('visible')
    root = TaskPath(files, files.root)
    matcher = NativeUnreadableMatcher(root, (
        str(tmp_path / 'outside' / '**'), str(tmp_path / pattern),
    ))
    assert matcher.blocks(root / 'secret.py')
    assert matcher.blocks(root / 'deep/secret.py') is blocks_nested
    assert not matcher.blocks(root / 'visible.py')


def test_runtime_discovery_reads_native_layout_and_declarations(bwrap, tmp_path, monkeypatch):
    import hashlib
    from _config_helpers import make_config
    from scripts.llm_solver.harness import runtime_discovery
    from scripts.llm_solver.harness.task_path import activate_task_files

    source, files = namespace_files(bwrap, tmp_path)
    target = Path(str(files.root))
    (source / 'package.json').write_text('{"scripts":{"test":"native-check"}}')
    (source / 'native.js').write_text('// task language')
    (source / 'hidden.py').write_text('# excluded task source')
    (target / 'pyproject.toml').write_text('[project]\nrequires-python="HOST_ONLY"\n')
    (target / 'host.py').write_text('# hidden host language')
    monkeypatch.setattr(runtime_discovery, '_run_in_sandbox', lambda *a, **k: ('', 0, False))
    cfg = make_config(sandbox_bash=True, sandbox_backend='container',
                      unreadable_paths=(str(target / 'hidden.py'),))
    with activate_task_files(files, host_root=target):
        report = runtime_discovery.discover_runtime(
            target, cfg, effective_env={}, unreadable_paths=cfg.unreadable_paths,
            selection_only=True)
        inputs = runtime_discovery.selection_inputs(target, ('package.json',))
    assert inputs['package.json']['sha256'] == hashlib.sha256((source / 'package.json').read_bytes()).hexdigest()
    layout = next(row for row in report['observations'] if row['source'] == 'task_directory_scan')
    assert set(layout['source_languages']) == {'JavaScript'}
    declared = next(row for row in report['observations'] if row.get('path') == 'package.json')
    assert declared['values']['scripts.test'] == 'native-check'
    assert 'HOST_ONLY' not in str(report) and 'host.py' not in str(report)


def test_native_global_guidance_uses_task_home_and_keeps_ancestor_chain(bwrap, tmp_path, monkeypatch):
    import shlex
    from scripts.llm_solver.harness.project_instructions import discover_project_instructions
    from scripts.llm_solver.harness.task_path import TaskPath

    source, files = namespace_files(bwrap, tmp_path)
    target = Path(str(files.root))
    (source / '.git').mkdir()
    (source / 'child').mkdir()
    (source / 'AGENTS.md').write_text('NATIVE_PARENT')
    (source / 'child/AGENTS.md').write_text('NATIVE_CHILD')
    (source / 'guidance').mkdir()
    (source / 'guidance/AGENTS.md').write_text('NATIVE_GLOBAL')
    (target / 'guidance').mkdir()
    (target / 'guidance/AGENTS.md').write_text('HOST_GLOBAL')
    host_home = tmp_path / 'host-home'
    (host_home / 'guidance').mkdir(parents=True)
    (host_home / 'guidance/AGENTS.md').write_text('HOST_HOME_GLOBAL')
    monkeypatch.setenv('HOME', str(host_home))
    native_run = files.run
    files.run = lambda script, args, data: native_run(
        'export HOME=' + shlex.quote(str(files.root)) + '\n' + script, args, data)
    task = TaskPath(files.readonly_view('/'), files.root / 'child')
    result = discover_project_instructions(task, global_dir='~/guidance')
    assert [document.content for document in result.documents] == [
        'NATIVE_GLOBAL', 'NATIVE_PARENT', 'NATIVE_CHILD']
    assert 'HOST_GLOBAL' not in result.content
    assert 'HOST_HOME_GLOBAL' not in result.content


def test_native_catalog_alias_reads_existing_view_without_host_mount_grant(bwrap, tmp_path):
    from _config_helpers import make_config
    from scripts.llm_solver.harness._tools.read import read
    from scripts.llm_solver.harness.task_path import activate_task_files
    from scripts.llm_solver.harness.tools import _bash_readable_paths

    source, files = namespace_files(bwrap, tmp_path)
    (source / 'skill').mkdir()
    (source / 'skill/SKILL.md').write_text('NATIVE_SKILL_BODY')
    (source / 'task').mkdir()
    (Path(str(files.root)) / 'skill').mkdir()
    (Path(str(files.root)) / 'skill/SKILL.md').write_text('HOST_SKILL_BODY')
    task_files = type(files)(str(files.root / 'task'), files.run, binding=files.binding)
    cfg = make_config(sandbox_bash=True,
                      skills_native_readable_dirs=(str(files.root / 'skill'),))
    assert _bash_readable_paths(cfg) == ()
    with activate_task_files(task_files, host_root=str(files.root / 'task')):
        result = read(str(files.root / 'skill/SKILL.md'), cwd=str(files.root / 'task'), cfg=cfg)
    assert 'NATIVE_SKILL_BODY' in result and 'HOST_SKILL_BODY' not in result


def test_native_layout_batches_file_metadata(bwrap, tmp_path):
    from scripts.llm_solver.harness.runtime_discovery import _inspect_layout
    from scripts.llm_solver.harness.prompt_imports import _UnreadableMatcher
    from scripts.llm_solver.harness.task_path import TaskPath

    source, files = namespace_files(bwrap, tmp_path)
    for index in range(200):
        (source / f'file-{index}.js').write_text('// source')
    root = TaskPath(files, files.root).resolve()
    blocked = _UnreadableMatcher(root, ())
    calls = []
    original = files.run
    def recorded(script, args, data):
        calls.append(args)
        return original(script, args, data)
    files.run = recorded
    report = _inspect_layout(root, {'source_suffixes': {'.js': 'JavaScript'}}, blocked, float('inf'))
    assert report['source_languages']['JavaScript']['files'] == 200
    operations = [args[3] for args in calls if len(args) >= 5]
    assert operations == ['scandir']
