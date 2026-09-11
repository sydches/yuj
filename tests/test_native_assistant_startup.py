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
