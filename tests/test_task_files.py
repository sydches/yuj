"""Task file primitives must observe real namespace mounts and permissions."""
import shutil
import shlex
import subprocess

import pytest

from scripts.llm_solver.harness.task_files import NamespaceFiles, TaskFileError


@pytest.mark.parametrize('retarget', [False, True])
def test_legacy_chmod_uses_a_checked_descriptor(tmp_path, retarget):
    root = tmp_path / 'task'
    root.mkdir()
    outside = tmp_path / 'outside'
    outside.write_bytes(b'private')
    outside.chmod(0o600)
    utility = root / 'legacy-chmod'
    swap = (f'rm -f -- "${{@: -1}}"; ln -s {shlex.quote(str(outside))} "${{@: -1}}"\n'
            if retarget else '')
    utility.write_text('#!/bin/bash\n'
        'if [[ "$1" == --help ]]; then echo legacy-chmod; exit 0; fi\n'
        'if [[ "$1" == --no-dereference ]]; then\n' + swap +
        '  echo unsupported-option >&2; exit 2\nfi\n' +
        f'exec {shlex.quote(shutil.which("chmod"))} "$@"\n')
    utility.chmod(0o755)
    def run(script, args, data):
        return subprocess.run(['bash', '-c', script, 'fixture', *args],
                              input=data, capture_output=True)
    files = NamespaceFiles(str(root), run, binding={'fixture': str(root)})
    files._utilities['chmod'] = str(utility)
    if retarget:
        with pytest.raises(TaskFileError):
            files.replace_bytes('restored', b'new', mode=0o640)
    else:
        files.replace_bytes('restored', b'new', mode=0o640)
        assert (root / 'restored').read_bytes() == b'new'
        assert (root / 'restored').stat().st_mode & 0o777 == 0o640
    assert outside.read_bytes() == b'private'
    assert outside.stat().st_mode & 0o777 == 0o600


@pytest.fixture(scope='module')
def bwrap():
    binary = shutil.which('bwrap')
    if binary is None:
        pytest.skip('bwrap is unavailable')
    result = subprocess.run([binary, '--ro-bind', '/', '/', '--unshare-net',
                             '--', 'true'], capture_output=True)
    if result.returncode:
        pytest.skip('mount namespace unavailable: ' + result.stderr.decode(errors='replace'))
    return binary


def namespace_files(bwrap, tmp_path, *, overlay=None, readonly=False, writable=()):
    source = tmp_path / 'source'
    view = tmp_path / 'view'
    source.mkdir()
    view.mkdir()
    mounts = ['--ro-bind' if readonly else '--bind', str(source), str(view)]
    if overlay:
        (source / 'nested').mkdir()
        mounts += ['--ro-bind', str(overlay), str(view / 'nested')]
    for path in writable:
        mounts += ['--bind', str(path), str(path)]

    def run(script, args, data):
        return subprocess.run(
            [bwrap, '--ro-bind', '/', '/', '--dev', '/dev', '--unshare-net', *mounts,
             '--chdir', str(view), '--', 'bash', '--noprofile', '--norc',
             '-c', script, 'task-file-test', *args],
            input=data, capture_output=True,
        )
    return source, NamespaceFiles(str(view), run, binding={'test_namespace': str(view)},
                                  shares_host_kernel=True)


def test_binary_roundtrip_metadata_and_literal_paths(bwrap, tmp_path):
    source, files = namespace_files(bwrap, tmp_path)
    name = "literal ' $(touch escaped)\nfile\n"
    data = b'zero\x00byte\xff\r\nlast\n'
    assert files.write_bytes(name, data) == len(data)
    assert files.read_bytes(name) == data
    assert (source / name).read_bytes() == data
    assert not (source / 'escaped').exists()
    info = files.metadata(name)
    assert info.is_file and not info.is_dir and info.size == len(data)
    assert files.resolve(name) == files.root / name
    assert files.iterdir() == [files.root / name]


def test_nested_mount_hides_host_source_for_read_stat_and_listing(bwrap, tmp_path):
    overlay = tmp_path / 'overlay'
    overlay.mkdir()
    (overlay / 'visible').write_bytes(b'container bytes')
    source, files = namespace_files(bwrap, tmp_path, overlay=overlay)
    (source / 'nested' / 'hidden').write_bytes(b'host answer')
    assert files.read_bytes('nested/visible') == b'container bytes'
    assert files.metadata('nested/visible').size == len(b'container bytes')
    assert files.iterdir('nested') == [files.root / 'nested/visible']
    with pytest.raises(FileNotFoundError):
        files.read_bytes('nested/hidden')
    with pytest.raises(TaskFileError):
        files.write_bytes('nested/new', b'changed')
    assert not (source / 'nested' / 'new').exists()
    assert not (overlay / 'new').exists()


def test_readonly_mount_preserves_reads_and_rejects_mutations(bwrap, tmp_path):
    source, files = namespace_files(bwrap, tmp_path, readonly=True)
    (source / 'file').write_bytes(b'original')
    assert files.read_bytes('file') == b'original'
    for operation in (lambda: files.write_bytes('file', b'changed'),
                      lambda: files.write_bytes('new', b'created'),
                      lambda: files.mkdir('directory'),
                      lambda: files.unlink('file')):
        with pytest.raises(TaskFileError):
            operation()
    assert (source / 'file').read_bytes() == b'original'
    assert sorted(p.name for p in source.iterdir()) == ['file']


def test_symlink_resolution_uses_namespace_and_refuses_escape(bwrap, tmp_path):
    source, files = namespace_files(bwrap, tmp_path)
    (source / 'file').write_bytes(b'task')
    (source / 'alias').symlink_to('file')
    outside = tmp_path / 'outside'
    outside.write_bytes(b'not permitted')
    (source / 'escape').symlink_to(outside)
    assert files.read_bytes('alias') == b'task'
    for name in ('escape', '../outside'):
        with pytest.raises(PermissionError):
            files.read_bytes(name)
        with pytest.raises(PermissionError):
            files.write_bytes(name, b'changed')
    assert outside.read_bytes() == b'not permitted'


def test_creation_and_removal_in_selected_view(bwrap, tmp_path):
    source, files = namespace_files(bwrap, tmp_path)
    files.mkdir('nested/child', parents=True)
    files.write_bytes('nested/child/file', b'body')
    files.unlink('nested/child/file')
    assert (source / 'nested/child').is_dir()
    assert files.iterdir('nested/child') == []


def test_unlink_removes_the_directory_entry_not_its_symlink_target(bwrap, tmp_path):
    source, files = namespace_files(bwrap, tmp_path)
    (source / 'file').write_bytes(b'keep')
    (source / 'alias').symlink_to('file')
    files.unlink('alias')
    assert not (source / 'alias').is_symlink()
    assert (source / 'file').read_bytes() == b'keep'


def test_missing_utility_never_selects_a_host_fallback(tmp_path):
    calls = []
    def unavailable(script, args, data):
        calls.append(args)
        return subprocess.CompletedProcess([], 0, b'realpath\0\0', b'')
    files = NamespaceFiles(str(tmp_path), unavailable, binding={'namespace': 'unavailable'})
    (tmp_path / 'secret').write_bytes(b'host bytes')
    with pytest.raises(TaskFileError, match='no usable realpath'):
        files.read_bytes('secret')
    assert calls == [['realpath']]


def test_descriptive_binding_cannot_assert_shared_kernel_identity(tmp_path):
    def forbidden(*args):
        raise AssertionError('unverified identity must not query or compare device/inode pairs')
    files = NamespaceFiles(str(tmp_path), forbidden,
                           binding={'host_root': str(tmp_path), 'shares_host_kernel': True})
    assert files.observe_host_entry('metrics.json', tmp_path / 'metrics.json') == {
        'relation': 'unverified', 'basis': 'unverified_kernel_relation',
    }


def test_startup_project_guidance_and_imports_use_selected_contents(bwrap, tmp_path):
    from scripts.llm_solver.harness.task_path import activate_task_files
    from scripts.llm_solver.harness.project_instructions import (
        discover_project_instructions, resolve_project_instruction_imports,
    )
    source, files = namespace_files(bwrap, tmp_path)
    view = tmp_path / 'view'
    (view / 'AGENTS.md').write_text('hidden host guidance')
    (source / '.git').mkdir()
    (source / 'guides').mkdir()
    (source / 'AGENTS.md').write_text('Selected task guidance\n@guides/rules.md\n')
    (source / 'guides' / 'rules.md').write_text('Selected imported rule\n')
    global_dir = tmp_path / 'global'
    global_dir.mkdir()
    (global_dir / 'AGENTS.md').write_text('Operator global guidance\n')
    with activate_task_files(files, host_root=view):
        project = discover_project_instructions(view, global_dir=global_dir)
        resolved = resolve_project_instruction_imports(project, enabled=True, max_depth=3)
    assert 'Selected task guidance' in resolved.content
    assert 'Selected imported rule' in resolved.content
    assert 'Operator global guidance' in resolved.content
    assert 'hidden host guidance' not in resolved.content
    assert resolved.files == ('global/AGENTS.md', 'AGENTS.md')
    assert resolved.imported_bytes == len('Selected imported rule\n')


def test_native_instruction_parent_chain_and_masks(bwrap, tmp_path):
    from scripts.llm_solver.harness.task_path import TaskPath
    from scripts.llm_solver.harness.project_instructions import (
        discover_project_instructions, resolve_project_instruction_imports,
    )
    source, files = namespace_files(bwrap, tmp_path)
    (source / '.git').mkdir()
    (source / 'nested').mkdir()
    (source / 'AGENTS.md').write_text('Root guidance\n')
    (source / 'nested' / 'AGENTS.md').write_text('Nested guidance\n@private.md\n')
    (source / 'nested' / 'private.md').write_text('must stay hidden')
    readonly = files.readonly_view('/')
    cwd = TaskPath(readonly, files.root / 'nested')
    masks = (str(files.root / 'nested/private.md'),)
    project = discover_project_instructions(cwd, unreadable_paths=masks)
    resolved = resolve_project_instruction_imports(project, enabled=True, max_depth=3,
                                                    unreadable_paths=masks)
    assert resolved.files == ('AGENTS.md', 'nested/AGENTS.md')
    assert 'Root guidance' in resolved.content and 'Nested guidance' in resolved.content
    assert 'must stay hidden' not in resolved.content
    assert 'blocked' in resolved.content or 'unreadable' in resolved.content


def test_operator_prompt_import_into_task_tree_uses_native_bytes(bwrap, tmp_path):
    from scripts.llm_solver.harness.task_path import TaskPath
    from scripts.llm_solver.harness.solver import resolve_system_prompt_source
    source, files = namespace_files(bwrap, tmp_path)
    view = tmp_path / 'view'
    (view / 'rules.md').write_text('hidden host import')
    (source / 'rules.md').write_text('selected task import')
    prompt = tmp_path / 'prompt.md'
    prompt.write_text('Operator prompt\n@view/rules.md\n')
    readonly = files.readonly_view('/')
    root = TaskPath(readonly, files.root)
    resolved = resolve_system_prompt_source(prompt, allowed_dirs=(root, prompt.parent))
    assert 'Operator prompt' in resolved.content
    assert 'selected task import' in resolved.content
    assert 'hidden host import' not in resolved.content
    blocked = resolve_system_prompt_source(prompt, allowed_dirs=(root, prompt.parent),
                                            unreadable_paths=(str(view / 'rules.md'),))
    assert 'selected task import' not in blocked.content


def test_native_home_and_skill_metadata_are_discovered_in_the_view(bwrap, tmp_path):
    import shlex
    from scripts.llm_solver.harness.task_path import activate_task_files
    from scripts.llm_solver.harness.skills import discover_skills
    source, original = namespace_files(bwrap, tmp_path)
    (source / '.git').mkdir()
    directory = source / 'skills' / 'selected'
    directory.mkdir(parents=True)
    (directory / 'SKILL.md').write_text('---\nname: selected\ndescription: Native metadata\n---\nBody\n')
    view = tmp_path / 'view'
    hidden = view / 'skills' / 'hidden'
    hidden.mkdir(parents=True)
    (hidden / 'SKILL.md').write_text('---\nname: hidden\ndescription: Host metadata\n---\n')
    def run(script, args, data):
        return original.run('export HOME=' + shlex.quote(str(original.root))
                            + '\nexport LITERAL="~/literal"\n' + script, args, data)
    files = NamespaceFiles(str(original.root), run, binding=original.binding)
    assert files.expand_path('~/skills') == str(files.root / 'skills')
    assert files.expand_path('$HOME/skills', variables=True) == str(files.root / 'skills')
    assert files.expand_path('$LITERAL', variables=True) == '~/literal'
    literal = source / '~' / 'literal' / 'literal'
    literal.mkdir(parents=True)
    (literal / 'SKILL.md').write_text('---\nname: literal\ndescription: Expanded once\n---\n')
    with activate_task_files(files, host_root=view):
        catalog = discover_skills(view, enabled=True, skills_dirs=('~/skills',))
        literal_catalog = discover_skills(view, enabled=True, skills_dirs=('$LITERAL',))
    assert [skill.name for skill in catalog.skills] == ['selected']
    assert catalog.skills[0].description == 'Native metadata'
    assert [skill.name for skill in literal_catalog.skills] == ['literal']


def test_prompt_assembly_uses_the_selected_task_guidance(bwrap, tmp_path):
    from types import SimpleNamespace
    from tests._config_helpers import make_config
    from scripts.llm_solver.harness.task_path import activate_task_files
    from scripts.llm_solver.harness._loop._driver_setup import load_system_prompt_and_provenance
    source, files = namespace_files(bwrap, tmp_path)
    view = tmp_path / 'view'
    (source / '.git').mkdir()
    (source / 'AGENTS.md').write_text('Selected instruction\n@rules.md\n')
    (source / 'rules.md').write_text('Selected imported instruction')
    (view / 'AGENTS.md').write_text('Hidden host instruction')
    cfg = make_config(project_docs_enabled=True, project_doc_global_dir='')
    client = SimpleNamespace(profile=SimpleNamespace(preamble=''))
    with activate_task_files(files, host_root=view):
        prompt, _, _, metadata = load_system_prompt_and_provenance(
            cfg, client, view, None, None, None, None,
        )
    assert 'Selected instruction' in prompt and 'Selected imported instruction' in prompt
    assert 'Hidden host instruction' not in prompt
    assert metadata.project_instruction_files[0]['path'] == 'AGENTS.md'


def test_declared_prompt_and_global_guidance_inside_task_use_native_view(bwrap, tmp_path):
    from types import SimpleNamespace
    from tests._config_helpers import make_config
    from scripts.llm_solver.harness.task_path import activate_task_files
    from scripts.llm_solver.harness._loop._driver_setup import load_system_prompt_and_provenance
    source, files = namespace_files(bwrap, tmp_path)
    view = tmp_path / 'view'
    (source / '.git').mkdir()
    for tree, marker in ((source, 'Selected'), (view, 'Hidden host')):
        (tree / 'global').mkdir()
        (tree / 'global' / 'AGENTS.md').write_text(marker + ' global guidance')
        (tree / 'arm.md').write_text(marker + ' declared prompt')
    cfg = make_config(project_docs_enabled=True, project_doc_global_dir=str(view / 'global'))
    client = SimpleNamespace(profile=SimpleNamespace(preamble=''))
    with activate_task_files(files, host_root=view):
        prompt, _, _, _ = load_system_prompt_and_provenance(
            cfg, client, view, view / 'arm.md', None, None, None,
        )
    assert 'Selected declared prompt' in prompt
    assert 'Selected global guidance' in prompt
    assert 'Hidden host' not in prompt


def test_declared_global_skills_remain_available_through_private_home(bwrap, tmp_path):
    from tests._config_helpers import make_config
    from scripts.llm_solver.harness.startup_files import discover_task_skills
    from scripts.llm_solver.harness.task_file_runtime import make_task_files
    from dataclasses import replace
    task = tmp_path / 'task'
    task.mkdir()
    home = tmp_path / 'home'
    package = home / 'skills' / 'global'
    package.mkdir(parents=True)
    content = '---\nname: global\ndescription: Declared resource\n---\nBody stays lazy\n'
    (package / 'SKILL.md').write_text(content)
    environment = {'PATH': '/usr/bin:/bin', 'HOME': str(home)}
    cfg = make_config(sandbox_bash=True, bwrap_bin=bwrap, skills_enabled=True,
                      skills_dirs=('~/skills',), unreadable_paths=())
    catalog = discover_task_skills(str(task), cfg, environment=environment)
    assert catalog.readable_dirs == (str(package),)
    assert 'Body stays lazy' not in catalog.format_prompt_block()
    assert cfg.skills_readable_dirs == ()
    files = make_task_files(str(task), replace(cfg, skills_readable_dirs=catalog.readable_dirs),
                            environment=environment, readable_paths=catalog.readable_dirs)
    resource = files.readonly_view(str(package))
    assert resource.read_bytes('SKILL.md') == content.encode()
    with pytest.raises(PermissionError):
        resource.write_bytes('SKILL.md', b'changed')
    assert (package / 'SKILL.md').read_text() == content


def test_startup_conditional_rules_and_imports_use_selected_task_files(bwrap, tmp_path):
    import hashlib
    from tests._config_helpers import make_config
    from scripts.llm_solver.harness.task_path import activate_task_files
    from scripts.llm_solver.harness._loop._driver_setup import (
        load_session_injections, load_session_stream_rules,
    )
    source, files = namespace_files(bwrap, tmp_path)
    view = tmp_path / 'view'
    (source / '.git').mkdir()
    for tree, marker in ((source, 'Selected'), (view, 'Hidden host')):
        injections = tree / '.harness' / 'injections'
        injections.mkdir(parents=True)
        (injections / 'hint.md').write_text(
            '+++\nname = "hint"\ntrigger = "always"\n+++\n@../../guidance.md\n')
        (tree / 'guidance.md').write_text(marker + ' imported guidance')
        rules = tree / '.harness' / 'stream_rules'
        rules.mkdir()
        (rules / 'hint.md').write_text('+++\ncondition = "x"\n+++\n' + marker + ' stream guidance')
    cfg = make_config(injections_enabled=True, stream_rules_enabled=True, imports_enabled=True)
    with activate_task_files(files, host_root=view):
        injections, metadata = load_session_injections(cfg, view)
        rules, rule_files = load_session_stream_rules(cfg, view)
    assert injections[0].body == 'Selected imported guidance'
    assert rules[0].body == 'Selected stream guidance'
    assert rules[0].name == 'hint'
    assert metadata[0]['imported_bytes'] == len('Selected imported guidance')
    assert rule_files[0]['sha256'] == hashlib.sha256(
        (source / '.harness' / 'stream_rules' / 'hint.md').read_bytes()).hexdigest()


def test_path_guidance_targets_the_native_symlink_destination(bwrap, tmp_path):
    from scripts.llm_solver.harness.task_path import activate_task_files
    from scripts.llm_solver.harness.injections import path_targets_for_tool
    source, files = namespace_files(bwrap, tmp_path)
    view = tmp_path / 'view'
    (source / 'native.py').write_text('selected')
    (source / 'alias.py').symlink_to('native.py')
    (view / 'host.py').write_text('hidden')
    (view / 'alias.py').symlink_to('host.py')
    with activate_task_files(files, host_root=view):
        for path in ('alias.py', str(view / 'alias.py')):
            targets = path_targets_for_tool('read', {'path': path}, cwd=str(view))
            assert len(targets) == 1
            assert targets[0].path == 'native.py'
            assert targets[0].candidates == ('alias.py', 'native.py')


def test_ignore_policy_reads_rules_and_discovers_masks_in_task_view(bwrap, tmp_path):
    from scripts.llm_solver.harness.task_path import activate_task_files, TaskPath
    from scripts.llm_solver.harness.sandbox.ignore_policy import load_ignore_policy
    source, files = namespace_files(bwrap, tmp_path)
    view = tmp_path / 'view'
    (view / '.yujignore').write_text('host-only\n')
    (source / '.yujignore').write_text('private/\n!private/visible.py\n')
    (source / 'private').mkdir()
    (source / 'private' / 'visible.py').write_text('visible')
    (source / 'private' / 'secret.py').write_text('secret')
    with activate_task_files(files, host_root=view):
        policy = load_ignore_policy(view)
    # The loaded policy keeps its source view even outside the loading scope.
    assert policy.is_ignored('private/secret.py')
    assert not policy.is_ignored('private/visible.py')
    assert not policy.is_ignored('host-only')
    assert policy.existing_ignored_paths() == (str(files.root / 'private/secret.py'),)
    secret = TaskPath(files, files.root / 'private/secret.py')
    with pytest.raises(FileNotFoundError):
        policy.require_visible(secret)
    assert not policy.is_model_hidden(TaskPath(files, files.root / 'private'))
    (source / 'private' / 'visible.py').unlink()
    assert policy.existing_ignored_paths() == (str(files.root / 'private'),)
    assert policy.is_model_hidden(TaskPath(files, files.root / 'private'))


def test_dispatch_activates_native_access_and_resets_scope(bwrap, tmp_path):
    from tests._config_helpers import make_config
    from scripts.llm_solver.harness.tools import dispatch
    from scripts.llm_solver.harness.task_path import active_task_files
    (tmp_path / 'visible').write_text('visible bytes')
    (tmp_path / 'secret').write_text('host secret bytes')
    cfg = make_config(sandbox_bash=True, bwrap_bin=bwrap,
                      unreadable_paths=(str(tmp_path / 'secret'),))
    assert 'visible bytes' in dispatch('read', {'path': 'visible'}, cwd=str(tmp_path), cfg=cfg)
    assert 'host secret bytes' not in dispatch('read', {'path': 'secret'}, cwd=str(tmp_path), cfg=cfg)
    assert active_task_files(tmp_path) is None
    assert (tmp_path / 'secret').read_text() == 'host secret bytes'


def test_runtime_scope_rebinds_changed_permissions_and_restores_parent(bwrap, tmp_path):
    import os
    from dataclasses import replace
    from tests._config_helpers import make_config
    from scripts.llm_solver.harness.task_file_runtime import task_file_scope
    from scripts.llm_solver.harness.task_path import active_task_files
    (tmp_path / 'secret').write_text('parent-visible')
    cfg = make_config(sandbox_bash=True, bwrap_bin=bwrap, unreadable_paths=())
    environment = {'PATH': os.environ['PATH']}
    with task_file_scope(str(tmp_path), cfg, environment=environment) as parent:
        assert parent.read_bytes('secret') == b'parent-visible'
        with task_file_scope(str(tmp_path), cfg, environment=environment) as same:
            assert same is parent
        restricted = replace(cfg, unreadable_paths=(str(tmp_path / 'secret'),))
        with task_file_scope(str(tmp_path), restricted, environment=environment) as child:
            assert child is not parent
            with pytest.raises(TaskFileError):
                child.read_bytes('secret')
        assert active_task_files(tmp_path) is parent
        from scripts.llm_solver.harness.task_environment import TaskEnvironmentUnavailable
        with pytest.raises(TaskEnvironmentUnavailable, match='sandbox disabled'):
            with task_file_scope(str(tmp_path), replace(cfg, sandbox_bash=False)):
                pytest.fail('a nested scope replaced the selected native executor')
        assert active_task_files(tmp_path) is parent
    assert active_task_files(tmp_path) is None


def test_startup_ignore_loader_obeys_configured_execution_mask(bwrap, tmp_path):
    import os
    from tests._config_helpers import make_config
    from scripts.llm_solver.harness.task_file_runtime import load_task_ignore_policy
    from scripts.llm_solver.harness.sandbox.ignore_policy import IgnorePolicyError
    (tmp_path / '.yujignore').write_text('secret\n')
    cfg = make_config(sandbox_bash=True, bwrap_bin=bwrap,
                      unreadable_paths=(str(tmp_path / '.yujignore'),))
    with pytest.raises(IgnorePolicyError, match='not a regular file'):
        load_task_ignore_policy(
            str(tmp_path), cfg, environment={'PATH': os.environ['PATH']},
            allow_login_shell=False,
        )
    assert (tmp_path / '.yujignore').read_text() == 'secret\n'


def test_external_readonly_resource_uses_the_same_namespace(bwrap, tmp_path):
    from scripts.llm_solver.harness._tools._common import _resolve_read
    from scripts.llm_solver.harness.task_path import activate_task_files
    source, files = namespace_files(bwrap, tmp_path)
    resource = tmp_path / 'resource'
    resource.mkdir()
    (resource / 'guide').write_text('hidden host guide')
    visible = tmp_path / 'visible-resource'
    visible.mkdir()
    (visible / 'guide').write_text('selected guide')
    (visible / 'escape').symlink_to(source / 'private')
    (source / 'private').write_text('outside resource')

    def run(script, args, data):
        return subprocess.run(
            [bwrap, '--ro-bind', '/', '/', '--unshare-net',
             '--bind', str(source), str(files.root),
             '--ro-bind', str(visible), str(resource),
             '--', 'bash', '--noprofile', '--norc', '-c', script,
             'resource-test', *args], input=data, capture_output=True,
        )
    files = NamespaceFiles(str(files.root), run, binding=files.binding)
    with activate_task_files(files, host_root=tmp_path / 'view'):
        target = _resolve_read(str(tmp_path / 'view'), str(resource / 'guide'),
                               readonly_roots=(str(resource),))
        assert target.read_text() == 'selected guide'
        with pytest.raises(PermissionError):
            target.write_text('must not change')
        with pytest.raises(PermissionError):
            _resolve_read(str(tmp_path / 'view'), str(resource / 'escape'),
                          readonly_roots=(str(resource),))
    assert (resource / 'guide').read_text() == 'hidden host guide'
    assert (visible / 'guide').read_text() == 'selected guide'


def test_read_write_edit_and_glob_share_the_selected_view(bwrap, tmp_path):
    from scripts.llm_solver.harness.task_path import activate_task_files
    from scripts.llm_solver.harness._tools.read import read
    from scripts.llm_solver.harness._tools.write import write
    from scripts.llm_solver.harness._tools.edit import edit
    from scripts.llm_solver.harness._tools.glob import glob_files

    overlay = tmp_path / 'overlay'
    overlay.mkdir()
    (overlay / 'visible.txt').write_text('container content')
    source, files = namespace_files(bwrap, tmp_path, overlay=overlay)
    (source / 'nested' / 'hidden.txt').write_text('host answer')
    with activate_task_files(files, host_root=source):
        result = read('nested/visible.txt', cwd=str(source))
        assert result == '1: container content'
        assert result.inspection_evidence['namespace'] == 'task_execution'
        assert result.inspection_evidence['task_view'] == files.binding
        assert 'host answer' not in read('nested/hidden.txt', cwd=str(source))
        assert glob_files('**/*.txt', cwd=str(source)) == 'nested/visible.txt'
        assert write('created.txt', 'before', cwd=str(source)).startswith('OK')
        assert edit('created.txt', 'before', 'after', cwd=str(source)).startswith('OK')
        assert read('created.txt', cwd=str(source)) == '1: after'
        assert write('nested/visible.txt', 'replace', cwd=str(source)).startswith('ERROR')
    assert (source / 'created.txt').read_text() == 'after'
    assert (overlay / 'visible.txt').read_text() == 'container content'
    assert (source / 'nested' / 'hidden.txt').read_text() == 'host answer'


def test_file_tools_preserve_readonly_task_and_refuse_host_coercion(bwrap, tmp_path):
    from pathlib import Path
    from scripts.llm_solver.harness.task_path import activate_task_files, bound_task_path
    from scripts.llm_solver.harness._tools.write import write
    from scripts.llm_solver.harness._tools.edit import edit

    source, files = namespace_files(bwrap, tmp_path, readonly=True)
    (source / 'file').write_text('original')
    with activate_task_files(files, host_root=source):
        assert write('file', 'changed', cwd=str(source)).startswith('ERROR')
        assert edit('file', 'original', 'changed', cwd=str(source)).startswith('ERROR')
        with pytest.raises(TypeError):
            Path(bound_task_path(str(source), 'file'))
    assert (source / 'file').read_text() == 'original'


def test_search_uses_namespace_bytes_and_keeps_aliases(bwrap, tmp_path):
    from scripts.llm_solver.harness.task_path import activate_task_files
    from scripts.llm_solver.harness._tools.grep import grep_files
    from scripts.llm_solver.harness._tools.glob import glob_files

    overlay = tmp_path / 'overlay'
    overlay.mkdir()
    (overlay / 'visible.txt').write_text('needle\n')
    source, files = namespace_files(bwrap, tmp_path, overlay=overlay)
    (source / 'nested' / 'hidden.txt').write_text('needle host answer\n')
    (source / 'link').symlink_to('nested')
    with activate_task_files(files, host_root=source):
        assert glob_files('*/*.txt', cwd=str(source)).splitlines() == [
            'link/visible.txt', 'nested/visible.txt',
        ]
        matches = grep_files('needle', cwd=str(source))
        assert 'host answer' not in matches and 'hidden.txt' not in matches
        assert './nested/visible.txt:1:needle' in matches
        assert grep_files('absent', cwd=str(source)) == 'No matches found.'


def test_glob_uses_one_operation_for_a_tree_and_sees_later_changes(bwrap, tmp_path):
    from scripts.llm_solver.harness.task_path import activate_task_files
    from scripts.llm_solver.harness._tools.glob import glob_files
    source, files = namespace_files(bwrap, tmp_path)
    for index in range(40):
        folder = source / f'folder-{index:02}' / 'units'
        folder.mkdir(parents=True)
        (folder / 'example.py').write_text('')
        (folder / 'unrelated.txt').write_text('')
    operations = []
    run = files.run
    def observed(script, args, data):
        operations.append(args)
        return run(script, args, data)
    files.run = observed
    with activate_task_files(files, host_root=source):
        first = glob_files('**/units/*.py', cwd=str(source)).splitlines()
        assert len(first) == 40
        # Utility discovery and scope resolution are constant startup costs.
        assert len(operations) < 10
        assert sum(len(args) > 3 and args[3] == 'glob' for args in operations) == 1
        (source / 'folder-00' / 'units' / 'later.py').write_text('')
        second = glob_files('**/units/*.py', cwd=str(source)).splitlines()
        assert len(second) == 41
        assert 'folder-00/units/later.py' in second


@pytest.mark.parametrize('readonly', [False, True])
def test_patch_formats_use_task_view_and_preserve_readonly_mount(bwrap, tmp_path, readonly):
    from scripts.llm_solver.harness.task_path import activate_task_files
    from scripts.llm_solver.harness.apply_patch import parse_patch, verify_and_apply, PatchVerifyError
    from scripts.llm_solver.harness.udiff import parse_unified_diff, verify_and_apply_unified_diff, UnifiedDiffApplyError

    source, files = namespace_files(bwrap, tmp_path, readonly=readonly)
    (source / 'file').write_text('before\n')
    patch = parse_patch('*** Begin Patch\n*** Update File: file\n@@\n-before\n+after\n*** End Patch\n')
    diff = parse_unified_diff('--- a/file\n+++ b/file\n@@ -1 +1 @@\n-after\n+final\n')
    with activate_task_files(files, host_root=source):
        if readonly:
            with pytest.raises(PatchVerifyError) as denied:
                verify_and_apply(patch, str(source))
            assert denied.value.kind == 'write_failed'
            diff = parse_unified_diff('--- a/file\n+++ b/file\n@@ -1 +1 @@\n-before\n+final\n')
            with pytest.raises(UnifiedDiffApplyError) as denied:
                verify_and_apply_unified_diff(diff, str(source))
            assert denied.value.kind == 'write_failed'
        else:
            verify_and_apply(patch, str(source))
            verify_and_apply_unified_diff(diff, str(source))
    assert (source / 'file').read_text() == ('before\n' if readonly else 'final\n')


def test_working_set_rereads_the_selected_task_view(bwrap, tmp_path):
    from scripts.llm_solver.harness.task_path import activate_task_files
    from scripts.llm_solver.harness.context_strategies._working_set import WorkingSet
    overlay = tmp_path / 'overlay'
    overlay.mkdir()
    (overlay / 'file').write_text('visible')
    source, files = namespace_files(bwrap, tmp_path, overlay=overlay)
    (source / 'nested/file').write_text('hidden host content')
    working_set = WorkingSet(source)
    with activate_task_files(files, host_root=source):
        working_set.record_mutation('nested/file', turn=1)
        assert working_set.files['nested/file'].content == 'visible'
        (overlay / 'file').write_text('updated visible bytes')
        assert working_set._read_disk('nested/file') == 'updated visible bytes'
        assert working_set._read_disk('../outside') is None


def test_symbol_tools_and_index_read_only_namespace_sources(bwrap, tmp_path, monkeypatch):
    from scripts.llm_solver.harness.task_path import activate_task_files, bound_task_path
    from scripts.llm_solver.harness._tools.list_definitions import list_definitions
    from scripts.llm_solver.harness import structural_index
    from tests._config_helpers import make_config
    import ast

    class Extractor:
        def detect_language(self, path):
            return 'python' if path.suffix == '.py' else None

        def extract(self, source, *, language, display_path):
            return tuple(structural_index.StructuralRow(
                path=display_path, line=node.lineno, column=1, kind='def',
                name=node.name, signature='def ' + node.name,
                language=language, capture='definition.function',
            ) for node in ast.parse(source).body if isinstance(node, ast.FunctionDef))

    monkeypatch.setattr(structural_index, 'TreeSitterTagExtractor', Extractor)
    overlay = tmp_path / 'overlay'
    overlay.mkdir()
    (overlay / 'module.py').write_text('def visible():\n    pass\n')
    source, files = namespace_files(bwrap, tmp_path, overlay=overlay)
    (source / 'nested/module.py').write_text('def hidden_host_answer():\n    pass\n')
    cfg = make_config(tools_list_definitions_enabled=True,
                      tools_ast_search_enabled=True, unreadable_paths=())
    with activate_task_files(files, host_root=source):
        single = list_definitions('nested/module.py', cwd=str(source), cfg=cfg)
        whole = list_definitions('.', cwd=str(source), cfg=cfg, repo_wide=True)
        for output in (single, whole):
            assert 'visible' in output and 'hidden_host_answer' not in output
        index = structural_index.StructuralIndex(bound_task_path(str(source), '.'))
        assert index.scan().rows[0].name == 'visible'
        before = index.fingerprint(contents=False)
        (overlay / 'module.py').write_text('def new_visible_name():\n    pass\n')
        assert index.fingerprint(contents=False) != before
        assert index.scan().rows[0].name == 'new_visible_name'
        assert files.metadata('nested/module.py').mtime_ns == (overlay / 'module.py').stat().st_mtime_ns
