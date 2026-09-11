"""Bind task file operations to the same configured execution dispatcher."""
from dataclasses import asdict, dataclass
from contextlib import contextmanager, nullcontext
from functools import wraps
import hashlib
import json
import os
import shlex

from .task_files import NamespaceFiles
from .sandbox.env_policy import build_bash_argv
from .task_environment import discover_task_environment
from .time_budget import execution_deadline, remaining_before
from .container_binding import (
    container_image_scope, bind_container_image, current_container_image_binding,
)


@dataclass(frozen=True)
class TaskFileExecution:
    """Execution inputs for consumers that do not own a solver Config."""

    sandbox_bash: bool
    sandbox_backend: str
    bwrap_bin: str
    sandbox_container_runtime: str
    sandbox_container_image: str
    sandbox_container_flags: tuple[str, ...]


def _execution_selection(cfg, environment, allow_login_shell):
    """Describe execution inputs independently of per-operation path masks."""
    from .sandbox import container_mode
    return (
        cfg.sandbox_bash, cfg.sandbox_backend, cfg.bwrap_bin,
        cfg.sandbox_container_runtime, cfg.sandbox_container_image,
        tuple(cfg.sandbox_container_flags or ()),
        container_mode() if cfg.sandbox_bash else None,
        tuple(sorted(environment.items())) if environment is not None else None,
        bool(allow_login_shell),
    )


@container_image_scope()
def make_task_files(cwd, cfg, *, environment, allow_login_shell=False,
                    unreadable_paths=(), readable_paths=(), persistent=True):
    from ._tools._run_in_sandbox import _run_in_sandbox
    from .sandbox import AMBIENT_CONTAINER, container_mode
    from .task_path import capture_task_execution_paths, capture_host_task_root
    from .sandbox.policy import sandbox_execution_kwargs

    execution = sandbox_execution_kwargs(cfg)
    cfg = TaskFileExecution(
        sandbox_bash=execution['sandbox'],
        sandbox_backend=execution['sandbox_backend'],
        bwrap_bin=cfg.bwrap_bin,
        sandbox_container_runtime=execution['container_runtime'],
        sandbox_container_image=execution['container_image'],
        sandbox_container_flags=execution['container_flags'],
    )

    cwd, unreadable_paths, readable_paths = capture_task_execution_paths(
        cwd, unreadable_paths, readable_paths)
    mode = container_mode() if cfg.sandbox_bash else None
    host_task_root = capture_host_task_root(
        cwd, sandbox=cfg.sandbox_bash, sandbox_backend=cfg.sandbox_backend)
    from .sandbox._filesystem import capture_frozen_filesystem_view
    filesystem_view = capture_frozen_filesystem_view(host_task_root)
    image_binding = current_container_image_binding()
    task = None
    from .task_environment import docker_client_fingerprint
    selected_docker_client = docker_client_fingerprint()
    uses_docker = cfg.sandbox_bash and (
        mode not in (None, AMBIENT_CONTAINER)
        or (cfg.sandbox_backend == 'container' and cfg.sandbox_container_runtime == 'docker')
    )
    if mode and mode != AMBIENT_CONTAINER:
        task = discover_task_environment(cwd)
        root = task.working_directory
        binding = asdict(task)
    else:
        root = os.path.abspath(cwd)
        binding = {'host_root': cwd,
                   'working_directory': root, 'container': mode or '',
                   'sandbox_backend': cfg.sandbox_backend,
                   'sandbox_enabled': cfg.sandbox_bash}
    if uses_docker:
        binding['docker_client_fingerprint'] = selected_docker_client
    if cfg.sandbox_bash and cfg.sandbox_backend == 'container':
        if mode is not None:
            raise RuntimeError('container backend cannot be combined with legacy YUJ_CONTAINER')
        from .sandbox.container_backend import ContainerBackend
        backend = ContainerBackend(cfg.sandbox_container_runtime,
                                   cfg.sandbox_container_image,
                                   tuple(cfg.sandbox_container_flags or ()))
        runtime_bin = execution['container_runtime_bin'] or backend.resolve_runtime(sandbox_required=True)
        bound = bind_container_image(backend, runtime_bin)
        binding.update(container_runtime=bound.runtime,
                       container_runtime_bin=runtime_bin,
                       container_image_digest=bound.image,
                       container_flags=bound.flags)
    captured_environment = dict(environment) if environment is not None else None
    if captured_environment is not None:
        # Receipts need to distinguish selected environments without storing
        # their values. This fingerprints inputs, not a live mount or process.
        selection = _execution_selection(cfg, captured_environment, allow_login_shell)
        binding['execution_selection_sha256'] = hashlib.sha256(
            json.dumps(selection, separators=(',', ':')).encode('ascii')).hexdigest()

    def run(script, args, data):
        if host_task_root is not None:
            host_task_root.verify()
        if (container_mode() if cfg.sandbox_bash else None) != mode or (
            uses_docker and docker_client_fingerprint() != selected_docker_client
        ):
            from .task_environment import TaskEnvironmentUnavailable
            raise TaskEnvironmentUnavailable('task execution selection changed after file binding')
        command = shlex.join([*build_bash_argv(script), 'yuj-file-operation', *args])
        from .task_environment import use_task_environment
        with (container_image_scope(image_binding)
              if cfg.sandbox_bash and cfg.sandbox_backend == 'container' else nullcontext()), (
            use_task_environment(task) if task is not None else nullcontext()
        ):
            return execute(command, data)

    def execute(command, data):
        return _run_in_sandbox(
            command, cwd=cwd, timeout=remaining_before(execution_deadline()),
            sandbox=cfg.sandbox_bash, bwrap_bin=cfg.bwrap_bin,
            # An intentional unsandboxed configuration is distinct from a
            # failed selected backend. File access cannot silently downgrade.
            sandbox_required=bool(cfg.sandbox_bash),
            sandbox_backend=cfg.sandbox_backend,
            container_runtime=cfg.sandbox_container_runtime,
            container_runtime_bin=execution['container_runtime_bin'],
            container_image=cfg.sandbox_container_image,
            container_flags=tuple(cfg.sandbox_container_flags or ()),
            unreadable_paths=tuple(unreadable_paths),
            readable_paths=tuple(readable_paths),
            effective_env=captured_environment,
            allow_login_shell=allow_login_shell,
            raw_result=True, input_bytes=data, use_persistent=persistent,
            _host_task_root=host_task_root,
            _filesystem_view=filesystem_view,
        )

    local_kernel = (not cfg.sandbox_bash or mode == AMBIENT_CONTAINER or
                    mode is None and cfg.sandbox_backend == 'bwrap')
    files = NamespaceFiles(root, run, binding=binding, shares_host_kernel=local_kernel)
    files._host_task_root = host_task_root
    files._filesystem_view = filesystem_view
    files._task_environment = task
    if cfg.sandbox_bash and cfg.sandbox_backend == 'container':
        files._container_image_binding = image_binding
    files.unreadable_paths = tuple(unreadable_paths)
    return files


@contextmanager
def task_file_scope(cwd, cfg, *, environment=None, allow_login_shell=None,
                    ignore_policy=None, persistent=True):
    from .sandbox.ignore_policy import activate_ignore_policy
    from .task_path import active_task_files
    binding = getattr(active_task_files(cwd), '_container_image_binding', None)
    with container_image_scope(binding), activate_ignore_policy(ignore_policy), _task_file_scope(
        cwd, cfg, environment=environment, allow_login_shell=allow_login_shell,
        ignore_policy=ignore_policy, persistent=persistent,
    ) as files:
        yield files


@contextmanager
def _task_file_scope(cwd, cfg, *, environment=None, allow_login_shell=None,
                     ignore_policy=None, persistent=True):
    from .task_path import activate_task_files, active_task_files
    from .tools import _effective_command_environment, _bash_unreadable_paths, _bash_readable_paths
    existing = active_task_files(cwd)
    if existing is not None and not hasattr(existing, '_runtime_key'):
        # An explicitly supplied executor already owns its namespace contract.
        yield existing
        return
    if not cfg.sandbox_bash:
        if existing is not None:
            from .task_environment import TaskEnvironmentUnavailable
            raise TaskEnvironmentUnavailable(
                'task execution selection changed within the file scope: sandbox disabled')
        # The harness and commands deliberately share the local namespace.
        with activate_task_files(None, host_root=cwd):
            yield None
        return
    from .sandbox import container_mode
    from .task_environment import TaskEnvironmentUnavailable, docker_client_fingerprint
    if existing is not None and (
        existing.binding.get('container', '') != (container_mode() or '')
        or (existing.binding.get('docker_client_fingerprint') and
            existing.binding['docker_client_fingerprint'] != docker_client_fingerprint())
    ):
        raise TaskEnvironmentUnavailable('task execution selection changed within the file scope')
    if environment is None:
        environment, allow_login_shell = _effective_command_environment(cfg, cwd=cwd)
    execution_key = _execution_selection(cfg, environment, allow_login_shell)
    if existing is not None and existing._execution_key != execution_key:
        raise TaskEnvironmentUnavailable(
            'task execution selection changed within the file scope: backend or environment')
    unreadable_paths = _bash_unreadable_paths(cwd, cfg, ignore_policy)
    readable_paths = _bash_readable_paths(cfg)
    key = (execution_key, unreadable_paths, readable_paths, persistent)
    if existing is not None and existing._runtime_key == key:
        yield existing
        return
    files = make_task_files(
        cwd, cfg, environment=environment,
        allow_login_shell=bool(allow_login_shell),
        unreadable_paths=unreadable_paths,
        readable_paths=readable_paths,
        persistent=persistent,
    )
    files._runtime_key = key
    files._execution_key = execution_key
    with activate_task_files(files, host_root=cwd):
        yield files


def file_scoped_dispatch(function):
    @wraps(function)
    def scoped(*args, **kwargs):
        with task_file_scope(
            kwargs['cwd'], kwargs['cfg'],
            environment=kwargs.get('effective_env'),
            allow_login_shell=kwargs.get('allow_login_shell'),
            ignore_policy=kwargs.get('ignore_policy'),
        ):
            return function(*args, **kwargs)
    return scoped


def load_task_ignore_policy(cwd, cfg, *, environment, allow_login_shell):
    """Read startup rules through the command view before adding their masks."""
    from .sandbox.ignore_policy import load_ignore_policy, PROJECT_INIT_PRIVATE_RULES
    with task_file_scope(cwd, cfg, environment=environment,
                         allow_login_shell=allow_login_shell, persistent=False):
        return load_ignore_policy(
            cwd, enabled=getattr(cfg, 'state_ignore_file_enabled', True),
            file_names=getattr(cfg, 'state_ignore_file_names', ('.yujignore',)),
            builtin_rules=(PROJECT_INIT_PRIVATE_RULES
                           if getattr(cfg, 'assistant_project_init_destination', '') else ()),
        )


def file_scoped_session(function):
    @wraps(function)
    def scoped(session, *args, **kwargs):
        retained = getattr(session, '_task_file_scope', nullcontext)
        with retained(), task_file_scope(
            session.cwd, session.cfg, environment=session._effective_env,
            allow_login_shell=session._allow_login_shell,
            ignore_policy=session._ignore_policy,
        ):
            return function(session, *args, **kwargs)
    return scoped
