"""Observe the task view to which a durable workspace checkpoint belongs."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess

from .process_identity import OBSERVE_SCRIPT, parse_process_identity
from .task_files import NamespaceFiles, TaskFileError
from .task_path import NativeUnreadableMatcher, bound_task_path
from .task_view import task_view_identity
from .time_budget import execution_deadline, remaining_before


def checkpoint_hidden_paths(workspace):
    """Resolve the selected executor's declared masks in its native view."""
    native = bound_task_path(str(workspace), '.')
    if native is None:
        return ()
    matcher = NativeUnreadableMatcher(native, getattr(native.files, 'unreadable_paths', ()))
    paths = []
    for path in matcher.blocked:
        if path.is_relative_to(native):
            paths.append(path.relative_to(native).as_posix())
    return tuple(sorted(set(paths)))


def checkpoint_task_binding(workspace, *, hidden_paths=(), excludes=()):
    native = bound_task_path(str(workspace), '.')
    if native is None:
        root = Path(workspace).resolve()

        def run(script, args, data):
            return subprocess.run(
                ['bash', '--noprofile', '--norc', '-p', '-c', script,
                 'yuj-checkpoint-binding', *args], cwd=root, input=data,
                capture_output=True, timeout=remaining_before(execution_deadline()),
            )

        files = NamespaceFiles(str(root), run, binding={'access': 'host'})
        access = 'host'
    else:
        files, root, access = native.files, native.path, 'native'
    # Checkpoints exclude only the root .git subtree, not nested namesakes.
    view = task_view_identity(files, root=root, excluded_paths=('.git', *hidden_paths))
    identity = files.run(OBSERVE_SCRIPT, [], None)
    if identity.returncode:
        raise TaskFileError('checkpoint process credential observation unavailable')
    credentials = parse_process_identity(identity.stdout)
    executor = json.dumps(files.binding, sort_keys=True, separators=(',', ':')).encode()
    return {
        'version': 1,
        'access': access,
        'view_sha256': view,
        'executor_sha256': hashlib.sha256(executor).hexdigest(),
        'process_identity': asdict(credentials),
        'hidden_paths': list(hidden_paths),
        'checkpoint_excludes': list(excludes),
    }
