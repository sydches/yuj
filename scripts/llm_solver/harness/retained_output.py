"""Save output at a path readable through the selected task executor."""
from contextlib import nullcontext
import logging
from pathlib import Path

from .task_path import TaskPath, resolve_task_path
from .time_budget import BudgetExhausted

log = logging.getLogger(__name__)


def output_file_scope(session):
    from .task_file_runtime import task_file_scope
    if not getattr(getattr(session, 'cfg', None), 'sandbox_bash', False):
        return nullcontext()
    return task_file_scope(
        session.cwd, session.cfg, environment=getattr(session, '_effective_env', None),
        allow_login_shell=getattr(session, '_allow_login_shell', None),
        ignore_policy=getattr(session, '_ignore_policy', None),
    )


def save_output(session, text, turn, *, trace=False):
    """Return a relative path only after an exclusive native/local write."""
    session._sink_counter += 1
    try:
        with output_file_scope(session):
            root = resolve_task_path(session.cwd, '.')
            directory = root / '.tool_output'
            if directory.is_symlink():
                return ''
            directory = resolve_task_path(root, '.tool_output')
            directory.mkdir(parents=True, exist_ok=True)
            while True:
                suffix = '_trace' if trace else ''
                name = f'{session._session_number}_{session._sink_counter:04d}_t{turn}{suffix}.log'
                path = directory / name
                try:
                    data = text.encode('utf-8', errors='replace')
                    if isinstance(path, TaskPath):
                        path.files.create_bytes(str(path), data)
                    else:
                        with path.open('xb') as saved:
                            saved.write(data)
                    break
                except FileExistsError:
                    session._sink_counter += 1
            from .worktree_runtime import exclude_created_runtime_file
            exclude_created_runtime_file(Path(session.cwd), path)
            return str(path.relative_to(root))
    except (OSError, ValueError, BudgetExhausted) as error:
        log.debug('Output save unavailable: %s', error)
        return ''
