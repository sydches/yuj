"""Private undo records for interrupted or failed workspace restores."""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import PurePosixPath
import shutil
import stat

from .task_path import TaskPath
from .time_budget import execution_deadline, remaining_before

DIRECTORY = '.restore_recovery'


@contextmanager
def restore_lock(store):
    from .workspace_checkpoints import WorkspaceCheckpointError
    store._ensure_initialized()
    with (store.shadow_dir / '.restore.lock').open('a+b') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise WorkspaceCheckpointError('another workspace restore is active') from error
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def entry_state(path):
    """Read the entry itself; a symlink's target is never the backup source."""
    remaining_before(execution_deadline())
    try:
        info = path.lstat()
    except (FileNotFoundError, NotADirectoryError):
        return {'kind': 'missing'}, None
    if stat.S_ISDIR(info.st_mode):
        return {'kind': 'directory', 'mode': stat.S_IMODE(info.st_mode)}, None
    if stat.S_ISREG(info.st_mode):
        kind, data = 'file', path.read_bytes()
    elif stat.S_ISLNK(info.st_mode):
        kind = 'symlink'
        data = os.fsencode(path.files.readlink(str(path)) if isinstance(path, TaskPath)
                           else os.readlink(path))
    else:
        raise OSError(f'unsupported restore recovery entry: {path}')
    return {'kind': kind, 'mode': stat.S_IMODE(info.st_mode),
            'sha256': hashlib.sha256(data).hexdigest()}, data


def _signature(entry):
    return {key: value for key, value in entry.items() if key != 'blob'}


def _parents(path):
    parent = PurePosixPath(path).parent
    while str(parent) != '.':
        yield parent.as_posix()
        parent = parent.parent


class RestoreJournal:
    def __init__(self, store, root, state):
        self.store, self.root, self.state = store, root, state
        self.directory = store.shadow_dir / DIRECTORY

    @classmethod
    def load(cls, store, root):
        from .workspace_checkpoints import WorkspaceCheckpointError, _safe_relative_path
        directory = store.shadow_dir / DIRECTORY
        if not directory.exists():
            return None
        try:
            state = json.loads((directory / 'state.json').read_text())
            if state['version'] != 1 or state['phase'] not in (
                'preparing', 'prepared', 'applying', 'incomplete', 'complete',
            ):
                raise ValueError('invalid restore recovery state')
            for path, entry in state['before'].items():
                _safe_relative_path(path)
                if entry.get('blob') and not (
                    entry['blob'].startswith('blob-') and entry['blob'][5:].isdigit()
                ):
                    raise ValueError('invalid recovery blob name')
            for operation in state['operations']:
                if operation['path'] not in state['before']:
                    raise ValueError('recovery operation has no prior entry')
            return cls(store, root, state)
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise WorkspaceCheckpointError(f'restore recovery record unavailable: {directory}') from error

    @classmethod
    def prepare(cls, store, root, paths, binding, commit):
        directory = store.shadow_dir / DIRECTORY
        directory.mkdir(mode=0o700)
        state = {'version': 1, 'phase': 'preparing', 'binding': binding,
                 'checkpoint': commit, 'before': {}, 'operations': []}
        journal = cls(store, root, state)
        try:
            journal.save()
            index = store.shadow_dir / 'index'
            if index.exists():
                data = index.read_bytes()
                with (directory / 'index-before').open('xb') as output:
                    output.write(data)
                    output.flush()
                    os.fsync(output.fileno())
                state['index_sha256'] = hashlib.sha256(data).hexdigest()
            paths = set(paths)
            paths.update(parent for path in tuple(paths) for parent in _parents(path))
            for path in sorted(paths):
                entry, data = entry_state(root / path)
                if data is not None:
                    name = f'blob-{len(state["before"])}'
                    with (directory / name).open('xb') as output:
                        output.write(data)
                        output.flush()
                        os.fsync(output.fileno())
                    entry['blob'] = name
                state['before'][path] = entry
            state['phase'] = 'prepared'
            journal.save()
            return journal
        except BaseException:
            # No task mutation is allowed until prepare returns.
            shutil.rmtree(directory)
            raise

    def save(self):
        temporary = self.directory / '.state-next'
        with temporary.open('w') as output:
            json.dump(self.state, output, sort_keys=True)
            output.write('\n')
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, self.directory / 'state.json')
        fd = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def record(self, path, *, data=None, mode=None):
        remaining_before(execution_deadline())
        after = {'kind': 'missing'} if data is None else {
            'kind': 'symlink' if mode == 0o120000 else 'file',
            'mode': 0o777 if mode == 0o120000 else stat.S_IMODE(mode),
            'sha256': hashlib.sha256(data).hexdigest(),
        }
        self.state['operations'].append({'path': path, 'after': after})
        self.state['phase'] = 'applying'
        self.save()

    def finish(self):
        self.state['phase'] = 'complete'
        self.save()
        shutil.rmtree(self.directory)

    def recover(self):
        """Undo recorded operations, refusing unexpected current file contents."""
        from .workspace_checkpoints import WorkspaceCheckpointError
        try:
            if self.state['phase'] == 'complete' or not self.state['operations']:
                self.finish()
                return
            self.store._require_task_binding(self.state['binding'])
            operations = {operation['path']: operation['after']
                          for operation in self.state['operations']}
            affected = set(operations)
            affected.update(parent for path in operations for parent in _parents(path))
            current = {}
            for path in affected:
                before = _signature(self.state['before'][path])
                now, _ = entry_state(self.root / path)
                current[path] = now
                if now == before or now == operations.get(path):
                    continue
                # A file/link replacement can fail after removing an empty
                # directory or old link. Parent creation/removal is also part
                # of the existing restore implementation.
                if now['kind'] == 'missing':
                    continue
                if path not in operations and before['kind'] == 'missing' and now['kind'] == 'directory':
                    continue
                if now['kind'] == 'directory' and any(child.startswith(path + '/') for child in operations):
                    if before['kind'] != 'directory' or now == before:
                        continue
                raise WorkspaceCheckpointError(f'file changed outside recorded restore: {path}')

            # Validate every required backup before recovery itself changes
            # task files. A damaged later blob must not cause another partial
            # restore merely because earlier backups were readable.
            for path in affected:
                remaining_before(execution_deadline())
                before = self.state['before'][path]
                if 'blob' in before:
                    data = (self.directory / before['blob']).read_bytes()
                    if hashlib.sha256(data).hexdigest() != before['sha256']:
                        raise WorkspaceCheckpointError(f'restore recovery bytes changed: {path}')
            if 'index_sha256' in self.state:
                data = (self.directory / 'index-before').read_bytes()
                if hashlib.sha256(data).hexdigest() != self.state['index_sha256']:
                    raise WorkspaceCheckpointError('restore recovery index bytes changed')

            deepest = sorted(affected, key=lambda path: (-len(PurePosixPath(path).parts), path))
            for path in deepest:
                before, now = self.state['before'][path], current[path]
                if now['kind'] == 'missing' or now['kind'] == before['kind']:
                    continue
                target = self.root / path
                target.rmdir() if now['kind'] == 'directory' else target.unlink()

            created_directories = []
            for path in reversed(deepest):
                before = self.state['before'][path]
                if before['kind'] == 'directory' and not (self.root / path).is_dir():
                    (self.root / path).mkdir()
                    created_directories.append(path)
            for path in reversed(deepest):
                before = self.state['before'][path]
                if before['kind'] in ('missing', 'directory'):
                    continue
                target = self.root / path
                now, _ = entry_state(target)
                if now == _signature(before):
                    continue
                data = (self.directory / before['blob']).read_bytes()
                if hashlib.sha256(data).hexdigest() != before['sha256']:
                    raise WorkspaceCheckpointError(f'restore recovery bytes changed: {path}')
                if before['kind'] == 'symlink':
                    if now['kind'] != 'missing':
                        target.unlink()
                    target.symlink_to(os.fsdecode(data))
                else:
                    self.store._write_regular_file(target, data, before['mode'])
            for path in created_directories:
                (self.root / path).chmod(self.state['before'][path]['mode'])
            for path in affected:
                now, _ = entry_state(self.root / path)
                if now != _signature(self.state['before'][path]):
                    raise WorkspaceCheckpointError(f'restore recovery incomplete: {path}')
            self.store._require_task_binding(self.state['binding'])
            if 'index_sha256' in self.state:
                data = (self.directory / 'index-before').read_bytes()
                if hashlib.sha256(data).hexdigest() != self.state['index_sha256']:
                    raise WorkspaceCheckpointError('restore recovery index bytes changed')
                temporary = self.store.shadow_dir / '.index-recovery-next'
                with temporary.open('wb') as output:
                    output.write(data)
                    output.flush()
                    os.fsync(output.fileno())
                os.replace(temporary, self.store.shadow_dir / 'index')
            self.finish()
        except Exception as error:
            self.state['phase'] = 'incomplete'
            self.state['recovery_error'] = f'{type(error).__name__}: {error}'
            if self.directory.exists():
                self.save()
            raise
