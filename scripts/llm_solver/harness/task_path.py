"""Path-like task access that cannot silently become a host filesystem path."""
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path, PurePosixPath
from dataclasses import dataclass
from functools import cache
from types import SimpleNamespace
import fnmatch
import io
import os

from .task_files import NamespaceFiles, TaskFileError


from ._task_host_root import HostTaskRoot, capture_host_task_root

@dataclass(frozen=True)
class TaskPath:
    files: NamespaceFiles
    path: PurePosixPath

    def __str__(self):
        return str(self.path)

    def __fspath__(self):
        raise TypeError('task paths require namespace access, not host filesystem I/O')

    def __truediv__(self, child):
        return TaskPath(self.files, self.path / child)

    def __lt__(self, other):
        return self.path < other.path

    @property
    def name(self):
        return self.path.name

    @property
    def suffix(self):
        return self.path.suffix

    @property
    def stem(self):
        return self.path.stem

    @property
    def parent(self):
        return TaskPath(self.files, self.path.parent)

    @property
    def parents(self):
        return tuple(TaskPath(self.files, p) for p in self.path.parents)

    def relative_to(self, other):
        if isinstance(other, TaskPath):
            if self.files is not other.files:
                raise ValueError('paths belong to different task views')
            other = other.path
        return self.path.relative_to(other)

    def is_relative_to(self, other):
        try:
            self.relative_to(other)
            return True
        except ValueError:
            return False

    def as_posix(self):
        return self.path.as_posix()

    def as_uri(self):
        return self.path.as_uri()

    def resolve(self, strict=False):
        resolved = self.files.resolve(str(self.path))
        if strict and self.files.kind(str(resolved)) == 'missing':
            raise FileNotFoundError(str(resolved))
        return TaskPath(self.files, resolved)

    def exists(self):
        return self.files.kind(str(self.path)) != 'missing'

    def is_file(self):
        return self.files.kind(str(self.path)) == 'file'

    def is_dir(self):
        return self.files.kind(str(self.path)) == 'directory'

    def is_symlink(self):
        return self.files.is_symlink(str(self.path))

    def stat(self):
        return self._stat(follow_symlinks=True)

    def lstat(self):
        return self._stat(follow_symlinks=False)

    def readlink(self):
        return PurePosixPath(self.files.readlink(str(self.path)))

    def _stat(self, *, follow_symlinks):
        value = self.files.metadata(str(self.path), follow_symlinks=follow_symlinks)
        return SimpleNamespace(st_mode=value.mode, st_size=value.size,
                               st_mtime=value.mtime, st_ctime=value.ctime,
                               st_mtime_ns=value.mtime_ns, st_ctime_ns=value.ctime_ns,
                               st_ino=value.inode, st_dev=value.device)

    def walk(self):
        """Yield a mutable directory list like os.walk, without host access."""
        directories, filenames = [], []
        for child in self.iterdir():
            try:
                if child.is_dir():
                    directories.append(child.name)
                else:
                    filenames.append(child.name)
            except OSError:
                continue
        yield self, directories, filenames
        for name in directories:
            child = self / name
            if not child.is_symlink():
                yield from child.walk()

    def read_bytes(self):
        return self.files.read_bytes(str(self.path))

    def open(self, mode='r', **kwargs):
        if mode != 'rb' or kwargs:
            raise ValueError('task stream access currently requires binary read mode')
        path = self
        class Reader(io.RawIOBase):
            position = 0

            def readable(self):
                return True

            def readinto(self, buffer):
                data = path.files.read_range(str(path.path), self.position, len(buffer))
                buffer[:len(data)] = data
                self.position += len(data)
                return len(data)

        return io.BufferedReader(Reader())

    def read_text(self, encoding=None, errors=None):
        value = self.read_bytes().decode(encoding or 'utf-8', errors or 'strict')
        return value.replace('\r\n', '\n').replace('\r', '\n')

    def write_bytes(self, data):
        return self.files.write_bytes(str(self.path), data)

    def write_text(self, data, encoding=None, errors=None):
        self.write_bytes(data.encode(encoding or 'utf-8', errors or 'strict'))
        return len(data)

    def mkdir(self, mode=0o777, parents=False, exist_ok=False):
        if mode != 0o777:
            raise NotImplementedError('explicit task directory modes need native support')
        if exist_ok and self.is_dir():
            return
        self.files.mkdir(str(self.path), parents=parents)

    def unlink(self, missing_ok=False):
        if missing_ok and not self.exists():
            return
        self.files.unlink(str(self.path))

    def rmdir(self):
        self.files.rmdir(str(self.path))

    def chmod(self, mode):
        self.files.chmod(str(self.path), mode)

    def symlink_to(self, target):
        self.files.symlink_to(str(self.path), target)

    def iterdir(self):
        return iter(TaskPath(self.files, p) for p in self.files.iterdir(str(self.path)))

    def glob(self, pattern, *, files_only=False):
        """Expand within the selected namespace, without per-path launches."""
        from ._tools.glob import _NATIVE_GLOB
        parts = PurePosixPath(pattern).parts
        if not parts or PurePosixPath(pattern).is_absolute():
            raise ValueError('glob pattern must be nonempty and relative')
        output = self.files._call('glob', str(self.path), script_prefix=_NATIVE_GLOB,
                                 args=(str(int(files_only)), str(int(pattern.endswith('/'))), *parts))
        if output and not output.endswith(b'\x00'):
            raise TaskFileError('invalid task glob response')
        return iter(TaskPath(self.files, PurePosixPath(os.fsdecode(value)))
                    for value in output.split(b'\x00') if value)


_ACTIVE = ContextVar('task_file_access', default=None)


@dataclass(frozen=True)
class _TaskFileBinding:
    host_root: str
    host_alias: str
    files: NamespaceFiles

    def owns(self, cwd):
        # Only compare spellings captured at entry. Resolving host symlinks
        # again could discard the native executor after an alias is retargeted.
        return os.path.abspath(cwd) in (self.host_root, self.host_alias)


@contextmanager
def activate_task_files(files, *, host_root):
    root = getattr(files, '_host_task_root', None)
    if root is not None:
        root.verify()
    selected = (_TaskFileBinding(root.path if root is not None else str(Path(host_root).resolve()),
                                 os.path.abspath(host_root), files)
                if files is not None else None)
    token = _ACTIVE.set(selected)
    try:
        yield files
    finally:
        _ACTIVE.reset(token)


def retain_task_file_scope(cwd):
    """Capture this task's reader and aliases for later use without rediscovery."""
    selected = _ACTIVE.get()
    if selected is not None and not selected.owns(cwd):
        selected = None

    @contextmanager
    def scope():
        root = getattr(selected.files, '_host_task_root', None) if selected is not None else None
        if root is not None:
            root.verify()
        token = _ACTIVE.set(selected)
        try:
            yield selected.files if selected is not None else None
        finally:
            _ACTIVE.reset(token)

    return scope


def bound_task_path(cwd, path):
    active = _ACTIVE.get()
    if active is None or not active.owns(cwd):
        return None
    files = active.files
    value = PurePosixPath(path)
    if value.is_absolute():
        if value.is_relative_to(files.root):
            value = value.relative_to(files.root)
        elif value.is_relative_to(active.host_alias):
            value = value.relative_to(active.host_alias)
        elif value.is_relative_to(active.host_root):
            value = value.relative_to(active.host_root)
        else:
            raise ValueError('absolute path is outside the selected task view')
    return TaskPath(files, files.root / value).resolve()


def active_task_files(cwd):
    active = _ACTIVE.get()
    return active.files if active is not None and active.owns(cwd) else None


def active_task_host_root(cwd):
    """Return the host root observed at entry, without following later links."""
    active = _ACTIVE.get()
    if active is None or not active.owns(cwd):
        from .sandbox._filesystem import _ACTIVE as filesystem_active
        view = filesystem_active.get()
        if (view is not None and view.task_root is not None
                and os.path.abspath(cwd) in (view.cwd, view.host_alias)):
            view.task_root.verify()
            return view.cwd
        return None
    root = getattr(active.files, '_host_task_root', None)
    if root is not None:
        root.verify()
    return active.host_root


def _rebase_host_name(value, alias, root):
    if value == alias:
        return root
    prefix = alias.rstrip('/') + '/'
    if value.startswith(prefix):
        return root.rstrip('/') + '/' + value[len(prefix):]
    return value


def captured_host_path(cwd, path):
    """Rebase a declared task path before host-side mount discovery.

    This maps captured names only. It does not resolve links, expand globs,
    inspect files or authorize access outside the selected executor.
    """
    value = str(path)
    active = _ACTIVE.get()
    if active is not None and active.owns(cwd):
        return _rebase_host_name(value, active.host_alias, active.host_root)
    from .sandbox._filesystem import _ACTIVE as filesystem_active
    view = filesystem_active.get()
    if view is not None and os.path.abspath(cwd) in (view.cwd, view.host_alias):
        return _rebase_host_name(value, view.host_alias or view.cwd, view.cwd)
    return value


def capture_task_execution_paths(cwd, unreadable_paths=(), readable_paths=()):
    """Capture one task root and its mask/resource declarations.

    Retain path declarations rather than freezing their current glob matches.
    External resource declarations keep their own paths.
    """
    alias = os.path.abspath(cwd)
    root = active_task_host_root(cwd) or str(Path(cwd).resolve())
    captured = []
    for pattern in unreadable_paths:
        optional = 'optional:' if pattern.startswith('optional:') else ''
        value = pattern.removeprefix(optional) if optional else pattern
        captured.append(optional + _rebase_host_name(value, alias, root))
    resources = tuple(_rebase_host_name(str(path), alias, root) for path in readable_paths)
    return root, tuple(captured), resources


def _captured_native_alias(files, path):
    """Map recorded task names for this executor, including read-only views."""
    active = _ACTIVE.get()
    if active is None or active.files.run is not files.run or not path.is_absolute():
        return None
    if path.is_relative_to(active.files.root):
        return path
    for host in (active.host_alias, active.host_root):
        if path.is_relative_to(host):
            return active.files.root / path.relative_to(host)
    return None


def resolve_task_path(cwd, path):
    """Resolve a task reread without re-rooting outside absolute paths."""
    if isinstance(cwd, TaskPath):
        active = _ACTIVE.get()
        if PurePosixPath(path).is_absolute() and active is not None and active.files is cwd.files:
            target = bound_task_path(active.host_root, str(path))
        else:
            target = native_requested_path(cwd, str(path), expand=False).resolve()
        target.relative_to(cwd)
        host_root = active.host_root if active is not None and active.files is cwd.files else None
    else:
        target = bound_task_path(str(cwd), str(path))
        if target is None:
            root = Path(cwd).resolve()
            target = (root / path).resolve()
            target.relative_to(root)
        host_root = cwd
    from .sandbox.ignore_policy import active_ignore_policy, IgnoredPathError
    policy = active_ignore_policy(host_root) if host_root is not None else None
    if policy is not None and policy.is_model_hidden(target, is_dir=target.is_dir()):
        raise IgnoredPathError(str(path))
    return target


def startup_task_path(cwd):
    """Read ancestor guidance only where the selected namespace permits it."""
    if isinstance(cwd, TaskPath):
        return cwd.resolve()
    selected = bound_task_path(str(cwd), '.')
    if selected is None:
        return Path(cwd).resolve()
    readonly = selected.files.readonly_view('/')
    return TaskPath(readonly, selected.path).resolve()


def native_requested_path(base, value, *, variables=False, expand=True):
    """Resolve a task request without consulting the host's home or symlinks."""
    value = PurePosixPath(base.files.expand_path(str(value), variables=variables)
                         if expand else str(value))
    captured = _captured_native_alias(base.files, value)
    if captured is not None:
        return TaskPath(base.files, captured)
    host = base.files.binding.get('host_root')
    working = base.files.binding.get('working_directory')
    if (value.is_absolute() and host and working and value.is_relative_to(host)
            and not value.is_relative_to(working)):
        value = PurePosixPath(working) / value.relative_to(host)
    return TaskPath(base.files, value if value.is_absolute() else base.path / value)


def startup_source_path(source, task_dir):
    """Keep explicit operator resources local, but read task sources natively.

    Classify the requested spelling before host symlink resolution: a hidden
    host tree cannot decide where an in-task source points.
    """
    if isinstance(source, TaskPath):
        return source
    candidate = Path(source).expanduser().absolute()
    if not isinstance(task_dir, TaskPath):
        return candidate
    binding = task_dir.files.binding
    task_root = PurePosixPath(binding.get('working_directory', str(task_dir.path)))
    host_root = PurePosixPath(binding.get('host_root', str(task_root)))
    requested = PurePosixPath(str(candidate))
    captured = _captured_native_alias(task_dir.files, requested)
    if captured is not None:
        return TaskPath(task_dir.files, captured)
    if requested.is_relative_to(task_root):
        return TaskPath(task_dir.files, requested)
    if requested.is_relative_to(host_root):
        return TaskPath(task_dir.files, task_root / requested.relative_to(host_root))
    return candidate


class NativeUnreadableMatcher:
    """Compile explicit path masks using namespace enumeration and resolution."""

    def __init__(self, base, patterns):
        self.blocked = []
        for original in patterns:
            pattern = str(original).removeprefix('optional:')
            candidate = native_requested_path(base, pattern, variables=True)
            if any(character in str(candidate) for character in '*?['):
                root = TaskPath(base.files, base.files.root)
                if candidate.path.is_relative_to(root.path):
                    matches = root.glob(candidate.path.relative_to(root.path).as_posix())
                else:
                    literal = []
                    for part in candidate.path.parts:
                        if any(character in part for character in '*?['):
                            break
                        literal.append(part)
                    prefix = PurePosixPath(*literal)
                    if not root.path.is_relative_to(prefix):
                        continue
                    # An ancestor glob can overlap the task, but its expansion
                    # must enumerate only this permitted root, never the host.
                    matches = (path for path in (root, *root.glob('**/*'))
                               if _path_glob_matches(path.path, candidate.path))
            else:
                matches = (candidate,)
            for match in matches:
                try:
                    self.blocked.append(match.resolve())
                except (OSError, ValueError):
                    # A target outside this permitted namespace cannot be read.
                    continue

    def blocks(self, path):
        resolved = path.resolve()
        return any(resolved == root or resolved.is_relative_to(root) for root in self.blocked)


def _path_glob_matches(path, pattern):
    """Match whole path segments with the same recursive ** rule as glob."""
    parts, patterns = path.parts, pattern.parts

    @cache
    def match(i, j):
        if j == len(patterns):
            return i == len(parts)
        if patterns[j] == '**':
            return match(i, j + 1) or (i < len(parts) and match(i + 1, j))
        return (i < len(parts) and fnmatch.fnmatchcase(parts[i], patterns[j])
                and match(i + 1, j + 1))

    return match(0, 0)
