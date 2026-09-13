"""Local task I/O through retained directories and non-following final opens."""
from contextlib import contextmanager
import errno
import os
from pathlib import Path
import stat as stat_module


_DIRECTORY_FLAGS = (getattr(os, 'O_SEARCH', getattr(os, 'O_PATH', os.O_RDONLY))
                    | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, 'O_CLOEXEC', 0))


@contextmanager
def checked_local_parent(cwd, target, *, create_parents=False):
    """Yield the retained parent and basename of an already resolved task path.

    Resolve permitted aliases before calling this helper. Any symlink introduced
    afterwards is refused, including in the task root's own ancestors.
    """
    root = Path(cwd).resolve()
    target = Path(target)
    try:
        relative = target.relative_to(root)
    except ValueError:
        raise PermissionError(errno.EACCES, 'path is outside the current task root', str(target)) from None
    if '..' in relative.parts:
        raise PermissionError(errno.EACCES, 'path escapes task root', str(target))
    parts = (*root.parts[1:], *relative.parts[:-1])
    descriptor = os.open(root.anchor, _DIRECTORY_FLAGS)
    try:
        for index, component in enumerate(parts):
            try:
                child = os.open(component, _DIRECTORY_FLAGS, dir_fd=descriptor)
            except FileNotFoundError:
                # Only create requested task descendants, never missing task
                # roots or host ancestors.
                if not create_parents or index < len(root.parts) - 1:
                    raise
                try:
                    os.mkdir(component, dir_fd=descriptor)
                except FileExistsError:
                    pass
                child = os.open(component, _DIRECTORY_FLAGS, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        yield descriptor, relative.name if relative.parts else '.'
    finally:
        os.close(descriptor)


@contextmanager
def open_local_file(cwd, target, flags=os.O_RDONLY, mode=0o666, *, create_parents=False):
    """Open through the checked parent; a late final symlink cannot be followed."""
    with checked_local_parent(cwd, target, create_parents=create_parents) as (parent, name):
        descriptor = os.open(name, flags | os.O_NOFOLLOW | getattr(os, 'O_CLOEXEC', 0),
                             mode, dir_fd=parent)
        try:
            yield descriptor
        finally:
            os.close(descriptor)


def _native(target):
    from .task_path import TaskPath
    return isinstance(target, TaskPath)


def read_bytes(cwd, target):
    if _native(target):
        return target.read_bytes()
    with open_local_file(cwd, target) as descriptor:
        with os.fdopen(descriptor, 'rb', closefd=False) as stream:
            return stream.read()


def read_observation(cwd, target):
    """Return bytes and metadata from the same stable, checked open file."""
    if _native(target):
        data, metadata = target.files.read_observation(str(target))
        return data, metadata.mtime_ns
    for _attempt in range(3):
        with open_local_file(cwd, target, os.O_RDONLY | os.O_NONBLOCK) as descriptor:
            before = os.fstat(descriptor)
            if stat_module.S_ISDIR(before.st_mode):
                raise IsADirectoryError(str(target))
            if not stat_module.S_ISREG(before.st_mode):
                raise OSError(f'unsupported task entry: {target}')
            with os.fdopen(descriptor, 'rb', closefd=False) as stream:
                data = stream.read()
            after = os.fstat(descriptor)
        def revision(info):
            return (info.st_dev, info.st_ino, info.st_mode, info.st_size,
                    info.st_mtime_ns, info.st_ctime_ns)
        if revision(before) == revision(after) and len(data) == after.st_size:
            return data, after.st_mtime_ns
    raise OSError(f'file changed while being read: {target}')


def read_text(cwd, target, *, encoding=None, errors=None):
    if _native(target):
        return target.read_text(encoding=encoding, errors=errors)
    with open_local_file(cwd, target) as descriptor:
        with os.fdopen(descriptor, 'r', encoding=encoding, errors=errors, closefd=False) as stream:
            return stream.read()


def write_bytes(cwd, target, data, *, create_parents=False):
    if _native(target):
        if create_parents:
            target.parent.mkdir(parents=True, exist_ok=True)
        return target.write_bytes(data)
    with open_local_file(cwd, target, os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
                         create_parents=create_parents) as descriptor:
        with os.fdopen(descriptor, 'wb', closefd=False) as stream:
            return stream.write(data)


def write_text(cwd, target, text, *, encoding=None, create_parents=False):
    if _native(target):
        if create_parents:
            target.parent.mkdir(parents=True, exist_ok=True)
        return target.write_text(text, encoding=encoding)
    with open_local_file(cwd, target, os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
                         create_parents=create_parents) as descriptor:
        with os.fdopen(descriptor, 'w', encoding=encoding, closefd=False) as stream:
            return stream.write(text)


def unlink(cwd, target):
    if _native(target):
        return target.unlink()
    with checked_local_parent(cwd, target) as (parent, name):
        return os.unlink(name, dir_fd=parent)


def stat(cwd, target):
    if _native(target):
        return target.stat()
    with checked_local_parent(cwd, target) as (parent, name):
        return os.stat(name, dir_fd=parent, follow_symlinks=False)


def exists(cwd, target):
    if _native(target):
        return target.exists()
    try:
        stat(cwd, target)
    except (FileNotFoundError, NotADirectoryError):
        return False
    return True


def is_file(cwd, target):
    if _native(target):
        return target.is_file()
    try:
        return stat_module.S_ISREG(stat(cwd, target).st_mode)
    except (FileNotFoundError, NotADirectoryError):
        return False


def is_dir(cwd, target):
    if _native(target):
        return target.is_dir()
    try:
        return stat_module.S_ISDIR(stat(cwd, target).st_mode)
    except (FileNotFoundError, NotADirectoryError):
        return False


def read_entry(cwd, target):
    """Read an entry's own mode and bytes, preserving a final symlink."""
    if _native(target):
        mode = target.lstat().st_mode
        if stat_module.S_ISDIR(mode):
            return mode, None
        if stat_module.S_ISLNK(mode):
            return mode, os.fsencode(target.files.readlink(str(target)))
        if stat_module.S_ISREG(mode):
            return mode, target.read_bytes()
        raise OSError(f'unsupported task entry: {target}')
    with checked_local_parent(cwd, target) as (parent, name):
        mode = os.stat(name, dir_fd=parent, follow_symlinks=False).st_mode
        if stat_module.S_ISDIR(mode):
            return mode, None
        if stat_module.S_ISLNK(mode):
            return mode, os.fsencode(os.readlink(name, dir_fd=parent))
        if not stat_module.S_ISREG(mode):
            raise OSError(f'unsupported task entry: {target}')
        descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
                             | getattr(os, 'O_CLOEXEC', 0), dir_fd=parent)
        try:
            mode = os.fstat(descriptor).st_mode
            if not stat_module.S_ISREG(mode):
                raise OSError(f'task entry changed type: {target}')
            with os.fdopen(descriptor, 'rb', closefd=False) as stream:
                return mode, stream.read()
        finally:
            os.close(descriptor)


def rmdir(cwd, target):
    if _native(target):
        return target.rmdir()
    with checked_local_parent(cwd, target) as (parent, name):
        return os.rmdir(name, dir_fd=parent)


def mkdir(cwd, target, mode=0o777, *, parents=False, exist_ok=False):
    if _native(target):
        return target.mkdir(mode=mode, parents=parents, exist_ok=exist_ok)
    with checked_local_parent(cwd, target, create_parents=parents) as (parent, name):
        try:
            return os.mkdir(name, mode=mode, dir_fd=parent)
        except FileExistsError:
            if not exist_ok or not stat_module.S_ISDIR(
                    os.stat(name, dir_fd=parent, follow_symlinks=False).st_mode):
                raise


def symlink(cwd, target, value):
    if _native(target):
        return target.symlink_to(value)
    with checked_local_parent(cwd, target) as (parent, name):
        return os.symlink(value, name, dir_fd=parent)


def chmod_created_directory(cwd, target, mode):
    """Set the final mode of a readable directory just created by recovery."""
    if _native(target):
        return target.chmod(mode)
    with open_local_file(cwd, target, os.O_RDONLY | os.O_DIRECTORY | os.O_NONBLOCK) as descriptor:
        return os.fchmod(descriptor, mode)
