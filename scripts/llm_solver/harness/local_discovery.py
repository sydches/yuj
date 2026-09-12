"""Local glob traversal using retained, checked directory descriptors."""
from fnmatch import fnmatchcase
import os
from pathlib import Path, PurePath
import stat
import sys

from .local_file_access import checked_local_parent, open_local_file


def canonical_entry(root: Path, path: Path) -> Path:
    """Reject outside parents even when a later link leads back into the task."""
    relative = path.relative_to(root)
    candidate = root
    for component in relative.parts:
        candidate = (candidate / component).resolve()
        candidate.relative_to(root)
    return candidate


def entry_mode(root: Path, path: Path) -> int:
    canonical = canonical_entry(root, path)
    with checked_local_parent(root, canonical) as (descriptor, name):
        return os.stat(name, dir_fd=descriptor, follow_symlinks=False).st_mode


def local_glob(root: Path, base: Path, pattern: str):
    """Match pathlib's default Unix glob contract without reopening walked names."""
    parts = PurePath(pattern).parts
    if any('**' in part and part != '**' for part in parts):
        raise ValueError("Invalid pattern: '**' can only be an entire path component")
    if not parts:
        raise ValueError('Unacceptable pattern: ' + repr(pattern))
    directory_only = pattern.endswith(os.sep)
    seen = set()

    def emit(path, entry):
        try:
            mode = entry.stat(follow_symlinks=False).st_mode
            if stat.S_ISLNK(mode):
                mode = entry_mode(root, path)
        except (OSError, ValueError):
            return
        if not stat.S_ISREG(mode) or directory_only or path in seen:
            return
        seen.add(path)
        yield path

    def visit(directory, index):
        if index == len(parts):
            return
        try:
            canonical = canonical_entry(root, directory)
            with open_local_file(root, canonical, flags=os.O_RDONLY | os.O_DIRECTORY) as descriptor:
                with os.scandir(descriptor) as entries:
                    # DirEntry metadata remains relative to this retained FD.
                    entries = list(entries)
                part = parts[index]
                if part == '**':
                    yield from visit(directory, index + 1)
                    for entry in entries:
                        # pathlib includes files for a trailing ** since 3.13.
                        if index + 1 == len(parts) and sys.version_info >= (3, 13):
                            yield from emit(directory / entry.name, entry)
                        if entry.is_dir(follow_symlinks=False):
                            yield from visit(directory / entry.name, index)
                elif part == '..':
                    yield from visit(directory / '..', index + 1)
                else:
                    for entry in entries:
                        if not fnmatchcase(entry.name, part):
                            continue
                        path = directory / entry.name
                        if index + 1 == len(parts):
                            yield from emit(path, entry)
                        else:
                            yield from visit(path, index + 1)
        except (OSError, ValueError):
            # pathlib glob skips inaccessible or concurrently removed entries.
            return

    yield from visit(base, 0)


def local_walk(root: Path):
    """Yield mutable directory names like os.walk, retaining each open parent."""
    def walk(directory):
        try:
            canonical = canonical_entry(root, directory)
            with open_local_file(root, canonical, flags=os.O_RDONLY | os.O_DIRECTORY) as descriptor:
                directories, files, links = [], [], set()
                with os.scandir(descriptor) as entries:
                    for entry in entries:
                        try:
                            mode = entry.stat(follow_symlinks=False).st_mode
                            if stat.S_ISLNK(mode):
                                links.add(entry.name)
                                mode = entry_mode(root, directory / entry.name)
                            (directories if stat.S_ISDIR(mode) else files).append(entry.name)
                        except (OSError, ValueError):
                            continue
                yield directory, directories, files
                for name in directories:
                    if name not in links:
                        yield from walk(directory / name)
        except (OSError, ValueError):
            return

    yield from walk(root)
