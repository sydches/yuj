"""Task file operations executed in the command namespace.

The transport owns namespace, identity and time allowance. Paths and utility
locations are discovered there; none of these operations opens a host task
path. Callers keep presentation, parsing and edit policy outside this layer.
"""
from __future__ import annotations

from dataclasses import dataclass
import errno
import os
import re
from pathlib import Path, PurePosixPath
import stat
from typing import Callable

from .task_file_scripts import _DISCOVER, _OPERATE, _DIGEST_BATCH, _RESOLVE_FILES


class TaskFileError(OSError):
    """The selected namespace could not complete a file operation."""


class TaskUtilityUnavailable(TaskFileError):
    """Discovery completed, but the named utility was absent."""


@dataclass(frozen=True)
class FileMetadata:
    mode: int
    size: int
    mtime: int
    ctime: int
    inode: int
    device: int
    mtime_ns: int
    ctime_ns: int

    @property
    def is_file(self):
        return stat.S_ISREG(self.mode)

    @property
    def is_dir(self):
        return stat.S_ISDIR(self.mode)


class NamespaceFiles:
    """Byte-preserving operations through an explicitly bound transport.

    ``run`` receives a constant shell program, literal argv and optional stdin
    bytes. It must execute them with the command namespace and user, enforce
    the current deadline, and return a binary CompletedProcess. It must never
    fall back to executing on the host after a namespace error.
    """

    def __init__(self, root: str, run: Callable, *, binding: dict, readonly=False,
                 shares_host_kernel=False):
        if not root.startswith('/') or '\x00' in root:
            raise ValueError('task root must be an absolute NUL-free path')
        self.root = PurePosixPath(root)
        self.run = run
        self.binding = dict(binding)
        self.readonly = readonly
        # Set by the execution transport, never inferred from a path spelling
        # or from descriptive binding records. Device/inode pairs can only be
        # compared across views known to share the host kernel.
        self.shares_host_kernel = shares_host_kernel
        self._utilities: dict[str, str] = {}

    def readonly_view(self, root):
        """Use the same executor for an explicitly admitted external root."""
        view = NamespaceFiles(str(root), self.run, binding={
            **self.binding, 'readonly_root': str(root),
        }, readonly=True, shares_host_kernel=self.shares_host_kernel)
        view.root = view.resolve('.')
        return view

    def observe_host_entry(self, path, host_path):
        """Compare directory entries without following their final symlinks."""
        if not self.shares_host_kernel:
            return {'relation': 'unverified', 'basis': 'unverified_kernel_relation'}
        try:
            host = Path(host_path).lstat()
            native = self.metadata(path, follow_symlinks=False)
        except OSError as error:
            return {'relation': 'unverified', 'basis': 'entry_metadata_unavailable',
                    'error_kind': type(error).__name__}
        left = {'device': host.st_dev, 'inode': host.st_ino, 'kind': stat.S_IFMT(host.st_mode)}
        right = {'device': native.device, 'inode': native.inode, 'kind': stat.S_IFMT(native.mode)}
        return {'relation': 'same_entry' if left == right else 'different_entry',
                'basis': 'shared_kernel_entry_metadata', 'host': left, 'native': right}

    def environment_value(self, name):
        if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', name):
            raise ValueError('invalid environment variable name')
        result = self.run('if [[ -v "$1" ]]; then printf "1\\0%s\\0" "${!1}"; '
                          'else printf "0\\0\\0"; fi', [name], None)
        fields = result.stdout.split(b'\x00')
        if result.returncode or len(fields) != 3 or fields[-1] or fields[0] not in (b'0', b'1'):
            raise TaskFileError('invalid task environment response')
        return os.fsdecode(fields[1]) if fields[0] == b'1' else None

    def expand_path(self, value, *, variables=False):
        """Expand requested home/environment references from native facts."""
        if value.startswith('~'):
            user, separator, rest = value.partition('/')
            if user == '~':
                home = self.environment_value('HOME')
            else:
                result = self.run('exec "$@"', [self._utility('getent'), 'passwd', user[1:]], None)
                entries = os.fsdecode(result.stdout).splitlines()
                fields = entries[0].split(':') if len(entries) == 1 else []
                home = fields[5] if result.returncode == 0 and len(fields) == 7 else None
            if not home or not home.startswith('/'):
                raise TaskFileError('requested task home directory is unavailable')
            value = home.rstrip('/') + (separator + rest if separator else '')
        if variables:
            def replace(match):
                name = match.group(1) or match.group(2)
                observed = self.environment_value(name)
                return match.group(0) if observed is None else observed
            value = re.sub(r'\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)', replace, value)
        return value

    def _utility(self, name):
        if name not in self._utilities:
            result = self.run(_DISCOVER, [name], None)
            if result.returncode:
                raise TaskFileError('task utility discovery failed')
            fields = result.stdout.split(b'\x00')
            if fields == [name.encode(), b'', b'']:
                raise TaskUtilityUnavailable(f'task environment has no usable {name} utility')
            if len(fields) != 3 or fields[0] != name.encode() or not fields[1].startswith(b'/'):
                raise TaskFileError(f'task environment has no usable {name} utility')
            self._utilities[name] = os.fsdecode(fields[1])
        return self._utilities[name]

    def _relative(self, path):
        value = os.fspath(path)
        if not isinstance(value, str) or '\x00' in value:
            raise ValueError('task path must be a NUL-free string')
        parsed = PurePosixPath(value)
        if parsed.is_absolute():
            try:
                return str(parsed.relative_to(self.root))
            except ValueError:
                raise ValueError(f'path escapes task root: {value}') from None
        return value

    def _call(self, operation, path, *, utility='', args=(), data=None, success=(0,), script_prefix=''):
        if self.readonly and operation in ('write', 'create', 'mkdir', 'chmod', 'unlink', 'replace', 'link', 'rmdir'):
            raise PermissionError(errno.EROFS, 'external task resource is read-only', str(path))
        relative = self._relative(path)
        realpath = self._utility('realpath')
        executable = self._utility(utility) if utility else ''
        if operation in ('read', 'read_observation', 'read_range', 'search', 'search_files', 'glob', 'write', 'create', 'replace',
                         'link', 'unlink', 'rmdir', 'stat', 'lstat', 'list', 'scandir', 'readlink', 'symlink', 'chmod', 'mkdir'):
            args = (self._utility('readlink'), *args)
        result = self.run(script_prefix + _OPERATE, [str(self.root), relative, realpath,
                                   operation, executable, *args], data)
        if result.returncode not in success:
            if result.returncode == 77:
                raise PermissionError(errno.EACCES, os.fsdecode(result.stderr).strip(), str(path))
            if result.returncode == 66:
                raise FileNotFoundError(errno.ENOENT, 'task file does not exist', str(path))
            if result.returncode == 73:
                raise IsADirectoryError(errno.EISDIR, 'task path is a directory', str(path))
            if result.returncode == 75 and operation == 'create':
                raise FileExistsError(errno.EEXIST, 'task output path already exists', str(path))
            # Preserve native diagnostics without interpreting their language
            # as file status. Missing/unreadable states remain operation errors.
            detail = os.fsdecode(result.stderr).strip()
            raise TaskFileError(f'{operation} failed: {detail or result.returncode}')
        return result.stdout

    def resolve(self, path):
        output = self._call('resolve', path)
        if not output.endswith(b'\x00') or b'\x00' in output[:-1]:
            raise TaskFileError('invalid task path response')
        return PurePosixPath(os.fsdecode(output[:-1]))

    def resolve_regular_files(self, paths):
        """Resolve regular-file rule keys together; exceptional entries raise."""
        names = tuple(dict.fromkeys(str(self.root / self._relative(path)) for path in paths))
        if not names:
            return {}
        output = self._call('resolve_files', self.root,
                            data=b''.join(os.fsencode(name) + b'\0' for name in names),
                            script_prefix=_RESOLVE_FILES)
        fields = output.split(b'\0')
        if fields[-1] or len(fields) != len(names) * 2 + 1:
            raise TaskFileError('invalid task file resolution response')
        result = {os.fsdecode(name): os.fsdecode(target)
                  for name, target in zip(fields[0:-1:2], fields[1:-1:2])}
        if set(result) != set(names):
            raise TaskFileError('task resolved paths differ from request')
        return result

    def read_bytes(self, path):
        return self._call('read', path, utility='cat')

    def read_observation(self, path):
        """Read bytes and stable descriptor metadata in one namespace call."""
        for _attempt in range(3):
            output = self._call('read_observation', path, utility='cat',
                                args=(self._utility('stat'),))
            try:
                fields = output.split(b'\0', 9)
                opened = PurePosixPath(os.fsdecode(fields[0]))
                opened.relative_to(self.root)
                before = self._parse_metadata(b'\0'.join(fields[1:9]) + b'\0')
                tail = fields[9].rsplit(b'\0', 9)
                data = tail[0]
                after = self._parse_metadata(b'\0'.join(tail[1:]))
            except (ValueError, IndexError):
                raise TaskFileError('invalid task read observation') from None
            if before == after and len(data) == after.size and after.is_file:
                return data, after
        raise TaskFileError('file changed while being read')

    def read_entry(self, path):
        """Read checkpoint bytes and mode together; preserve final symlinks."""
        output = self._call('read_entry', path, utility='cat',
                            args=(self._utility('readlink'), self._utility('stat')))
        header, separator, data = output.partition(b'\0')
        if not separator or not re.fullmatch(b'[0-9a-f]+', header):
            raise TaskFileError('invalid task entry mode')
        mode = int(header, 16)
        if stat.S_ISLNK(mode):
            if not data.endswith(b'\0') or b'\0' in data[:-1]:
                raise TaskFileError('invalid task entry symlink')
            return mode, data[:-1]
        if not stat.S_ISREG(mode):
            raise TaskFileError('unsupported task entry type')
        return mode, data

    def entry_modes(self, paths):
        """Inspect only requested entries, batching shared checked parents."""
        parents = {}
        for path in paths:
            name = self.root / self._relative(path)
            parents.setdefault(name.parent, {})[name.name] = str(name)
        result = {}
        for parent, entries in parents.items():
            names = list(entries)
            for start in range(0, len(names), 64):
                batch = names[start:start + 64]
                try:
                    output = self._call('entry_modes', parent, utility='stat',
                                        args=(self._utility('readlink'), *batch))
                except FileNotFoundError:
                    continue
                fields = output.split(b'\0')
                if fields[-1] or len(fields) % 2 != 1:
                    raise TaskFileError('invalid task entry modes')
                remaining = set(batch)
                for raw_name, raw_mode in zip(fields[0:-1:2], fields[1:-1:2]):
                    name = os.fsdecode(raw_name)
                    if name not in remaining or not re.fullmatch(b'[0-9a-f]+', raw_mode):
                        raise TaskFileError('invalid task entry mode response')
                    result[entries[name]] = int(raw_mode, 16)
                    remaining.remove(name)
        return result

    def sha256_many(self, paths):
        """Hash current bytes through checked descriptors, in bounded batches."""
        names = tuple(dict.fromkeys(str(self.root / self._relative(path)) for path in paths))
        if not names:
            return {}
        output = self._call('digest_batch', self.root, utility='sha256sum',
                            args=(self._utility('readlink'),),
                            data=b''.join(os.fsencode(name) + b'\0' for name in names),
                            script_prefix=_DIGEST_BATCH)
        fields = output.split(b'\0')
        if fields[-1] or len(fields) != len(names) * 2 + 1:
            raise TaskFileError('invalid task digest response')
        result = {}
        for name, digest in zip(fields[0:-1:2], fields[1:-1:2]):
            if not re.fullmatch(b'[0-9a-f]{64}', digest):
                raise TaskFileError('invalid task digest')
            result[os.fsdecode(name)] = digest.decode('ascii')
        if set(result) != set(names):
            raise TaskFileError('task digest paths differ from request')
        return result

    def read_range(self, path, offset, size):
        if offset < 0 or size < 0:
            raise ValueError('file ranges must be non-negative')
        return self._call('read_range', path, utility='dd', args=(str(offset), str(size)))

    def kind(self, path):
        try:
            value = self.metadata(path)
        except FileNotFoundError:
            return 'missing'
        return 'directory' if value.is_dir else 'file' if value.is_file else 'other'

    def is_symlink(self, path):
        return self._call('symlink', path) == b'true'

    def write_bytes(self, path, data: bytes):
        if not isinstance(data, bytes):
            raise TypeError('file content must be bytes')
        self._call('write', path, utility='dd', data=data)
        return len(data)

    def replace_bytes(self, path, data: bytes, *, mode):
        if not isinstance(data, bytes):
            raise TypeError('file content must be bytes')
        if type(mode) is not int or mode < 0 or mode > 0o7777:
            raise ValueError('checkpoint file mode must contain permission bits only')
        utilities = tuple(self._utility(name) for name in ('mktemp', 'mv', 'rm', 'chmod', 'rmdir'))
        self._call('replace', path, utility='dd', args=(*utilities, format(mode, 'o')), data=data)

    def create_bytes(self, path, data: bytes):
        """Publish complete bytes under a new name without replacing any entry."""
        if not isinstance(data, bytes):
            raise TypeError('file content must be bytes')
        utilities = tuple(self._utility(name) for name in ('mktemp', 'ln', 'rm'))
        self._call('create', path, utility='dd', args=utilities, data=data)
        return len(data)

    def metadata(self, path, *, follow_symlinks=True):
        value = self._parse_metadata(self._call(
            'stat' if follow_symlinks else 'lstat', path, utility='stat'))
        if follow_symlinks and stat.S_ISLNK(value.mode):
            raise TaskFileError('task symlink changed after resolution')
        return value

    @staticmethod
    def _parse_metadata(output):
        output = output.split(b'\x00')
        try:
            if len(output) != 9 or output[-1]:
                raise ValueError
            def nanoseconds(seconds, timestamp):
                fraction = timestamp.rsplit(b'.', 1)[1].split(b' ', 1)[0]
                if len(fraction) != 9 or not fraction.isdigit():
                    raise ValueError
                return int(seconds) * 1_000_000_000 + int(fraction)
            value = FileMetadata(int(output[0], 16), *(int(v) for v in output[1:6]),
                                 nanoseconds(output[2], output[6]),
                                 nanoseconds(output[3], output[7]))
            return value
        except (ValueError, IndexError):
            raise TaskFileError('unsupported task stat response') from None

    def mkdir(self, path, *, parents=False):
        self._call('mkdir', path, utility='mkdir', args=('-p',) if parents else ())

    def chmod(self, path, mode):
        if type(mode) is not int or mode < 0 or mode > 0o7777:
            raise ValueError('task mode must contain permission bits only')
        self._call('chmod', path, utility='chmod', args=(format(mode, 'o'),))

    def unlink(self, path):
        self._call('unlink', path, utility='rm')

    def rmdir(self, path):
        self._call('rmdir', path, utility='rmdir')

    def symlink_to(self, path, target):
        self._call('link', path, utility='ln', args=(str(target),))

    def readlink(self, path):
        output = self._call('readlink', path, utility='readlink')
        if not output.endswith(b'\x00') or b'\x00' in output[:-1]:
            raise TaskFileError('invalid task symlink response')
        return os.fsdecode(output[:-1])

    def iterdir(self, path='.'):
        output = self._call('list', path, utility='find')
        if output and not output.endswith(b'\x00'):
            raise TaskFileError('invalid task directory response')
        requested = self.root / self._relative(path)
        return [requested / PurePosixPath(os.fsdecode(value)).name
                for value in output.split(b'\x00') if value]

    def scandir(self, path='.'):
        """List immediate names and entry kinds in one checked directory."""
        output = self._call('scandir', path, utility='find')
        fields = output.split(b'\x00')
        if fields[-1] or len(fields) % 2 != 1:
            raise TaskFileError('invalid task directory metadata')
        rows = []
        for name, kind in zip(fields[0:-1:2], fields[1:-1:2]):
            if not name or b'/' in name or kind not in (b'f', b'd', b'l', b'b', b'c', b'p', b's', b'?'):
                raise TaskFileError('invalid task directory entry')
            rows.append((os.fsdecode(name), kind.decode('ascii')))
        return rows

    def search_files(self, path, glob_filter=''):
        """Let native ripgrep select files; subsequent reads remain checked."""
        args = ('--glob', glob_filter) if glob_filter else ()
        output = self._call('search_files', path, utility='rg', args=args, success=(0, 1))
        if output and not output.endswith(b'\x00'):
            raise TaskFileError('invalid task search listing')
        requested = self.root / self._relative(path)
        base = requested if self.kind(path) == 'directory' else requested.parent
        return [base / os.fsdecode(value) for value in output.split(b'\x00') if value]

    def search(self, path, pattern):
        try:
            self._utility('rg')
            utility = 'rg'
        except TaskUtilityUnavailable:
            utility = 'grep'
        output = self._call('search', path, utility=utility, args=(pattern,), success=(0, 1))
        # Both utilities search the checked descriptor through stdin. Attach
        # the caller's native path to numbered matches without reopening it.
        label = os.fsencode(str(self.root) + '/' + self._relative(path)) + b':'
        return re.sub(rb'(?m)^[0-9]+:', lambda match: label + match[0], output)
