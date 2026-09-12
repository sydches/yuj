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


class TaskFileError(OSError):
    """The selected namespace could not complete a file operation."""


class TaskUtilityUnavailable(TaskFileError):
    """Discovery completed, but the named utility was absent."""


_DISCOVER = r'''
for utility in "$@"; do
    location=$(type -P -- "$utility") || location=''
    printf '%s\0%s\0' "$utility" "$location"
done
'''

# Arguments and file bytes stay separate from the executable script. Append a
# sentinel when capturing realpath output so filenames ending in LF survive.
_OPERATE = r'''
root=$1; requested=$2; realpath_bin=$3; operation=$4; utility=$5
shift 5
canonical() {
    local result
    result=$("$realpath_bin" -m -- "$2"; code=$?; printf '.'; exit "$code") || return
    result=${result%.}
    result=${result%$'\n'}
    printf -v "$1" '%s' "$result"
}
change_mode() {
    # Some chmod implementations skip a symlink without reporting an error.
    # Do not mistake that skipped change for success or dereference the link.
    [[ ! -L "$3" ]] || { printf 'task symlink changed before chmod' >&2; return 74; }
    local error code mode_fd opened expected
    if ! error=$("$1" --no-dereference "$2" -- "$3" 2>&1); then
        # Older utilities lack this option. A readable entry can instead be
        # changed through a checked open descriptor, never a mutable symlink.
        if [[ $("$1" --help 2>&1) == *--no-dereference* ]]; then
            printf '%s' "$error" >&2; return 74
        fi
        [[ ! -L "$3" ]] || return 74
        canonical expected "$3" || return 74
        case "$expected" in "$root"|"${root%/}/"*) ;; *) return 77 ;; esac
        exec {mode_fd}< "$3" || return 74
        IFS= read -r -d '' opened < <("$readlink_bin" -z -- "/proc/self/fd/$mode_fd") || return 74
        [[ "$opened" == "$expected" ]] || return 74
        "$1" "$2" -- "/proc/self/fd/$mode_fd"
        code=$?
        exec {mode_fd}<&-
        (( code == 0 )) || return "$code"
    fi
    [[ ! -L "$3" ]] || { printf 'task symlink changed during chmod' >&2; return 74; }
}
enter_directory() {
    local opened_parent
    cd -P -- "$1" || return 74
    IFS= read -r -d '' opened_parent < <("$2" -z -- /proc/self/cwd) || return 74
    case "$opened_parent" in
        "$root"|"${root%/}/"*) ;;
        *) printf 'opened parent escapes task root' >&2; return 77 ;;
    esac
}
canonical root "$root" || exit 74
entry="$root/$requested"
case "$operation:${entry##*/}" in
    *:.|*:..|*:"") canonical target "$entry" || exit 74 ;;
    symlink:*|readlink:*|lstat:*|unlink:*|replace:*|link:*|create:*|rmdir:*)
        canonical parent "${entry%/*}" || exit 74
        target="$parent/${entry##*/}"
        ;;
    *) canonical target "$entry" || exit 74 ;;
esac
case "$target" in
    "$root"|"${root%/}/"*) ;;
    *) printf 'path escapes task root' >&2; exit 77 ;;
esac
# Establish absent paths with native predicates, not translated stderr. Search
# permission on the nearest existing ancestor is required to claim absence.
ancestor=$target
while [[ ! -e "$ancestor" && ! -L "$ancestor" && "$ancestor" != / ]]; do
    ancestor=${ancestor%/*}
    [[ -n "$ancestor" ]] || ancestor=/
done
if [[ -d "$ancestor" && ! -x "$ancestor" &&
      ( "$ancestor" != "$target" ||
        ( "$operation" != chmod && "$operation" != stat && "$operation" != lstat && "$operation" != mkdir ) ) ]]; then
    printf 'directory search denied' >&2; exit 77
fi
# These operations use one checked directory and relative names. Metadata
# and entry operations do not follow the final component at execution.
case "$operation" in
    stat|lstat|list|readlink) [[ -e "$target" || -L "$target" ]] || exit 66 ;;
    symlink) [[ -e "$target" || -L "$target" ]] || { printf false; exit 0; } ;;
esac
case "$operation" in
    write|create|replace|link|unlink|rmdir|stat|lstat|list|readlink|symlink|chmod|search_files)
        readlink_bin=$1; shift
        if [[ "$operation" == list || "$target" == "$root" ||
              ( "$operation" == search_files && -d "$target" ) ]]; then
            directory=$target
            target=.
        else
            directory=${target%/*}
            [[ -n "$directory" ]] || directory=/
            target="./${target##*/}"
        fi
        enter_directory "$directory" "$readlink_bin" || exit
        ;;
esac
case "$operation" in
    resolve) printf '%s\0' "$target" ;;
    glob) native_glob "$@" ;;
    search_files)
        exec "$utility" --files --null --no-follow "$@" -- "$target"
        ;;
    symlink)
        if [[ -L "$target" ]]; then printf true; else printf false; fi
        ;;
    read|read_range|search)
        [[ -e "$target" ]] || exit 66
        [[ ! -d "$target" ]] || exit 73
        readlink_bin=$1; shift
        # Resolve the opened file's kernel path, not the requested name again.
        # A parent link can change between realpath and open. No file bytes
        # enter the response until this descriptor passes containment.
        exec {input_fd}< "$target" || exit 74
        IFS= read -r -d '' opened_path < <("$readlink_bin" -z -- "/proc/self/fd/$input_fd") || exit 74
        case "$opened_path" in
            "$root"|"${root%/}/"*) ;;
            *) printf 'opened file escapes task root' >&2; exit 77 ;;
        esac
        [[ ! -d "/proc/self/fd/$input_fd" ]] || exit 73
        if [[ "$operation" == search ]]; then
            exec "$utility" -n --no-filename --color=never -- "$1" - <&"$input_fd"
        fi
        if [[ "$operation" == read_range ]]; then
            exec "$utility" iflag=skip_bytes,count_bytes skip="$1" count="$2" status=none <&"$input_fd"
        fi
        exec "$utility" -- <&"$input_fd"
        ;;
    write)
        # Canonical resolution permits existing contained aliases. At the
        # actual open, refuse a new final symlink instead of following it.
        # O_WRONLY preserves write-only files; nocreat/excl avoid accepting a
        # different creation state between this predicate and the open.
        conversion=excl
        [[ ! -e "$target" && ! -L "$target" ]] || conversion=nocreat
        exec "$utility" of="$target" oflag=nofollow conv="$conversion" status=none
        ;;
    create)
        mktemp_bin=$1; ln_bin=$2; rm_bin=$3
        [[ ! -e "$target" && ! -L "$target" ]] || exit 75
        temporary=$("$mktemp_bin" --tmpdir="${target%/*}" .yuj-output-XXXXXXXXXX) || exit
        trap '"$rm_bin" -f -- "$temporary"' EXIT
        "$utility" of="$temporary" oflag=nofollow conv=nocreat status=none || exit
        if "$ln_bin" -T -- "$temporary" "$target"; then
            exit 0
        fi
        [[ ! -e "$target" && ! -L "$target" ]] || exit 75
        exit 74
        ;;
    replace)
        mktemp_bin=$1; mv_bin=$2; rm_bin=$3; chmod_bin=$4; rmdir_bin=$5; mode=$6
        temporary=$("$mktemp_bin" --tmpdir="${target%/*}" .yuj-restore-XXXXXXXXXX) || exit
        trap '"$rm_bin" -f -- "$temporary"' EXIT
        # Flush the writer's open file before applying its final permissions.
        # A later path-based sync can be redirected or denied by that mode.
        "$utility" of="$temporary" oflag=nofollow conv=nocreat,fdatasync status=none || exit
        change_mode "$chmod_bin" "$mode" "$temporary" || exit
        if [[ -d "$target" && ! -L "$target" ]]; then
            "$rmdir_bin" -- "$target" || exit
        fi
        "$mv_bin" -T -- "$temporary" "$target"
        ;;
    link) exec "$utility" -s -T -- "$1" "$target" ;;
    stat|lstat)
        [[ -e "$target" || -L "$target" ]] || exit 66
        export LC_ALL=C
        exec "$utility" --printf='%f\0%s\0%Y\0%Z\0%i\0%d\0%y\0%z\0' -- "$target"
        ;;
    readlink) exec "$utility" -z -- "$target" ;;
    mkdir)
        readlink_bin=$1; shift
        if [[ "$target" == "$root" ]]; then
            enter_directory "$root" "$readlink_bin" || exit
            exec "$utility" "$@" -- .
        fi
        directory=${target%/*}
        [[ -n "$directory" ]] || directory=/
        if [[ "${1:-}" == -p ]]; then
            while [[ ! -e "$directory" && ! -L "$directory" && "$directory" != / ]]; do
                directory=${directory%/*}
                [[ -n "$directory" ]] || directory=/
            done
        fi
        remaining=${target#"$directory"}
        remaining=${remaining#/}
        enter_directory "$directory" "$readlink_bin" || exit
        if [[ "${1:-}" != -p ]]; then
            exec "$utility" -- "./$remaining"
        fi
        while [[ "$remaining" == */* ]]; do
            component=${remaining%%/*}
            # Native mkdir -p grants owner write/search on new intermediates.
            # Preserve every other umask bit, and leave the final directory's
            # umask unchanged. Existing directory modes are not modified.
            (umask u+wx; "$utility" -p -- "./$component") || exit
            enter_directory "./$component" "$readlink_bin" || exit
            remaining=${remaining#*/}
        done
        "$utility" -p -- "./$remaining" || exit
        [[ ! -L "./$remaining" ]] || { printf 'task symlink changed during mkdir' >&2; exit 74; }
        ;;
    chmod)
        change_mode "$utility" "$1" "$target"
        ;;
    rmdir) exec "$utility" -- "$target" ;;
    unlink) exec "$utility" -- "$target" ;;
    list) exec "$utility" "$target" -mindepth 1 -maxdepth 1 -print0 ;;
    *) printf 'unsupported task file operation' >&2; exit 64 ;;
esac
'''


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
        if operation in ('read', 'read_range', 'search', 'search_files', 'write', 'create', 'replace',
                         'link', 'unlink', 'rmdir', 'stat', 'lstat', 'list', 'readlink', 'symlink', 'chmod', 'mkdir'):
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

    def read_bytes(self, path):
        return self._call('read', path, utility='cat')

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
        output = self._call('stat' if follow_symlinks else 'lstat', path, utility='stat').split(b'\x00')
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
            if follow_symlinks and stat.S_ISLNK(value.mode):
                raise TaskFileError('task symlink changed after resolution')
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
