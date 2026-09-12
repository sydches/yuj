"""Observe Linux process credentials and guard the process that starts work."""
from dataclasses import dataclass
import re
import subprocess

FIELDS = ('Uid', 'Gid', 'Groups', 'CapInh', 'CapPrm', 'CapEff', 'CapBnd', 'CapAmb')
VERIFIED = b'\0yuj-process-identity-v1\0'
PROCESS_GROUP = b'\0yuj-task-process-v1\0'


class ProcessIdentityError(RuntimeError):
    pass


@dataclass(frozen=True)
class ProcessIdentity:
    # Linux status order: real, effective, saved, filesystem IDs.
    uids: tuple[int, ...]
    gids: tuple[int, ...]
    groups: tuple[int, ...]
    capabilities: tuple[str, ...]

    def __post_init__(self):
        if len(self.uids) != 4 or len(self.gids) != 4 or len(self.capabilities) != 5:
            raise ValueError('incomplete process identity')
        if any(type(value) is not int or value < 0 for value in (*self.uids, *self.gids, *self.groups)):
            raise ValueError('invalid numeric process identity')
        if any(not isinstance(value, str) or not re.fullmatch('[0-9a-f]+', value) for value in self.capabilities):
            raise ValueError('invalid process capabilities')

    def wire_values(self):
        return tuple(' '.join(map(str, values)) for values in (self.uids, self.gids, self.groups)) + self.capabilities


_READ_IDENTITY = r'''
IFS=$' \t\n'
declare -A identity=()
while IFS=$' \t:' read -r name value; do
    case "$name" in
        Uid|Gid|Groups|CapInh|CapPrm|CapEff|CapBnd|CapAmb)
            read -ra parts <<< "$value"
            identity[$name]="${parts[*]}"
            ;;
    esac
done < /proc/self/status || exit 77
for name in Uid Gid Groups CapInh CapPrm CapEff CapBnd CapAmb; do
    [[ -v identity[$name] ]] || exit 77
done
'''
OBSERVE_SCRIPT = _READ_IDENTITY + r'''
for name in Uid Gid Groups CapInh CapPrm CapEff CapBnd CapAmb; do
    printf '%s\0%s\0' "$name" "${identity[$name]}"
done
'''
_VERIFY_IDENTITY = _READ_IDENTITY + r'''
for name in Uid Gid Groups CapInh CapPrm CapEff CapBnd CapAmb; do
    if [[ "${identity[$name]}" != "$1" ]]; then
        printf 'process credential binding failed\n' >&2
        exit 77
    fi
    shift
done
printf '\0yuj-process-identity-v1\0'
'''


def parse_process_identity(payload):
    try:
        parts = payload.decode('ascii').split('\0')
        if parts[-1] or len(parts) != 2 * len(FIELDS) + 1 or tuple(parts[:-1:2]) != FIELDS:
            raise ValueError('invalid credential frame')
        values = parts[1:-1:2]
        numeric = tuple(tuple(int(value) for value in group.split()) for group in values[:3])
        return ProcessIdentity(*numeric, tuple(values[3:]))
    except (UnicodeError, ValueError, TypeError, AttributeError) as error:
        raise ProcessIdentityError('invalid native process credential response') from error


class GuardedProcessArgv(list):
    """An argv whose stdout starts with an entry-process verification frame."""


def with_container_deadline(argv, timeout):
    """Enforce an exec deadline where the command and its children run."""
    if not isinstance(argv, GuardedProcessArgv) or timeout is None or argv[:2] != ['docker', 'exec']:
        return argv
    # This position belongs to our constant entry shell, not task arguments.
    entry = argv.index('yuj-process-guard') - 6
    result = GuardedProcessArgv(argv)
    allowance = max(timeout * 0.9, timeout - 0.25)
    result[entry:entry] = ['timeout', '--signal=TERM', '--kill-after=0.1', f'{allowance:.6f}s']
    result.container_deadline = True
    return result


_SUPERVISE_GROUP = r'''
trap 'trap "" TERM; while :; do sleep 3600; done' TERM
IFS= read -r status < /proc/self/stat || exit 77
read -ra fields <<< "${status##*) }"
printf '\0yuj-task-process-v1\0%020d:%020d\0' "$$" "${fields[19]}"
script=$1
shift
bash --noprofile --norc -p -c "$script" yuj-supervised-task "$@" <&0 &
child=$!
wait "$child"
exit $?
'''

_CANCEL_GROUP = r'''
pid=$1 start=$2 grace=$3
if ! IFS= read -r status < "/proc/$pid/stat" 2>/dev/null; then exit 0; fi
read -ra fields <<< "${status##*) }"
[[ "${fields[19]}" == "$start" && "${fields[0]}" != Z ]] || exit 0
[[ "${fields[2]}" == "$pid" && "${fields[3]}" == "$pid" ]] || exit 77
kill -TERM -- "-$pid" || exit 74
sleep "$grace"
IFS= read -r status < "/proc/$pid/stat" || exit 74
read -ra fields <<< "${status##*) }"
[[ "${fields[19]}" == "$start" && "${fields[2]}" == "$pid" ]] || exit 77
kill -KILL -- "-$pid" || exit 74
# Zombies have stopped executing. Check live members in the task namespace.
for ((attempt=0; attempt<100; attempt++)); do
    live=0
    for entry in /proc/[0-9]*/stat; do
        IFS= read -r status < "$entry" 2>/dev/null || continue
        read -ra fields <<< "${status##*) }"
        [[ "${fields[2]}" != "$pid" || "${fields[0]}" == Z ]] || live=1
    done
    ((live)) || exit 0
    sleep 0.01
done
exit 74
'''


class ContainerProcessGroup:
    """A task-reported PID/start-time pair, never a host transport PID."""

    def __init__(self, cleanup_argv):
        self.cleanup_argv = cleanup_argv
        self.handle = None

    def terminate(self, grace=0.1):
        if self.handle is None:
            raise ProcessIdentityError('task process startup handle is not available')
        argv = GuardedProcessArgv([*self.cleanup_argv, *map(str, self.handle), str(grace)])
        result = subprocess.run(argv, capture_output=True, timeout=grace + 3)
        verified_process_result(argv, result)
        if result.returncode:
            raise ProcessIdentityError('task process group termination was not confirmed')


def with_container_process_group(argv):
    """Keep a cancellable group leader for legacy Docker background/cell work."""
    if not isinstance(argv, GuardedProcessArgv) or argv[:2] != ['docker', 'exec']:
        return argv
    marker = argv.index('yuj-process-guard')
    entry = marker - 6
    script = argv[marker - 1]
    if not script.startswith(_VERIFY_IDENTITY + '\n'):
        raise ProcessIdentityError('unknown guarded task entry script')
    result = GuardedProcessArgv(argv)
    cleanup = [*argv[:marker - 1], _VERIFY_IDENTITY + '\n' + _CANCEL_GROUP,
               *argv[marker:marker + 1 + len(FIELDS)]]
    result.container_process_group = ContainerProcessGroup(cleanup)
    result[marker - 1] = _VERIFY_IDENTITY + '\n' + _SUPERVISE_GROUP
    result.insert(marker + 1 + len(FIELDS), script[len(_VERIFY_IDENTITY) + 1:])
    result[entry:entry] = ['setsid', '--wait']
    return result


class ProcessVerification:
    """Consume exactly one startup frame, even when transport splits it."""

    def __init__(self, argv):
        self.remaining = VERIFIED if isinstance(argv, GuardedProcessArgv) else b''
        self.group = getattr(argv, 'container_process_group', None)
        self.handle_bytes = b''
        self.handle_pending = self.group is not None
        if self.handle_pending:
            self.remaining += PROCESS_GROUP

    @property
    def verified(self):
        return not self.remaining and not self.handle_pending

    def feed(self, data):
        count = min(len(data), len(self.remaining))
        if data[:count] != self.remaining[:count]:
            raise ProcessIdentityError('task command did not start with verified process credentials')
        self.remaining = self.remaining[count:]
        data = data[count:]
        if not self.remaining and self.handle_pending:
            count = min(len(data), 42 - len(self.handle_bytes))
            self.handle_bytes += data[:count]
            data = data[count:]
            if len(self.handle_bytes) == 42:
                if not re.fullmatch(rb'[0-9]{20}:[0-9]{20}\x00', self.handle_bytes):
                    raise ProcessIdentityError('invalid task process startup handle')
                handle = tuple(int(value) for value in self.handle_bytes[:-1].split(b':'))
                if handle[0] <= 1 or handle[1] <= 0:
                    raise ProcessIdentityError('invalid task process startup handle')
                self.group.handle = handle
                self.handle_pending = False
        return data

    def finish(self):
        if not self.verified:
            raise ProcessIdentityError('task command did not start with verified process credentials')

    def read(self, stream):
        """Consume only the frame from a blocking stream; leave task bytes."""
        while not self.verified:
            data = stream.read(len(self.remaining) or 42 - len(self.handle_bytes))
            if not data:
                self.finish()
            self.feed(data)


def guarded_process_argv(prefix, command, identity, *, combine_output=False):
    if identity is None:
        return [*prefix, *command]
    script = 'exec "$@" 2>&1' if combine_output else 'exec "$@"'
    return guarded_script_argv(prefix, script, identity, command)


def guarded_script_argv(prefix, script, identity, arguments=()):
    """Run a harness-owned script in the process whose credentials are checked."""
    if identity is None:
        raise ProcessIdentityError('native process credentials have not been observed')
    # -p suppresses BASH_ENV and imported shell functions in this entry guard.
    # The caller's command and declared environment follow the verification.
    return GuardedProcessArgv([*prefix, 'bash', '--noprofile', '--norc', '-p', '-c',
                               _VERIFY_IDENTITY + '\n' + script, 'yuj-process-guard',
                               *identity.wire_values(), *arguments])


def verified_process_result(argv, result):
    if not isinstance(argv, GuardedProcessArgv):
        return result
    prefix = VERIFIED if isinstance(result.stdout, bytes) else VERIFIED.decode('ascii')
    if not result.stdout.startswith(prefix):
        if result.returncode and result.stderr:
            detail = result.stderr.decode('utf-8', 'replace') if isinstance(result.stderr, bytes) else result.stderr
            raise ProcessIdentityError(f'task command did not start (exit {result.returncode}): {detail.strip()}')
        raise ProcessIdentityError('task command did not start with verified process credentials')
    result.stdout = result.stdout[len(prefix):]
    return result


def observe_container_process(container_id, working_directory):
    from ._tools._run_in_sandbox import _execute
    from .time_budget import execution_deadline, remaining_before
    result = _execute(['docker', 'exec', '--workdir', working_directory, container_id,
                       'bash', '--noprofile', '--norc', '-p', '-c', OBSERVE_SCRIPT],
                      timeout=remaining_before(execution_deadline()), binary=True)
    if result.returncode:
        raise ProcessIdentityError('cannot observe the selected container process credentials')
    return parse_process_identity(result.stdout)
