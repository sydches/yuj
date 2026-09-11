"""Observe Linux process credentials and guard the process that starts work."""
from dataclasses import dataclass
import re

FIELDS = ('Uid', 'Gid', 'Groups', 'CapInh', 'CapPrm', 'CapEff', 'CapBnd', 'CapAmb')
VERIFIED = b'\0yuj-process-identity-v1\0'


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


class ProcessVerification:
    """Consume exactly one startup frame, even when transport splits it."""

    def __init__(self, argv):
        self.remaining = VERIFIED if isinstance(argv, GuardedProcessArgv) else b''

    @property
    def verified(self):
        return not self.remaining

    def feed(self, data):
        count = min(len(data), len(self.remaining))
        if data[:count] != self.remaining[:count]:
            raise ProcessIdentityError('task command did not start with verified process credentials')
        self.remaining = self.remaining[count:]
        return data[count:]

    def finish(self):
        if self.remaining:
            raise ProcessIdentityError('task command did not start with verified process credentials')

    def read(self, stream):
        """Consume only the frame from a blocking stream; leave task bytes."""
        while self.remaining:
            data = stream.read(len(self.remaining))
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
