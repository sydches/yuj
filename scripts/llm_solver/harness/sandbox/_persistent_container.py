"""A session-owned Docker connection with fresh, guarded command processes."""
import base64
import locale
import os
import secrets
import selectors
import shlex
import subprocess
import sys
import threading
import time

from ..operation_timing import measured
from ..process_identity import (
    FIELDS, GuardedProcessArgv, ProcessIdentityError, ProcessVerification,
    guarded_process_argv, with_container_deadline, with_container_process_group,
)
from ..task_environment import TaskEnvironmentUnavailable
from ..time_budget import command_timeout, execution_deadline, remaining_before
from ._binary_transport import _CAPTURE, captured_result
from .env_policy import build_clean_exec_argv


class UnsupportedPersistentContainer(RuntimeError):
    """Optional transport admission failed before any task command started."""


class _ConnectionClosed(RuntimeError):
    pass


_REQUEST_LOOP = r'''
for utility in base64 head wc mktemp rm rmdir setsid cat; do
    type -P "$utility" >/dev/null || exit 69
done
[[ $(printf YQ== | base64 --decode) == a ]] || exit 69
[[ $(printf abc | head -c 2) == ab ]] || exit 69
[[ $(printf abc | wc -c) -eq 3 ]] || exit 69
probe=$(mktemp -d) || exit 69
rm -f -- "$probe/probe" && rmdir -- "$probe" || exit 69
while IFS= read -r request; do
    command=$(printf '%s' "$request" | base64 --decode) || exit 74
    eval "$command"
done
'''

# Keep the established separate-stream response format. Exact-length input
# avoids large here-documents; pipe collectors retain descendant output/EOF.
_RAW_CAPTURE = _CAPTURE.replace(
    '"$encoder" --decode > "$capture_dir/in" || exit 74',
    r'''size=$1; shift
: > "$capture_dir/in"
if ((size)); then
    head -c "$size" > "$capture_dir/in" || exit 74
    observed=$(wc -c < "$capture_dir/in") || exit 74
    ((observed == size)) || exit 74
fi''',
)
_RAW_CAPTURE = _RAW_CAPTURE.replace(
    '"$@" < "$capture_dir/in" > "$capture_dir/out" 2> "$capture_dir/err"\nresult=$?',
    r'''exec {out_fd}> >(exec cat > "$capture_dir/out")
out_reader=$!
exec {err_fd}> >(exec {out_fd}>&-; exec cat > "$capture_dir/err")
err_reader=$!
cat "$capture_dir/in" {out_fd}>&- {err_fd}>&- | "$@" >&"$out_fd" 2>&"$err_fd" {out_fd}>&- {err_fd}>&-
result=$?
exec {out_fd}>&-
exec {err_fd}>&-
wait "$out_reader" || exit 74
wait "$err_reader" || exit 74''',
)
_ACK = b'yuj-container-start-v1\n'
_RAW_CAPTURE = ("IFS= read -r startup_ack || exit 74\n"
                "[[ $startup_ack == yuj-container-start-v1 ]] || exit 74\n" + _RAW_CAPTURE)


def _remaining(deadline, command, timeout):
    if deadline is None:
        return None
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise subprocess.TimeoutExpired(command, timeout)
    return remaining


class PersistentContainerSession:
    def __init__(self, cwd, environment, *, effective_env):
        if not environment.container_id or environment.process_identity is None:
            raise TaskEnvironmentUnavailable('persistent container requires an observed native binding')
        self.cwd = cwd
        self.environment = environment
        self._wrapper_env = {'PATH': dict(effective_env).get('PATH', os.defpath), 'LANG': 'C.UTF-8'}
        self._prefix = ['docker', 'exec', '-i', '--workdir', environment.working_directory,
                        environment.container_id]
        self._lock = threading.RLock()
        self._process = None
        self._group = None
        self._call_group = None
        self._verification = None
        self._verification_bytes = b''
        self._unsupported = False
        self._closed = False

    def supports(self, environment, cwd):
        return self.environment == environment and self.cwd == cwd and not self._closed

    def _exchange(self, request, marker, deadline, command, timeout, verifier, *, payload=None):
        process = self._process
        self._verification = verifier
        pending = memoryview(request)
        output = bytearray()
        marker_index = -1
        with selectors.DefaultSelector() as selector:
            os.set_blocking(process.stdout.fileno(), False)
            selector.register(process.stdout, selectors.EVENT_READ)
            if pending:
                os.set_blocking(process.stdin.fileno(), False)
                selector.register(process.stdin, selectors.EVENT_WRITE)
            while True:
                events = selector.select(_remaining(deadline, command, timeout))
                if not events:
                    raise subprocess.TimeoutExpired(command, timeout)
                for key, mask in events:
                    if mask & selectors.EVENT_WRITE:
                        count = os.write(key.fd, pending[:65536])
                        pending = pending[count:]
                        if not pending:
                            selector.unregister(process.stdin)
                    if mask & selectors.EVENT_READ:
                        block = os.read(key.fd, 65536)
                        if not block:
                            verifier.finish()
                            raise _ConnectionClosed('retained container ended before its response')
                        if not verifier.verified:
                            self._verification_bytes = block
                            block = verifier.feed(block)
                            self._verification_bytes = b''
                        if verifier.verified and payload is not None:
                            # The native handle is owned before task code or
                            # input can reach the freshly guarded child.
                            pending = memoryview(_ACK + payload)
                            payload = None
                            selector.register(process.stdin, selectors.EVENT_WRITE)
                        start = max(0, len(output) - len(marker) - 1)
                        output.extend(block)
                        if marker_index < 0:
                            marker_index = output.find(marker + b' ', start)
                        if marker_index >= 0 and output.endswith(b'\n'):
                            verifier.finish()
                            status = output[marker_index + len(marker) + 1:].strip()
                            if not status.isdigit() or len(status) > 3:
                                raise RuntimeError('invalid retained container status')
                            return bytes(output[:marker_index]), int(status)

    @measured('process.container_start')
    def _start(self, deadline, command, timeout):
        if self._process is not None:
            return
        _remaining(deadline, command, timeout)
        argv = with_container_process_group(guarded_process_argv(
            self._prefix,
            build_clean_exec_argv(['bash', '--noprofile', '--norc', '-p', '-c', _REQUEST_LOOP],
                                  self._wrapper_env),
            self.environment.process_identity,
        ), portable=True)
        self._group = argv.container_process_group
        self._process = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                         stderr=subprocess.STDOUT, bufsize=0, start_new_session=True)
        marker = ('___YUJ_CONTAINER_READY_' + secrets.token_hex(16)).encode()
        request = base64.b64encode(b"printf '%s %d\\n' '" + marker + b"' 0") + b'\n'
        verifier = ProcessVerification(argv)
        try:
            output, code = self._exchange(request, marker, deadline, command, timeout, verifier)
            if output or code:
                raise RuntimeError('invalid retained container startup response')
        except (_ConnectionClosed, ProcessIdentityError) as error:
            try:
                status = self._process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                raise error
            # The outer startup runs only harness admission code. Missing
            # setsid can fail before its credential frame; no task has run.
            if status in (69, 127):
                self._unsupported = True
                raise UnsupportedPersistentContainer('native framing capabilities unavailable') from error
            raise

    @measured('process.container_exchange')
    def execute(self, argv, *, timeout, binary=False, input_bytes=None):
        active_deadline = execution_deadline()
        timeout = remaining_before(active_deadline, command_timeout(timeout))
        deadline = None if timeout is None else time.monotonic() + timeout
        if active_deadline is not None:
            deadline = min(deadline, active_deadline) if deadline is not None else active_deadline
        allowance = _remaining(deadline, argv, timeout)
        acquired = self._lock.acquire() if allowance is None else self._lock.acquire(timeout=allowance)
        if not acquired:
            raise subprocess.TimeoutExpired(argv, timeout)
        try:
            if self._closed:
                raise RuntimeError('persistent container session is closed')
            if self._unsupported:
                raise UnsupportedPersistentContainer('native framing capabilities unavailable')
            if not isinstance(argv, GuardedProcessArgv) or argv[:2] != ['docker', 'exec']:
                raise TaskEnvironmentUnavailable('persistent container requires a guarded native command')
            if any('\0' in argument for argument in argv):
                raise ValueError('embedded null byte')
            marker_index = argv.index('yuj-process-guard')
            entry = marker_index - 6
            prefix = argv[:entry]
            if (prefix[-1] != self.environment.container_id or '--workdir' not in prefix
                    or prefix[prefix.index('--workdir') + 1] != self.environment.working_directory):
                raise TaskEnvironmentUnavailable('persistent container command binding changed')
            data = input_bytes if input_bytes is not None else b''
            if not isinstance(data, bytes):
                raise TypeError('persistent container input must be bytes')
            if self._process is None:
                self._start(deadline, argv, timeout)
            prepared = GuardedProcessArgv(argv)
            first_argument = marker_index + 1 + len(FIELDS)
            prepared[first_argument:] = [
                'bash', '--noprofile', '--norc', '-p', '-c', _RAW_CAPTURE,
                'yuj-container-capture', str(len(data)), *prepared[first_argument:],
            ]
            prepared = with_container_deadline(
                prepared, _remaining(deadline, argv, timeout), cancellable=True)
            self._call_group = prepared.container_process_group
            native = shlex.join(prepared[entry:])
            marker = ('___YUJ_CONTAINER_END_' + secrets.token_hex(16)).encode()
            header = os.fsencode('(cd -- ' + shlex.quote(self.environment.working_directory) + ' && ' + native + ')')
            header += (b"; status=$?; printf '%s %d\\n' '" + marker
                       + b"' \"$status\"; ((status == 0)) || exit \"$status\"")
            request = base64.b64encode(header) + b'\n'
            out, code = self._exchange(request, marker, deadline, argv, timeout,
                                       ProcessVerification(prepared), payload=data)
            if timeout is not None and code in (124, 137, 143):
                raise subprocess.TimeoutExpired(argv, timeout)
            if code:
                raise RuntimeError('retained container capture failed with status ' + str(code))
            result = captured_result(argv, out.decode('ascii'))
            if timeout is not None and result.returncode in (124, 137):
                raise subprocess.TimeoutExpired(argv, timeout)
            self._call_group = None
            self._verification = None
            if not binary:
                encoding = 'utf-8' if sys.flags.utf8_mode else locale.getencoding()
                result.stdout = result.stdout.decode(encoding).replace('\r\n', '\n').replace('\r', '\n')
                result.stderr = result.stderr.decode(encoding).replace('\r\n', '\n').replace('\r', '\n')
            return result
        except BaseException as error:
            try:
                self._stop(force=True)
            except BaseException as cleanup_error:
                raise ProcessIdentityError(
                    'Native transport termination was not confirmed after '
                    + type(error).__name__ + ': ' + str(cleanup_error)
                ) from error
            raise
        finally:
            self._lock.release()

    def _recover_group(self, process):
        verifier = self._verification
        if verifier is None or verifier.group is None or verifier.group.handle is not None:
            return
        until = time.monotonic() + 3
        if self._verification_bytes:
            try:
                verifier.feed(self._verification_bytes)
            except ProcessIdentityError:
                pass  # No ACK was sent; still require native pipe EOF below.
            self._verification_bytes = b''
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            while verifier.group.handle is None:
                left = until - time.monotonic()
                if left <= 0 or not selector.select(left):
                    raise ProcessIdentityError('native startup termination was not confirmed')
                block = os.read(process.stdout.fileno(), 256)
                if not block:
                    # No ACK was sent without a complete verification frame.
                    # Closed input and EOF mean the waiting startup has ended.
                    return
                try:
                    verifier.feed(block)
                except ProcessIdentityError:
                    # Killing the outer shell can add a signal diagnostic
                    # before any request starts. With no ACK, continue only
                    # to confirm EOF, never to admit or execute a command.
                    continue
        verifier.group.terminate(grace=0)

    def _stop(self, *, force=False):
        process = self._process
        if process is None:
            return
        error = None
        try:
            if process.stdin is not None:
                try:
                    process.stdin.close()
                except OSError as failure:
                    error = failure
            if force:
                for group in (self._call_group, self._group):
                    if group is not None and group.handle is not None:
                        try:
                            group.terminate(grace=0)
                        except BaseException as failure:
                            error = failure
                try:
                    self._recover_group(process)
                except BaseException as failure:
                    error = failure
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                if self._group is not None and self._group.handle is not None:
                    try:
                        self._group.terminate(grace=0)
                    except BaseException as failure:
                        error = failure
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=2)
            if process.stdout is not None:
                process.stdout.close()
            self._process = self._group = self._call_group = self._verification = None
            self._verification_bytes = b''
        if error is not None:
            raise error

    @measured('process.container_close')
    def close(self):
        with self._lock:
            self._closed = True
            self._stop()
