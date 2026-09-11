"""Carry separate byte streams through the persistent shell's text protocol."""
import base64
import binascii
import shlex
import subprocess

from .env_policy import build_bash_argv


_CAPTURE = r'''
encoder=$(type -P base64) || exit 69
temp_maker=$(type -P mktemp) || exit 69
remove_file=$(type -P rm) || exit 69
remove_dir=$(type -P rmdir) || exit 69
capture_dir=$("$temp_maker" -d) || exit 74
cleanup_capture() {
    "$remove_file" -f -- "$capture_dir/in" "$capture_dir/out" "$capture_dir/err"
    "$remove_dir" -- "$capture_dir"
}
trap cleanup_capture EXIT
"$encoder" --decode > "$capture_dir/in" || exit 74
"$@" < "$capture_dir/in" > "$capture_dir/out" 2> "$capture_dir/err"
result=$?
printf 'YUJ_BINARY_V1\n%d\n' "$result"
"$encoder" < "$capture_dir/out" || exit 74
printf ':\n'
"$encoder" < "$capture_dir/err" || exit 74
printf ':\n'
'''


def capture_command(command, data):
    """Keep data literal, and discover transport utilities in the task view."""
    encoded = base64.b64encode(data or b'').decode('ascii')
    invocation = shlex.join([
        *build_bash_argv(_CAPTURE), 'yuj-binary-capture',
        *build_bash_argv(command),
    ])
    # The retained shell receives this through stdin. A quoted here-document
    # keeps payload bytes out of exec argv and cannot interpret them as code.
    # The delimiter contains underscores, which cannot occur in base64 data.
    return invocation + " <<'YUJ_BINARY_INPUT'\n" + encoded + '\nYUJ_BINARY_INPUT\n'


def captured_result(command, output):
    """Reject broken framing rather than treating protocol text as file bytes."""
    try:
        header, code, streams = output.split('\n', 2)
        stdout, stderr, tail = streams.split(':\n')
        if header != 'YUJ_BINARY_V1' or tail or not code.isdigit():
            raise ValueError
        returncode = int(code)
        if not 0 <= returncode <= 255:
            raise ValueError
        out = base64.b64decode(''.join(stdout.splitlines()), validate=True)
        err = base64.b64decode(''.join(stderr.splitlines()), validate=True)
    except (ValueError, binascii.Error) as exc:
        raise RuntimeError('invalid persistent binary response') from exc
    return subprocess.CompletedProcess(command, returncode, out, err)
