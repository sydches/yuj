"""Trusted syscall worker; load this program before entering the task view.

This process executes no task utility and imports no task code. Its only
operations after entering the owned namespace add denial mounts or remove
mounts whose kernel identities match the controller's recorded masks.
"""
import ctypes
import json
import os
import stat
import sys


# Linux mount API flags from sys/mount.h; these are ABI constants, not budgets.
MS_BIND = 4096
MS_RDONLY = 1
MS_REMOUNT = 32
MS_NOSUID = 2
MS_NODEV = 4
MNT_DETACH = 2


def mount_identity(descriptor):
    with open(f'/proc/self/fdinfo/{descriptor}') as stream:
        for line in stream:
            if line.startswith('mnt_id:'):
                return int(line.split()[1])
    raise RuntimeError('kernel mount identity unavailable')


def main():
    request = json.load(sys.stdin)
    user, mount, pid, root = map(int, sys.argv[1:])
    libc = ctypes.CDLL(None, use_errno=True)
    libc.mount.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p,
                           ctypes.c_ulong, ctypes.c_void_p]
    libc.umount2.argtypes = [ctypes.c_char_p, ctypes.c_int]
    # PR_SET_DUMPABLE from linux/prctl.h. Same-UID task processes must not
    # inspect or alter the trusted worker while it has namespace capabilities.
    if libc.prctl(4, 0, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), 'cannot protect namespace supervisor')
    os.setns(user, 0)
    os.setns(mount, 0)
    os.setns(pid, 0)
    os.fchdir(root)
    os.chroot('.')
    os.chdir('/')
    child = os.fork()
    if child:
        _, status = os.waitpid(child, 0)
        return os.waitstatus_to_exitcode(status)

    def checked(result):
        if result != 0:
            number = ctypes.get_errno()
            raise OSError(number, os.strerror(number))

    try:
        # An opaque parent can hide an older child mount. Remove it first so
        # the child's recorded mount becomes reachable for identity checking.
        for path, expected in sorted(request['remove'].items(),
                                     key=lambda item: item[0].count('/')):
            descriptor = os.open(path, os.O_PATH | os.O_NOFOLLOW)
            try:
                if mount_identity(descriptor) != expected:
                    raise RuntimeError('refusing to unmount an unowned mount')
                checked(libc.umount2(os.fsencode(f'/proc/self/fd/{descriptor}'), MNT_DETACH))
            finally:
                os.close(descriptor)
        added = {}
        for path, kind in sorted(request['add'].items(),
                                 key=lambda item: item[0].count('/'), reverse=True):
            if not path.startswith('/') or '\x00' in path or kind not in ('file', 'directory'):
                raise ValueError('invalid mask operation')
            try:
                descriptor = os.open(path, os.O_PATH | os.O_NOFOLLOW)
            except FileNotFoundError:
                # A new host artifact need not exist in the shell's private
                # view. Record no mount, so a later update retries this path
                # if an entry becomes visible there.
                continue
            try:
                mode = os.fstat(descriptor).st_mode
                if stat.S_ISLNK(mode) or stat.S_ISDIR(mode) != (kind == 'directory'):
                    raise RuntimeError('mask target changed kind')
                target = os.fsencode(f'/proc/self/fd/{descriptor}')
                if kind == 'file':
                    checked(libc.mount(b'/dev/null', target, None, MS_BIND, None))
                else:
                    permissions = f'mode=755,uid={os.geteuid()},gid={os.getegid()}'.encode()
                    checked(libc.mount(b'tmpfs', target, b'tmpfs', MS_NOSUID | MS_NODEV, permissions))
            finally:
                os.close(descriptor)
            descriptor = os.open(path, os.O_PATH | os.O_NOFOLLOW)
            try:
                if kind == 'file':
                    checked(libc.mount(None, os.fsencode(f'/proc/self/fd/{descriptor}'), None,
                                       MS_BIND | MS_REMOUNT | MS_RDONLY | MS_NOSUID | MS_NODEV, None))
                added[path] = mount_identity(descriptor)
            finally:
                os.close(descriptor)
        print(json.dumps(added), flush=True)
        os._exit(0)
    except BaseException as error:
        print(f'{type(error).__name__}: {error}', file=sys.stderr, flush=True)
        os._exit(1)


if __name__ == '__main__':
    raise SystemExit(main())
