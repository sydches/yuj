"""Pinned handles for changing masks in an owned bwrap namespace."""
import json
import fcntl
import os
from pathlib import Path
import sys


def decode_mount_path(value):
    for escaped, literal in ((r'\040', ' '), (r'\011', '\t'),
                             (r'\012', '\n'), (r'\134', '\\')):
        value = value.replace(escaped, literal)
    return value


def mask_targets(arguments):
    """Decode only the mount forms emitted by the unreadable-path builder."""
    result = {}
    index = 0
    while index < len(arguments):
        if arguments[index] == '--ro-bind' and arguments[index + 1] == '/dev/null':
            result[arguments[index + 2]] = 'file'
            index += 3
        elif arguments[index] == '--tmpfs':
            result[arguments[index + 1]] = 'directory'
            index += 2
        else:
            raise ValueError('unsupported unreadable mount form')
    return result


class NamespaceMasks:
    """Use descriptors opened before model execution, never a later PID lookup."""

    def __init__(self, status, masks):
        self.fds = []
        self.mounts = {}
        try:
            process_dir = self._open(f'/proc/{int(status["child-pid"])}', os.O_PATH | os.O_DIRECTORY)
            self.mount = self._open('ns/mnt', os.O_RDONLY, dir_fd=process_dir)
            # NS_GET_USERNS (linux/nsfs.h) returns the namespace that owns
            # this mount namespace. Bwrap's solver can occupy a child user
            # namespace with fewer rights than its mount supervisor.
            self.user = fcntl.ioctl(self.mount, 0xb701)
            self.fds.append(self.user)
            self.pid = self._open('ns/pid', os.O_RDONLY, dir_fd=process_dir)
            self.root = self._open('root', os.O_PATH | os.O_DIRECTORY, dir_fd=process_dir)
            if os.fstat(self.mount).st_ino != status['mnt-namespace']:
                raise RuntimeError('bwrap mount namespace changed before binding')
            descriptor = os.open('mountinfo', os.O_RDONLY, dir_fd=process_dir)
            with os.fdopen(descriptor) as stream:
                for line in stream:
                    fields = line.split()
                    path = decode_mount_path(fields[4])
                    if path in masks:
                        self.mounts[path] = {'kind': masks[path], 'id': int(fields[0])}
            if self.mounts.keys() != masks.keys():
                raise RuntimeError('cannot bind all initial unreadable mounts')
        except BaseException:
            self.close()
            raise

    def _open(self, path, flags, **kwargs):
        descriptor = os.open(path, flags, **kwargs)
        self.fds.append(descriptor)
        return descriptor

    def update(self, desired, *, timeout):
        remove = {path: value['id'] for path, value in self.mounts.items()
                  if desired.get(path) != value['kind']}
        add = {path: kind for path, kind in desired.items()
               if path not in self.mounts or self.mounts[path]['kind'] != kind}
        if not remove and not add:
            return
        from .._tools._run_in_sandbox import _execute
        helper = str(Path(__file__).with_name('_namespace_mask_worker.py'))
        command = [sys.executable, '-I', helper, *(str(fd) for fd in
                   (self.user, self.mount, self.pid, self.root))]
        # _execute owns process-group timeout cleanup, including the child
        # needed to enter the already-existing PID namespace.
        result = _execute(command, timeout=timeout, binary=True,
                          input_bytes=json.dumps({'remove': remove, 'add': add}).encode(),
                          pass_fds=(self.user, self.mount, self.pid, self.root))
        if result.returncode:
            raise RuntimeError('namespace mask update failed: ' + os.fsdecode(result.stderr))
        updated = json.loads(result.stdout)
        for path in remove:
            self.mounts.pop(path)
        for path, identity in updated.items():
            self.mounts[path] = {'kind': add[path], 'id': identity}

    def close(self):
        for descriptor in self.fds:
            os.close(descriptor)
        self.fds.clear()
