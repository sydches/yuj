"""Search checked local file descriptors with the existing regex engines."""
from contextlib import ExitStack
import errno
from fnmatch import fnmatchcase
import os
from pathlib import Path
import re
import stat
import subprocess
import tempfile
import time

from ..local_file_access import checked_local_parent
from ..local_discovery import canonical_entry, local_glob


def local_grep(root, scope, pattern, glob_filter, *, rg, policy, timeout):
    root = Path(root).resolve()
    deadline = time.monotonic() + timeout

    def execute(command, **kwargs):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired(command, timeout)
        return subprocess.run(command, cwd=root, capture_output=True,
                              timeout=remaining, **kwargs)

    directory = scope.is_dir()
    # Capture authorized entry names with retained directories. Engine ignore
    # discovery cannot admit an entry from a swapped outside directory.
    allowed = set(local_glob(root, scope, '**/*')) if directory else {scope}
    if rg:
        command = [rg, '--files', '--null', '--no-follow']
        if glob_filter:
            command += ['--glob', glob_filter]
        discovered = execute([*command, '--', str(scope)])
        if discovered.returncode > 1:
            return subprocess.CompletedProcess(command, discovered.returncode, '',
                                               os.fsdecode(discovered.stderr))
        selected = [Path(os.fsdecode(name)) for name in discovered.stdout.split(b'\0') if name]
    else:
        selected = sorted(path for path in allowed
                          if not glob_filter or fnmatchcase(path.name, glob_filter))
    selected = [path for path in selected if path in allowed
                and (policy is None or not policy.is_ignored(path, is_dir=False))]
    output = []
    for start in range(0, max(1, len(selected)), 64):
        with ExitStack() as stack:
            labels, parents = {}, {}

            def parent_for(path):
                if path.parent not in parents:
                    canonical = canonical_entry(root, path.parent) / path.name
                    parents[path.parent], _ = stack.enter_context(checked_local_parent(root, canonical))
                return parents[path.parent]

            for path in selected[start:start + 64]:
                flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | getattr(os, 'O_CLOEXEC', 0)
                try:
                    descriptor = os.open(path.name, flags, dir_fd=parent_for(path))
                except OSError as exc:
                    if exc.errno != errno.ELOOP:
                        raise
                    # Existing contained file aliases remain readable. A
                    # regular-file batch needs no repeated parent resolution.
                    canonical = canonical_entry(root, path)
                    if policy is not None and policy.is_ignored(canonical, is_dir=False):
                        continue
                    descriptor = os.open(canonical.name, flags, dir_fd=parent_for(canonical))
                stack.callback(os.close, descriptor)
                if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                    continue
                labels[str(descriptor).encode()] = os.fsencode(path)
            # /dev/fd and pass_fds are available on supported Unix systems,
            # including macOS; no Linux /proc or namespace wrapper is required.
            command = [rg or 'grep', '-n', '--with-filename' if directory else '--no-filename', '--color=never',
                       '--', pattern, *('/dev/fd/' + name.decode() for name in labels)]
            if rg:
                command.insert(1, '--no-heading')
            staging = None
            if rg and directory and labels:
                # Directory-mode rg skips binary inputs differently from
                # explicit files. Private descriptor aliases preserve that
                # behaviour and original relative names without copying bytes.
                staging = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix='yuj-grep-')))
                for descriptor, original in labels.items():
                    alias = staging / Path(os.fsdecode(original)).relative_to(root)
                    alias.parent.mkdir(parents=True, exist_ok=True)
                    alias.symlink_to('/dev/fd/' + descriptor.decode())
                command = [rg, '-n', '--no-heading', '--with-filename', '--color=never', '--follow',
                           '--no-ignore', '--hidden', '--', pattern, str(staging)]
            result = execute(command, input=b'', pass_fds=tuple(int(name) for name in labels))

            def relabel(data):
                if staging is not None:
                    data = data.replace(os.fsencode(staging) + b'/', os.fsencode(root) + b'/')
                return re.sub(rb'(?m)^(?P<prefix>(?:[Bb]inary file |rg: |grep: )?)/dev/fd/(?P<fd>\d+)(?=:| matches)',
                              lambda match: match['prefix'] + labels.get(match['fd'], match[0]), data)

            if result.returncode > 1:
                return subprocess.CompletedProcess(command, result.returncode, '',
                                                   os.fsdecode(relabel(result.stderr)))
            output.append(relabel(result.stdout))
    return subprocess.CompletedProcess([], 0, b''.join(output).decode('utf-8', 'replace'), '')
