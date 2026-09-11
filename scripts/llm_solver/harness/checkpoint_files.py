"""Select native checkpoint entries without exposing the private Git store."""
import os
from pathlib import Path
import stat
import tempfile

from .time_budget import execution_deadline, remaining_before


def native_entry_matches(path, data, mode):
    """Avoid mutations when a mounted entry already has the captured state."""
    try:
        observed = path.lstat().st_mode
    except FileNotFoundError:
        return False
    if mode == 0o120000:
        return (stat.S_ISLNK(observed)
                and os.fsencode(path.files.readlink(str(path.path))) == data)
    return (stat.S_ISREG(observed) and bool(observed & 0o111) == bool(mode & 0o111)
            and path.read_bytes() == data)


def native_checkpoint_paths(root, *, shadow_dir, git, excluded):
    """Apply Git's ignore rules to names and rules read from the task view.

    The temporary worktree contains directories and permitted .gitignore
    bytes only. Git uses the existing private index to retain captured files
    that have since become ignored. It never opens the hidden host task tree.
    Symlinks remain entries; neither enumeration nor ignore loading follows
    them. The mirror is removed before ordinary file contents are captured.
    """
    with tempfile.TemporaryDirectory(prefix='.view-', dir=shadow_dir) as temporary:
        mirror = Path(temporary)
        paths = []
        pending = [root]
        while pending:
            remaining_before(execution_deadline())
            directory = pending.pop()
            relative_dir = directory.relative_to(root)
            (mirror / relative_dir).mkdir(parents=True, exist_ok=True)
            for child in directory.iterdir():
                remaining_before(execution_deadline())
                relative = child.relative_to(root).as_posix()
                if excluded(relative):
                    continue
                mode = child.lstat().st_mode
                if stat.S_ISDIR(mode):
                    pending.append(child)
                elif stat.S_ISREG(mode) or stat.S_ISLNK(mode):
                    paths.append(relative)
                    if child.name == '.gitignore' and stat.S_ISREG(mode):
                        (mirror / relative).write_bytes(child.read_bytes())
        paths.sort()
        if not paths:
            return []
        result = git(['check-ignore', '-z', '--stdin'],
                     input_bytes=b''.join(os.fsencode(path) + b'\0' for path in paths),
                     worktree=mirror, check=False)
        if result.returncode not in (0, 1):
            from .workspace_checkpoints import WorkspaceCheckpointError
            raise WorkspaceCheckpointError(
                'cannot apply native checkpoint ignore rules: '
                + result.stderr.decode(errors='replace').strip())
        ignored = {os.fsdecode(path) for path in result.stdout.split(b'\0') if path}
        return [path for path in paths if path not in ignored]
