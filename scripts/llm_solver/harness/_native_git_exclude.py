"""Update runtime-file exclusions only in the selected native repository."""
import re


def _exclude_native_runtime_file(path):
    """Update only the Git exclude file discovered in the task namespace."""
    from pathlib import PurePosixPath
    from .worktree_runtime import WorktreeRuntimeError
    if path.is_symlink() or not path.is_file():
        return
    files = path.files
    result = files.run('exec "$@"', [files._utility('git'), 'rev-parse',
                       '--show-toplevel', '--git-path', 'info/exclude'], None)
    locations = result.stdout.decode(errors='surrogateescape').splitlines()
    if result.returncode or len(locations) != 2:
        return
    repo_root = PurePosixPath(locations[0])
    relative = path.path.relative_to(repo_root).as_posix()
    if '\n' in relative or '\r' in relative:
        return
    literal = re.sub(r'([\\*?\[\] ])', r'\\\1', relative)
    exclude = files.root / locations[1]
    script = r'''
exclude=$1; rule=$2; mkdir_bin=$3
if [[ -f "$exclude" ]]; then
    while IFS= read -r line || [[ -n "$line" ]]; do
        [[ "$line" == "$rule" ]] && exit 0
    done < "$exclude"
fi
"$mkdir_bin" -p -- "${exclude%/*}" || exit
printf '\n%s\n' "$rule" >> "$exclude"
'''
    result = files.run(script, [str(exclude), '/' + literal, files._utility('mkdir')], None)
    if result.returncode:
        raise WorktreeRuntimeError('native Git exclude update unavailable')
