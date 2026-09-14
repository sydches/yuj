"""Recognize source grep output without executing or rewriting the command."""
from dataclasses import dataclass
import posixpath
import shlex

@dataclass(frozen=True)
class SourceGrep:
    directory: str
    single_path: str | None
    numbered: bool
    filenames: bool
    paths: tuple[str, ...] = ()


def source_grep(fragments, strip_assignments) -> SourceGrep | None:
    """Accept one source search, optional cd, and head/tail of its output."""
    directory = '.'
    search = None
    for fragment in fragments:
        try:
            argv = shlex.split(strip_assignments(fragment.text))
        except ValueError:
            return None
        if not argv:
            return None
        program = posixpath.basename(argv.pop(0))
        if search is not None:
            if not fragment.stdin_from_pipe or program not in {'head', 'tail'}:
                return None
            # Only a line slice of stdin; a filename would replace the source.
            if argv and not (len(argv) == 1 and argv[0].startswith('-') and argv[0][1:].isdigit()
                             or len(argv) == 2 and argv[0] in {'-n', '--lines'} and argv[1].lstrip('+-').isdigit()
                             or len(argv) == 1 and argv[0].startswith('--lines=') and argv[0][8:].lstrip('+-').isdigit()):
                return None
            continue
        if program == 'cd' and fragment.operator_after == '&&':
            if argv[:1] == ['--']:
                argv.pop(0)
            if len(argv) != 1 or any(c in argv[0] for c in '$`*?[]~'):
                return None
            directory = posixpath.normpath(posixpath.join(directory, argv[0]))
            continue
        if fragment.stdin_from_pipe:
            return None
        if program == 'git':
            if argv[:1] == ['-C'] and len(argv) >= 3:
                directory = posixpath.normpath(posixpath.join(directory, argv[1]))
                argv = argv[2:]
            if argv[:1] != ['grep']:
                return None
            argv = argv[1:]
            program = 'git grep'
        if program not in {'grep', 'egrep', 'fgrep', 'rg', 'git grep'}:
            return None
        numbered = False
        filenames = True
        recursive = program in {'rg', 'git grep'}
        pattern = False
        context = False
        paths = []
        options = True
        values = {'-e', '-f', '-A', '-B', '-C', '-m', '-g', '-t', '-T',
                  '--regexp', '--file', '--after-context', '--before-context', '--context',
                  '--max-count', '--glob', '--iglob', '--type', '--type-not',
                  '--include', '--exclude', '--exclude-dir', '--max-depth', '--threads', '-j',
                  '--color', '--colors', '--encoding', '--max-columns'}
        while argv:
            arg = argv.pop(0)
            if arg == '2>&1':
                continue
            if options and arg == '--':
                options = False
                continue
            if options and arg.startswith('--'):
                flag = arg.split('=', 1)[0]
                if flag in {'--files', '--files-with-matches', '--files-without-match', '--count',
                            '--count-matches', '--only-matching', '--json', '--null', '--null-data',
                            '--heading', '--replace', '--byte-offset', '--quiet', '--silent'}:
                    return None
                if flag in values and '=' not in arg:
                    if not argv:
                        return None
                    argv.pop(0)
                pattern |= flag in {'--regexp', '--file'}
                context |= flag in {'--after-context', '--before-context', '--context'}
                if flag in {'--line-number', '--no-line-number'}:
                    numbered = flag == '--line-number'
                if flag in {'--with-filename', '--no-filename'}:
                    filenames = flag == '--with-filename'
                recursive |= flag in {'--recursive', '--dereference-recursive'}
                continue
            if options and arg.startswith('-') and arg != '-':
                for index, flag in enumerate(arg[1:], 1):
                    if program == 'rg' and flag == 'r':
                        return None  # rg -r replaces matched text.
                    if flag in 'lcLoqZzbU':
                        return None
                    if flag in 'efABCmgtTj':
                        if index == len(arg) - 1:
                            if not argv:
                                return None
                            argv.pop(0)
                        pattern |= flag in 'ef'
                        context |= flag in 'ABC'
                        break
                    if flag in 'nN':
                        numbered = flag == 'n'
                    if flag in 'hH':
                        filenames = flag == 'H'
                    recursive |= flag in 'rR'
                continue
            if not pattern:
                pattern = True
            else:
                paths.append(arg)
        if not pattern or (context and not numbered) or '-' in paths:
            return None
        # grep without files reads stdin; rg and git grep search the worktree.
        if not paths and not recursive:
            return None
        single = paths[0] if len(paths) == 1 and (not recursive or program == 'rg') else None
        if single and any(c in single for c in '$`*?[]~'):
            single = None
        if not filenames and single is None:
            return None
        search = SourceGrep(directory, single, numbered, filenames, tuple(paths or ('.',)))
    return search
