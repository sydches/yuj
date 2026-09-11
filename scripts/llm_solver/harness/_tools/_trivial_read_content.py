"""Content reads used by the literal shell fast path."""
from ..sandbox.ignore_policy import IgnorePolicy
from ..inspection_evidence import InspectedText
from ._common import _resolve_read, _require_external_readable


def _read_cat(
    path: str, cwd: str, *, ignore_policy: IgnorePolicy | None = None,
    readonly_roots: tuple[str, ...] = (),
    unreadable_paths: tuple[str, ...] = (),
) -> tuple[str, int, bool] | None:
    """In-process equivalent of ``cat <path>``. Bytewise output match."""
    try:
        target = _resolve_read(cwd, path, readonly_roots=readonly_roots)
        _require_external_readable(
            cwd, target, unreadable_paths=unreadable_paths,
        )
        if ignore_policy is not None and ignore_policy.contains(target):
            ignore_policy.require_visible(target, is_dir=target.is_dir())
    except ValueError:
        return f"cat: {path}: No such file or directory\n", 1, False
    except FileNotFoundError:
        return f"cat: {path}: No such file or directory\n", 1, False
    try:
        data = target.read_bytes()
    except FileNotFoundError:
        return f"cat: {path}: No such file or directory\n", 1, False
    except IsADirectoryError:
        return f"cat: {path}: Is a directory\n", 1, False
    except PermissionError:
        return f"cat: {path}: Permission denied\n", 1, False
    except OSError:
        # Let the native command report an unclassified backend read failure.
        return None
    text = data.decode("utf-8", errors="replace")
    return InspectedText(text, path=target, data=data, body=text, start=1,
                         count=len(text.splitlines()), total=len(text.splitlines())), 0, False


def _read_head(
    path: str,
    cwd: str,
    *,
    n: int,
    ignore_policy: IgnorePolicy | None = None,
    readonly_roots: tuple[str, ...] = (),
    unreadable_paths: tuple[str, ...] = (),
) -> tuple[str, int, bool] | None:
    """In-process equivalent of ``head [-n N] <path>`` (default N=10)."""
    try:
        target = _resolve_read(cwd, path, readonly_roots=readonly_roots)
        _require_external_readable(
            cwd, target, unreadable_paths=unreadable_paths,
        )
        if ignore_policy is not None and ignore_policy.contains(target):
            ignore_policy.require_visible(target, is_dir=target.is_dir())
    except ValueError:
        return (
            f"head: cannot open '{path}' for reading: "
            "No such file or directory\n", 1, False,
        )
    except FileNotFoundError:
        return (
            f"head: cannot open '{path}' for reading: "
            "No such file or directory\n", 1, False,
        )
    try:
        data = target.read_bytes()
    except FileNotFoundError:
        return (
            f"head: cannot open '{path}' for reading: "
            "No such file or directory\n", 1, False,
        )
    except IsADirectoryError:
        return (
            f"head: error reading '{path}': Is a directory\n", 1, False,
        )
    except PermissionError:
        return (
            f"head: cannot open '{path}' for reading: Permission denied\n",
            1, False,
        )
    except OSError:
        return None
    text = data.decode("utf-8", errors="replace")
    # GNU head emits the first N newline-terminated lines. Split on
    # '\n' and rejoin first N parts; re-add a trailing newline iff
    # the file had at least N full lines (i.e. an N-th '\n' existed).
    parts = text.split("\n")
    head_text = "\n".join(parts[:n])
    if len(parts) > n:
        head_text += "\n"
    return InspectedText(head_text, path=target, data=data, body=head_text, start=1,
                         count=len(head_text.splitlines()), total=len(text.splitlines())), 0, False
