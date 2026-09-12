"""glob tool: find files matching a glob pattern, optionally paginated."""
from ...config import Config
from .._tool_filters import _strip_cwd_absolute
from ..sandbox.ignore_policy import active_ignore_policy
from ._common import _paginated_envelope, _resolve


_NATIVE_GLOB = r'''
native_glob() {
    files_only=$1; directories_only=$2; shift 2
    parts=("$@")
    declare -A visiting=()
    shopt -s nullglob dotglob
    emit_match() {
        local resolved
        canonical resolved "$1" || return 0
        case "$resolved" in "$root"|"${root%/}/"*) ;; *) return 0 ;; esac
        [[ "$files_only" != 1 || -f "$resolved" ]] || return 0
        [[ "$directories_only" != 1 || -d "$resolved" ]] || return 0
        printf '%s\0' "$1"
    }
    expand() {
        local directory=$1 index=$2 resolved key part child
        [[ -d "$directory" ]] || return 0
        canonical resolved "$directory" || return 0
        case "$resolved" in "$root"|"${root%/}/"*) ;; *) return 0 ;; esac
        if (( index == ${#parts[@]} )); then
            emit_match "$directory"
            return
        fi
        key="$resolved/$index"
        [[ ! ${visiting[$key]:-} ]] || return 0
        visiting[$key]=1
        part=${parts[index]}
        if [[ "$part" == .. ]]; then
            expand "$directory/.." "$((index + 1))"
        elif [[ "$part" == '**' ]]; then
            expand "$directory" "$((index + 1))"
            for child in "$directory"/*; do
                expand "$child" "$index"
            done
        else
            # Keep fnmatch's literal backslash and bracket-caret behavior.
            part=${part//\\/\\\\}
            part=${part//\[\^/[\\^}
            for child in "$directory"/*; do
                [[ "${child##*/}" == $part ]] || continue
                if (( index + 1 == ${#parts[@]} )); then
                    emit_match "$child"
                else
                    expand "$child" "$((index + 1))"
                fi
            done
        fi
        visiting[$key]=''
        return 0
    }
    expand "${root%/}/$requested" 0
}
'''


def glob_files(pattern: str, path: str = ".", *, cwd: str,
               page: int = 1, cfg: Config | None = None) -> str:
    """Find files matching a glob pattern.

    When ``cfg.search_pagination_enabled`` is true, wraps the result
    in a ``<search_result/>`` envelope with total/shown/page/next_page
    attributes. When false or ``cfg`` is None, returns the raw line
    list (backwards compatible with pre-pagination callers).
    """
    if not isinstance(pattern, str) or pattern == "":
        return "ERROR: glob pattern must be a non-empty string"
    if pattern.startswith("/"):
        return (
            f"ERROR: glob pattern must be relative (got '{pattern}'); "
            "use the `path` argument for the search scope"
        )
    try:
        root = _resolve(cwd, ".")
        base = _resolve(cwd, path)
        policy = active_ignore_policy(cwd)
        if policy is not None and policy.is_model_hidden(
            base, is_dir=base.is_dir()
        ):
            return "No files found."
        # Stable ordering is part of the harness contract, not an ablated
        # cleanup transform. Filesystem enumeration order varies across
        # byte-identical worktree copies.
        from ..task_path import TaskPath
        native = isinstance(base, TaskPath)
        matches = (sorted(base.glob(pattern, files_only=True))
                   if native else sorted(base.glob(pattern)))
        # An outside directory can contain aliases back into the task.
        # Its entry names are still outside the permitted discovery scope.
        rel = [
            str(m.relative_to(root))
            for m in matches
            if (native or (m.is_file()
            and m.resolve().is_relative_to(root)
            and all(
                parent.resolve().is_relative_to(root)
                for parent in m.parents
                if parent.is_relative_to(root)
            )))
            and (
                policy is None
                or not policy.is_ignored(m, is_dir=False)
            )
        ]
        if cfg is None or not cfg.search_pagination_enabled:
            if not rel:
                return "No files found."
            return "\n".join(rel)
        # Tool-quirks guards keep broad searches bounded and add a hint, while
        # still returning the requested deterministic page.
        from ...tool_quirks.transforms import apply_glob_caps
        guarded_page = apply_glob_caps(
            pattern=pattern, scope=path, total=len(rel), cfg=cfg, lines=rel,
            page=page,
        )
        if guarded_page is not None:
            return guarded_page
        return _paginated_envelope(
            tool="glob", pattern=pattern, scope=path,
            lines=rel, page=page,
            per_page=cfg.glob_max_matches_per_page,
            before_text="\n".join(rel),
        )
    except Exception as e:
        return f"ERROR: {_strip_cwd_absolute(str(e), cwd)}"
