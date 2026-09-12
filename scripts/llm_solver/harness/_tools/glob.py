"""glob tool: find files matching a glob pattern, optionally paginated."""
from ...config import Config
from .._tool_filters import _strip_cwd_absolute
from ..sandbox.ignore_policy import active_ignore_policy
from ._common import _paginated_envelope, _resolve


_NATIVE_GLOB = r'''
native_glob() {
    local readlink_bin=$1
    files_only=$2; directories_only=$3; shift 3
    parts=("$@")
    declare -A visiting=()
    local -a pending=() labels=()
    shopt -s nullglob dotglob
    emit_resolved() {
        local resolved=$2
        case "$resolved" in "$root"|"${root%/}/"*) ;; *) return 0 ;; esac
        [[ "$files_only" != 1 || -f "$resolved" ]] || return 0
        [[ "$directories_only" != 1 || -d "$resolved" ]] || return 0
        printf '%s\0' "$1"
    }
    flush_matches() {
        (( ${#pending[@]} )) || return 0
        local index resolved status
        local -a resolved_paths=()
        mapfile -d '' -t resolved_paths < <("$realpath_bin" -m -z -- "${pending[@]}")
        wait "$!"; status=$?
        if (( status == 0 && ${#resolved_paths[@]} == ${#pending[@]} )); then
            for index in "${!pending[@]}"; do
                emit_resolved "${labels[$index]}" "${resolved_paths[$index]}"
            done
        else
            # Preserve individual inaccessible-path handling on a partial batch.
            for index in "${!pending[@]}"; do
                canonical resolved "${pending[$index]}" || continue
                emit_resolved "${labels[$index]}" "$resolved"
            done
        fi
        pending=()
        labels=()
    }
    emit_match() {
        pending+=("$1")
        labels+=("$2")
        (( ${#pending[@]} < 64 )) || flush_matches
        return 0
    }
    expand() {
        local directory=$1 index=$2 display=$3 resolved key part child directory_fd
        [[ -d "$directory" ]] || return 0
        [[ -r "$directory" ]] || return 0
        exec {directory_fd}< "$directory" || return 74
        resolved=$("$readlink_bin" -- "/proc/self/fd/$directory_fd"; code=$?; printf '.'; exit "$code") || return 74
        resolved=${resolved%.}; resolved=${resolved%$'\n'}
        case "$resolved" in "$root"|"${root%/}/"*) ;; *) exec {directory_fd}<&-; return 0 ;; esac
        directory=/proc/self/fd/$directory_fd
        if (( index == ${#parts[@]} )); then
            emit_match "$directory" "$display"
            flush_matches
            exec {directory_fd}<&-
            return
        fi
        key="$resolved/$index"
        if [[ ${visiting[$key]:-} ]]; then exec {directory_fd}<&-; return 0; fi
        visiting[$key]=1
        part=${parts[index]}
        if [[ "$part" == .. ]]; then
            expand "$directory/.." "$((index + 1))" "$display/.." || return
        elif [[ "$part" == '**' ]]; then
            expand "$directory" "$((index + 1))" "$display" || return
            for child in "$directory"/*; do
                expand "$child" "$index" "$display/${child##*/}" || return
            done
        else
            # Keep fnmatch's literal backslash and bracket-caret behavior.
            part=${part//\\/\\\\}
            part=${part//\[\^/[\\^}
            for child in "$directory"/*; do
                [[ "${child##*/}" == $part ]] || continue
                if (( index + 1 == ${#parts[@]} )); then
                    emit_match "$child" "$display/${child##*/}"
                else
                    expand "$child" "$((index + 1))" "$display/${child##*/}" || return
                fi
            done
        fi
        visiting[$key]=''
        # Resolve queued entries while their parent descriptor is still open.
        flush_matches
        exec {directory_fd}<&-
        return 0
    }
    expand "${root%/}/$requested" 0 "${root%/}/$requested" || return
    flush_matches
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
        from ..local_discovery import local_glob
        native = isinstance(base, TaskPath)
        matches = (sorted(base.glob(pattern, files_only=True))
                   if native else sorted(local_glob(root, base, pattern)))
        # Both walkers retain checked directories and reject outside aliases.
        rel = [
            str(m.relative_to(root))
            for m in matches
            if (
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
