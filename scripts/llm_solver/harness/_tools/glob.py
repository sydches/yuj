"""glob tool: find files matching a glob pattern, optionally paginated."""
from ...config import Config
from ..sandbox.ignore_policy import active_ignore_policy
from ._common import _paginated_envelope, _resolve


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
        base = _resolve(cwd, path)
        policy = active_ignore_policy(cwd)
        if policy is not None and policy.is_model_hidden(
            base, is_dir=base.is_dir()
        ):
            return "No files found."
        matches = sorted(base.glob(pattern))
        rel = [
            str(m.relative_to(cwd))
            for m in matches
            if m.is_file()
            and (
                policy is None
                or not policy.is_ignored(m, is_dir=False)
            )
        ]
        if cfg is None or not cfg.search_pagination_enabled:
            if not rel:
                return "No files found."
            return "\n".join(rel)
        # tool_quirks gate: refuse panic globs (unscoped recursive + over-broad)
        # before rendering the paged envelope.
        from ...tool_quirks.transforms import apply_glob_caps
        refusal = apply_glob_caps(
            pattern=pattern, scope=path, total=len(rel), cfg=cfg, lines=rel,
        )
        if refusal is not None:
            return refusal
        return _paginated_envelope(
            tool="glob", pattern=pattern, scope=path,
            lines=rel, page=page,
            per_page=cfg.glob_max_matches_per_page,
            before_text="\n".join(rel),
        )
    except Exception as e:
        return f"ERROR: {e}"
