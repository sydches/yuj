"""Unreadable-paths masking — pre-expand glob patterns into bwrap mount args.

Uses the same structural idea as Codex's ``unreadable_globs`` mechanism:
pre-expand the patterns at sandbox-build time and mount over each match
so the model can ``cat`` the path but gets empty content (file) or
empty directory (dir) — no forbidden-regex whack-a-mole over
``cat``/``cp``/``ls``/``mv``/``tee``/``awk``/``sed``. A shell command
rule cannot block every way to read a protected file. Masking at the
bwrap-argv layer makes the file-read tools' protections converge with
the bash tool's protections — the model cannot reach the answer-key
bytes via ANY path, regardless of which tool it uses or which
alternative shell verb it tries.

Extracted from the legacy ``sandbox.py`` module.
"""
from __future__ import annotations

import glob as _glob
import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)


# Hard ceiling on bwrap argv additions from unreadable_paths expansion.
# Bwrap accepts thousands of mount args without issue, but a runaway glob
# (e.g. `**` against `/`) would generate hundreds of thousands and exhaust
# argv. Cap is checked AFTER de-dup; expansion logs a WARNING and truncates
# silently if the cap fires.
_UNREADABLE_HARD_CAP = 50_000

# Cache for unreadable_paths expansion. Keyed by (frozenset(patterns),
# hard_cap). The pattern set does not change mid-process — the config is
# loaded once at startup — so a one-shot expansion amortises across every
# bash call for the rest of the process lifetime. Entry value is the
# (mask_args, n_files, n_dirs) triple ready to splice into argv.
_UNREADABLE_CACHE: dict[tuple, tuple[list[str], int, int]] = {}


def _is_specific_pattern(pat: str) -> bool:
    """A pattern is "specific" if it has no glob metachars anywhere.

    Used to distinguish a mistyped absolute path from a glob pattern
    that may validly match no files. A pattern containing `*`, `?`, or `[` ANYWHERE counts as
    a glob — even if the leaf segment is literal — because the glob
    machinery may legitimately produce zero matches at runtime
    depending on what the parent directories contain.
    """
    if not pat:
        return False
    return "*" not in pat and "?" not in pat and "[" not in pat


def _expand_unreadable_paths(
    patterns: tuple[str, ...],
    *,
    hard_cap: int = _UNREADABLE_HARD_CAP,
    sandbox_required: bool = False,
) -> tuple[list[str], int, int]:
    """Expand a tuple of glob patterns into bwrap mount args that mask each match.

    Returns ``(mask_args, n_files_masked, n_dirs_masked)``. The mask_args
    list interleaves bwrap flags suitable for splicing into argv:
      - file match  → `["--ro-bind", "/dev/null", path]` (read returns EOF)
      - dir match   → `["--tmpfs", path]` (replaces with empty tmpfs)
    Symlink matches resolve to their targets via `Path.resolve()` and
    are masked by their resolved kind.

    Cached at module level keyed by `(frozenset(patterns), hard_cap)`; the
    expansion runs once per process and is reused for every bash call.
    Empty patterns tuple short-circuits to `([], 0, 0)`.
    """
    if not patterns:
        return [], 0, 0
    cache_key = (frozenset(patterns), hard_cap)
    cached = _UNREADABLE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    seen: set[str] = set()
    files: list[str] = []
    dirs: list[str] = []
    zero_match_specific: list[str] = []
    for pat in patterns:
        # "optional:" prefix marks a mask whose ABSENCE is a legitimate
        # state (e.g. results/ exists only after a run has produced it):
        # mask when present, silently skip when absent. Distinct from a
        # typo'd path, which stays fail-closed under strict mode.
        optional = pat.startswith("optional:")
        if optional:
            pat = pat[len("optional:"):]
        # Expand $VARS and ~ so the default config can reference
        # $YUJ_REPO_ROOT and operators on
        # other hosts get working masks without rewriting paths.
        pat = os.path.expandvars(os.path.expanduser(pat))
        # recursive=True so '**' matches across directories. include_hidden
        # so we catch dotted leaf names (.repo_cache) and dotted parents.
        try:
            matches = _glob.glob(pat, recursive=True, include_hidden=True)
        except Exception as e:  # malformed glob, permission walking, etc.
            log.warning("unreadable_paths: glob(%r) raised %s; skipping", pat, e)
            continue
        if not matches and _is_specific_pattern(pat) and not optional:
            zero_match_specific.append(pat)
        for raw in matches:
            try:
                # Resolve symlinks so the bind binds the real inode, not
                # the symlink itself (otherwise the model could deref via
                # /proc/self/root or readlink).
                resolved = str(Path(raw).resolve())
            except OSError:
                continue
            if resolved in seen:
                continue
            seen.add(resolved)
            try:
                is_dir = Path(resolved).is_dir()
            except OSError:
                continue
            if is_dir:
                dirs.append(resolved)
            else:
                files.append(resolved)
            if len(seen) >= hard_cap:
                msg = (
                    f"unreadable_paths: hit hard cap {hard_cap} while expanding {pat!r} — "
                    "later patterns / matches truncated. Tighten patterns or "
                    "raise _UNREADABLE_HARD_CAP if this is intentional."
                )
                log.warning("%s", msg)
                # Under sandbox_required=true, silent truncation is a
                # contract violation — the operator asked for strict
                # isolation and got partial.
                if sandbox_required:
                    raise RuntimeError(msg)
                break
        if len(seen) >= hard_cap:
            break

    # Warn on zero-match specific patterns. Hard-fail under strict mode so a
    # typo'd config doesn't silently expand to no masks.
    if zero_match_specific:
        msg = (
            "unreadable_paths: specific pattern(s) matched zero entries: "
            + ", ".join(repr(p) for p in zero_match_specific)
            + " — likely a typo'd path"
        )
        log.warning("%s", msg)
        if sandbox_required:
            raise RuntimeError(msg)

    mask_args: list[str] = []
    for f in files:
        mask_args += ["--ro-bind", "/dev/null", f]
    for d in dirs:
        mask_args += ["--tmpfs", d]
    log.info(
        "unreadable_paths: %d patterns expanded → %d files masked (--ro-bind /dev/null), "
        "%d dirs masked (--tmpfs)",
        len(patterns), len(files), len(dirs),
    )
    _UNREADABLE_CACHE[cache_key] = (mask_args, len(files), len(dirs))
    return mask_args, len(files), len(dirs)
