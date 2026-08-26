"""Bwrap preflight — verify the sandbox primitive actually works.

Runs the canonical Codex preflight (``bwrap --unshare-user --unshare-net
--ro-bind / / /bin/true``) once per process and caches the result.
Failure modes that we recognise: kernel ``unprivileged_userns_clone=0``,
AppArmor profile denying ns clone, seccomp filter from a parent
supervisor, missing bwrap binary. All produce stable stderr keywords
matched against ``_BWRAP_BROKEN_PATTERNS`` so the harness can fail
loudly at session start instead of falling back silently.

Extracted from the legacy ``sandbox.py`` module; see ``sandbox/__init__``
for the dispatch contract this preflight protects.
"""
from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)


# Bwrap preflight result memoised once per process. Tuple is
# (passed, failure_text_or_None). Recomputed only on cache miss.
_BWRAP_PREFLIGHT_CACHE: dict[str, tuple[bool, str | None]] = {}


# Known failure patterns from bwrap on a host whose user-namespace
# support has been disabled or broken. Ports the keyword set Codex uses
# in `system_bwrap_warning_for_path` (codex-rs/linux-sandbox/src/bwrap.rs).
# Substring match (case-sensitive — bwrap stderr is stable across versions).
_BWRAP_BROKEN_PATTERNS = (
    "loopback: Failed RTM_NEWADDR",
    "loopback: Failed RTM_NEWLINK",
    "setting up uid map: Permission denied",
    "No permissions to create a new namespace",
    "Operation not permitted",
)


def bwrap_preflight(bwrap_bin: str) -> tuple[bool, str | None]:
    """Verify bwrap can actually create a working sandbox on this host.

    Runs the canonical Codex preflight once per process:

        bwrap --unshare-user --unshare-net --ro-bind / / /bin/true

    Success means a fresh user+network namespace was set up and a
    trivial command ran inside it. The two flags exercise the same
    primitives _build_bwrap_argv relies on. Failures fall into known
    classes (kernel.unprivileged_userns_clone=0, AppArmor profile
    denying ns clone, seccomp filter from a parent supervisor, missing
    bwrap binary) which all produce stable stderr keywords.

    Returns (passed, failure_text_or_None). Cached at module level
    keyed by the bwrap_bin path: a typical run set restarts the harness
    per task, so the first call pays once and the rest hit the cache.

    The central sandbox policy turns a failed named-bwrap result into a
    startup error. Automatic selection may try another installed sandbox,
    but no path degrades to host execution.
    """
    cached = _BWRAP_PREFLIGHT_CACHE.get(bwrap_bin)
    if cached is not None:
        return cached
    if not Path(bwrap_bin).is_file():
        result = (False, f"bwrap binary not found at {bwrap_bin!r}")
        _BWRAP_PREFLIGHT_CACHE[bwrap_bin] = result
        return result
    import subprocess  # local import — preflight runs at session start, not per-call
    try:
        proc = subprocess.run(
            [bwrap_bin, "--unshare-user", "--unshare-net",
             "--ro-bind", "/", "/", "/bin/true"],
            capture_output=True, text=True, timeout=10,
        )
    except subprocess.TimeoutExpired:
        result = (False, "bwrap preflight timed out (10s)")
        _BWRAP_PREFLIGHT_CACHE[bwrap_bin] = result
        return result
    except Exception as e:
        result = (False, f"bwrap preflight raised: {e}")
        _BWRAP_PREFLIGHT_CACHE[bwrap_bin] = result
        return result
    if proc.returncode == 0:
        _BWRAP_PREFLIGHT_CACHE[bwrap_bin] = (True, None)
        return True, None
    err = (proc.stderr or "").strip()
    known = next((p for p in _BWRAP_BROKEN_PATTERNS if p in err), None)
    if known:
        msg = f"bwrap preflight failed (known pattern: {known!r}): {err[:400]}"
    else:
        msg = f"bwrap preflight returned exit={proc.returncode}: {err[:400]}"
    _BWRAP_PREFLIGHT_CACHE[bwrap_bin] = (False, msg)
    return False, msg
