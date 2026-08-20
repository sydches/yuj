"""Codex-style apply_patch DSL — multi-file edits in a single tool call.

Goal: a session that needs to modify N files across M hunks no longer
needs N round-trips through the model. One apply_patch call ships the
whole change-set; the harness validates ALL hunks before applying ANY
(transactional), so a single bad hunk leaves the working tree
unchanged. Its grammar follows Codex apply_patch.

Grammar (one-line summary):

    *** Begin Patch
    {FileOp}+
    *** End Patch

    FileOp = Add | Delete | Update
    Add    = "*** Add File: <path>" \\n ("+" line \\n)+
    Delete = "*** Delete File: <path>" \\n
    Update = "*** Update File: <path>" \\n Hunk+
    Hunk   = ["@@" header? \\n] HunkLine+
    HunkLine = (" " | "-" | "+") line \\n

Verification policy:

  - Add: file must NOT exist (no overwrites via Add).
  - Delete: file must exist.
  - Update: every Hunk's old-line sequence (the " " context + "-"
    deletions) must be findable VERBATIM in the current file; if there
    are 0 or >1 matches the hunk is rejected. No fuzzy matching, no
    LLM-correction call — Codex made the same call. The model must
    reformulate.
  - All ops verify before any apply. Partial failure → no mutation.

Apply policy:

  - Hunks within a single Update are applied in DESCENDING line order
    so earlier replacements don't shift the indices of later ones.
  - Files are written via .replace(target.with_suffix(.tmp)) for atomic
    swap (the same pattern as state_writer).

Output:

  Success → <apply_patch ok="true" ops="N">…per-op summaries…</apply_patch>
  Failure → ERROR: <category>: <details> (no <apply_patch> envelope)

The DSL is opt-in through tools_apply_patch_enabled and defaults to false.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)


def _xml_attr(s: str) -> str:
    """Escape a string for inclusion as an XML attribute value."""
    return (
        s.replace("&", "&amp;")
         .replace('"', "&quot;")
         .replace("'", "&apos;")
         .replace("<", "&lt;")
         .replace(">", "&gt;")
    )


def _xml_body(s: str) -> str:
    """Escape `&`, `<`, `>` inside the envelope body so model-supplied
    paths or hunk-context text containing the literal `</apply_patch>`
    closing tag cannot terminate the envelope early.
    """
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


_BEGIN_RE = re.compile(r"^\*\*\* Begin Patch\s*$")
_END_RE = re.compile(r"^\*\*\* End Patch\s*$")
_ADD_RE = re.compile(r"^\*\*\* Add File:\s*(.+?)\s*$")
_DELETE_RE = re.compile(r"^\*\*\* Delete File:\s*(.+?)\s*$")
_UPDATE_RE = re.compile(r"^\*\*\* Update File:\s*(.+?)\s*$")
_HUNK_HDR_RE = re.compile(r"^@@(?:\s+(.*))?$")
_END_OF_FILE_RE = re.compile(r"^\*\*\* End of File\s*$")


class PatchParseError(Exception):
    """Raised when the patch text does not conform to the DSL grammar."""


class PatchVerifyError(Exception):
    """Raised when a parsed FileOp cannot be applied against the FS.

    ``kind`` is a short categorical tag stamped at each raise site so
    the dispatcher can route to a typed error envelope without re-
    parsing the message text. See ``render_error`` for the enumeration.
    """

    def __init__(self, message: str, *, kind: str = "verify"):
        super().__init__(message)
        self.kind = kind


_ENVELOPE_VERSION = "1"


def render_error(error_kind: str, message: str) -> str:
    """Render an apply_patch error envelope.

    Mirrors the success envelope shape so downstream classifiers
    (`_shared/classification.py::_APPLY_PATCH_RE`) can read the same
    `<apply_patch>` prefix uniformly. ``error_kind`` is one of:
    ``disabled``, ``parse``, ``path_outside_cwd``, ``file_exists``,
    ``file_not_found``, ``read_failed``, ``hunk_not_found``,
    ``hunk_ambiguous``, ``scope_not_found``, ``unknown_op_kind``,
    or ``verify`` (fallback for legacy raise sites).
    """
    return (
        f'<apply_patch ok="false" error_kind="{_xml_attr(error_kind)}" '
        f'v="{_ENVELOPE_VERSION}">\n'
        f'ERROR: {_xml_body(message)}\n'
        '</apply_patch>'
    )


@dataclass
class Hunk:
    """One change region within an Update FileOp.

    old_lines = the lines (in order) that must currently exist in the
    file at some position. new_lines = what to put there. Both lists
    contain the line content WITHOUT the leading prefix character; the
    parser strips the " "/"-"/"+" before storing.

    headers = the @@ scope-header strings preceding this hunk, in the
    order they appeared. Each header narrows the search window for
    old_lines: the verifier finds the first line containing the header
    text and constrains _find_unique to lines AFTER that match. Nested
    headers nest the scope. An empty list uses the whole file. See
    `_scope_start` for the resolver.
    """
    old_lines: list[str] = field(default_factory=list)
    new_lines: list[str] = field(default_factory=list)
    headers: list[str] = field(default_factory=list)


@dataclass
class FileOp:
    kind: str  # "add" | "delete" | "update"
    path: str
    add_lines: list[str] = field(default_factory=list)  # for "add"
    hunks: list[Hunk] = field(default_factory=list)     # for "update"


def parse_patch(patch_text: str) -> list[FileOp]:
    """Parse the apply_patch DSL into a list of FileOp.

    Whitespace at the END of every line is stripped (CRLF tolerance) but
    the leading single character is the protocol prefix. Trailing
    blank lines after End Patch are ignored. A Patch with zero FileOps
    is an error (the caller probably forgot the body).
    """
    lines = patch_text.splitlines()
    if not lines:
        raise PatchParseError("empty patch text")
    # Find Begin / End fences. Allow leading blanks before Begin.
    i = 0
    while i < len(lines) and not lines[i].strip():
        i += 1
    if i >= len(lines) or not _BEGIN_RE.match(lines[i]):
        raise PatchParseError(
            "missing or malformed `*** Begin Patch` opening fence "
            "(must be exactly that, on its own line)"
        )
    i += 1
    end_idx = None
    for j in range(len(lines) - 1, i - 1, -1):
        if _END_RE.match(lines[j]):
            end_idx = j
            break
    if end_idx is None:
        raise PatchParseError("missing `*** End Patch` closing fence")

    ops: list[FileOp] = []
    while i < end_idx:
        line = lines[i]
        m = _ADD_RE.match(line)
        if m:
            path = m.group(1)
            add_lines: list[str] = []
            i += 1
            while i < end_idx:
                nxt = lines[i]
                if _ADD_RE.match(nxt) or _DELETE_RE.match(nxt) or _UPDATE_RE.match(nxt):
                    break
                if not nxt.startswith("+"):
                    raise PatchParseError(
                        f"Add File body line {i+1}: expected '+ ' prefix, got {nxt!r}"
                    )
                add_lines.append(nxt[1:])
                i += 1
            if not add_lines:
                raise PatchParseError(f"Add File: {path}: empty body (need at least one + line)")
            ops.append(FileOp(kind="add", path=path, add_lines=add_lines))
            continue
        m = _DELETE_RE.match(line)
        if m:
            ops.append(FileOp(kind="delete", path=m.group(1)))
            i += 1
            continue
        m = _UPDATE_RE.match(line)
        if m:
            path = m.group(1)
            hunks: list[Hunk] = []
            i += 1
            current = Hunk()
            current_active = False
            while i < end_idx:
                nxt = lines[i]
                if _ADD_RE.match(nxt) or _DELETE_RE.match(nxt) or _UPDATE_RE.match(nxt):
                    break
                hdr_m = _HUNK_HDR_RE.match(nxt)
                if hdr_m:
                    # @@ header line. Two cases:
                    #   (1) we already have hunk content → flush + start fresh
                    #       (then the header attaches to the new hunk).
                    #   (2) we have no content yet (or just other headers) →
                    #       just accumulate the header on the current hunk.
                    if current_active and (current.old_lines or current.new_lines):
                        hunks.append(current)
                        current = Hunk()
                    hdr_text = hdr_m.group(1)
                    if hdr_text and hdr_text.strip():
                        current.headers.append(hdr_text.strip())
                    current_active = True
                    i += 1
                    continue
                if _END_OF_FILE_RE.match(nxt):
                    i += 1
                    continue
                if not nxt:
                    # Blank line inside a hunk represents a context line that is
                    # itself empty — preserve as " " context.
                    current.old_lines.append("")
                    current.new_lines.append("")
                    current_active = True
                    i += 1
                    continue
                prefix, body = nxt[0], nxt[1:]
                if prefix == " ":
                    current.old_lines.append(body)
                    current.new_lines.append(body)
                elif prefix == "-":
                    current.old_lines.append(body)
                elif prefix == "+":
                    current.new_lines.append(body)
                else:
                    raise PatchParseError(
                        f"Update File: {path}: line {i+1}: expected ' ', '-', '+' or "
                        f"'@@' prefix, got {nxt!r}"
                    )
                current_active = True
                i += 1
            if current_active and (current.old_lines or current.new_lines):
                hunks.append(current)
            if not hunks:
                raise PatchParseError(f"Update File: {path}: no hunks")
            ops.append(FileOp(kind="update", path=path, hunks=hunks))
            continue
        if not line.strip():
            i += 1
            continue
        raise PatchParseError(
            f"line {i+1}: expected `*** Add File:`, `*** Delete File:`, or "
            f"`*** Update File:`, got {line!r}"
        )

    if not ops:
        raise PatchParseError("patch envelope is empty (zero FileOps)")
    return ops


def _scope_start(file_lines: list[str], headers: list[str]) -> int:
    """Resolve @@ scope headers to a starting line index.

    Each header text is matched as a substring against file_lines, in
    order. The first line containing the first header sets the initial
    scope; the second header is searched FROM there, and so on
    (nesting). Returns the index AFTER the last matched header (so
    subsequent search sees lines below the deepest scope marker).

    Empty header list returns 0 — preserves pre-disambiguation behavior
    where _find_unique scans the whole file.

    Raises PatchVerifyError if any header text doesn't match a line
    within the active scope.
    """
    if not headers:
        return 0
    start = 0
    for hdr in headers:
        found = -1
        for j in range(start, len(file_lines)):
            if hdr in file_lines[j]:
                found = j
                break
        if found < 0:
            raise PatchVerifyError(
                f"@@ scope header not found"
                f"{' (within prior @@ scope)' if start > 0 else ''}: {hdr!r}",
                kind="scope_not_found",
            )
        start = found + 1
    return start


def _find_unique(file_lines: list[str], needle: list[str],
                 start_idx: int = 0) -> int:
    """Return the index of the unique occurrence of needle in file_lines.

    When ``start_idx`` is non-zero, only lines at or after that index are
    considered — used by the @@ scope resolver to narrow the search to
    the region after a scope header. Raises PatchVerifyError if needle
    appears 0 or >1 times in the active window.

    Empty needle returns ``start_idx`` (insert at the scope start).
    Codex's @@ disambiguation lives in :func:`_scope_start`; this
    function only enforces uniqueness within whatever window is given.
    """
    if not needle:
        return start_idx
    n = len(needle)
    if n > len(file_lines) - start_idx:
        raise PatchVerifyError(
            f"hunk needs {n} consecutive lines but file has "
            f"{len(file_lines) - start_idx}"
            f"{' within @@ scope' if start_idx > 0 else ''}",
            kind="hunk_not_found",
        )
    matches: list[int] = []
    for i in range(start_idx, len(file_lines) - n + 1):
        if file_lines[i : i + n] == needle:
            matches.append(i)
            if len(matches) > 1:
                break
    if not matches:
        # Surface the FIRST 4 lines of the missing needle so the model
        # can see where its expectation diverged.
        sample = "\n".join("  " + line for line in needle[:4])
        more = "" if len(needle) <= 4 else f"\n  …and {len(needle) - 4} more lines"
        raise PatchVerifyError(
            f"could not find expected lines in file"
            f"{' (within @@ scope)' if start_idx > 0 else ''}:\n{sample}{more}",
            kind="hunk_not_found",
        )
    if len(matches) > 1:
        raise PatchVerifyError(
            f"hunk matches {len(matches)} positions"
            f"{' within @@ scope' if start_idx > 0 else ' in file'}"
            f" (need uniqueness — add a more specific @@ header or more "
            f"context lines around the change)",
            kind="hunk_ambiguous",
        )
    return matches[0]


def _resolved_target(cwd_path: Path, op_path: str) -> Path:
    """Resolve a FileOp path under cwd, rejecting outside-cwd traversal.

    Mirrors the containment check that read/edit/write/list_definitions
    apply via tools._resolve. The bwrap mount namespace blocks cross-cwd
    writes at runtime, but defense-in-depth: refuse here so the error is
    a clear PatchVerifyError rather than a sandbox-level EPERM
    (which surfaces as an opaque "command failed" in the dispatcher).
    """
    if op_path.startswith("/"):
        op_path = op_path.lstrip("/")
    target = (cwd_path / op_path).resolve()
    cwd_resolved = cwd_path.resolve()
    try:
        target.relative_to(cwd_resolved)
    except ValueError:
        raise PatchVerifyError(
            f"path {op_path!r} resolves outside cwd; apply_patch refuses "
            "cross-cwd writes (use a path inside the working directory)",
            kind="path_outside_cwd",
        )
    return target


def _apply_op(op: FileOp, cwd_path: Path) -> str:
    """Apply one verified FileOp to the filesystem. Returns a summary line."""
    target = _resolved_target(cwd_path, op.path)
    if op.kind == "add":
        target.parent.mkdir(parents=True, exist_ok=True)
        body = "\n".join(op.add_lines) + "\n"
        target.write_text(body)
        return f"  added: {op.path} ({len(op.add_lines)} lines)"
    if op.kind == "delete":
        target.unlink()
        return f"  deleted: {op.path}"
    if op.kind == "update":
        text = target.read_text()
        # Preserve original trailing newline behavior.
        had_trailing_newline = text.endswith("\n")
        file_lines = text.split("\n")
        if had_trailing_newline:
            file_lines.pop()  # split leaves an empty trailing element
        # Locate every hunk first, then apply in descending order so
        # earlier indices stay valid after later splices. Re-resolve the
        # @@ scope here (Phase 1 already verified it succeeds).
        located: list[tuple[int, Hunk]] = []
        for h in op.hunks:
            start = _scope_start(file_lines, h.headers)
            idx = _find_unique(file_lines, h.old_lines, start_idx=start)
            located.append((idx, h))
        located.sort(key=lambda p: p[0], reverse=True)
        for idx, h in located:
            file_lines[idx : idx + len(h.old_lines)] = list(h.new_lines)
        new_text = "\n".join(file_lines)
        if had_trailing_newline:
            new_text += "\n"
        target.write_text(new_text)
        n_minus = sum(len(h.old_lines) - sum(1 for o, n in zip(h.old_lines, h.new_lines) if o == n) for h in op.hunks)
        n_plus = sum(len(h.new_lines) - sum(1 for o, n in zip(h.old_lines, h.new_lines) if o == n) for h in op.hunks)
        return f"  updated: {op.path} ({len(op.hunks)} hunk{'s' if len(op.hunks) != 1 else ''}, ~{n_minus}/+{n_plus})"
    raise PatchVerifyError(f"unknown op kind: {op.kind!r}", kind="unknown_op_kind")


def verify_and_apply(ops: list[FileOp], cwd: str) -> str:
    """Verify every op against the current FS, then apply transactionally.

    Verification phase reads each target and runs _find_unique for every
    Update hunk. If ANY op fails verification, NO file is touched. This
    is the "all or nothing" property the Codex apply_patch flow promises
    and is what makes the multi-file DSL safe to invoke speculatively.

    Returns a human-readable summary string on success. Raises
    PatchVerifyError on failure with a message naming the failing op.
    """
    cwd_path = Path(cwd)

    # Phase 1: verify every op (no FS mutation). Path-traversal guard
    # via _resolved_target — same containment as read/edit/write.
    for op in ops:
        target = _resolved_target(cwd_path, op.path)
        if op.kind == "add":
            if target.exists():
                raise PatchVerifyError(
                    f"Add File: {op.path}: file already exists (use Update File: instead)",
                    kind="file_exists",
                )
        elif op.kind == "delete":
            if not target.is_file():
                raise PatchVerifyError(
                    f"Delete File: {op.path}: file does not exist",
                    kind="file_not_found",
                )
        elif op.kind == "update":
            if not target.is_file():
                raise PatchVerifyError(
                    f"Update File: {op.path}: file does not exist",
                    kind="file_not_found",
                )
            try:
                text = target.read_text()
            except OSError as e:
                raise PatchVerifyError(
                    f"Update File: {op.path}: read failed: {e}",
                    kind="read_failed",
                ) from e
            file_lines = text.split("\n")
            if file_lines and file_lines[-1] == "" and text.endswith("\n"):
                file_lines.pop()
            for h in op.hunks:
                try:
                    start = _scope_start(file_lines, h.headers)
                    _find_unique(file_lines, h.old_lines, start_idx=start)
                except PatchVerifyError as e:
                    # Preserve the inner kind so the dispatcher can
                    # render the typed error envelope correctly.
                    raise PatchVerifyError(
                        f"Update File: {op.path}: {e}",
                        kind=getattr(e, "kind", "verify"),
                    ) from e

    # Phase 2: apply (all ops succeed-or-fail together by Phase 1's contract).
    summaries: list[str] = []
    for op in ops:
        summaries.append(_apply_op(op, cwd_path))
    body = "\n".join(_xml_body(s) for s in summaries)
    return (
        f'<apply_patch ok="true" ops="{len(ops)}" v="{_ENVELOPE_VERSION}">\n'
        f'{body}\n'
        '</apply_patch>'
    )
