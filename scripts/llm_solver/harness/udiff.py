"""Parser and checked applier for standard unified diffs.

Every file and hunk is resolved against an in-memory snapshot before the
first filesystem mutation. Hunk line numbers are hints: an exact block may
move, and whitespace-only drift may be accepted when it identifies one unique
whole-line window. Unsafe or missing placements use the same ``<candidates>``
repair shape as the exact-string ``edit`` tool.
"""
from __future__ import annotations

import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path

from .edit_replacers import (
    Candidate,
    format_candidates_block,
    fuzzy_line_matches,
    rank_candidates,
)


_HUNK_RE = re.compile(
    r"^@@ -(?P<old_start>\d+)(?:,(?P<old_count>\d+))? "
    r"\+(?P<new_start>\d+)(?:,(?P<new_count>\d+))? @@(?:.*)$"
)
_NO_NEWLINE = r"\ No newline at end of file"
_DEV_NULL = "/dev/null"


class UnifiedDiffParseError(ValueError):
    """The supplied text is not a well-formed unified diff."""


class UnifiedDiffApplyError(ValueError):
    """A parsed diff cannot be applied safely to the current workspace."""

    def __init__(self, message: str, *, kind: str = "apply"):
        super().__init__(message)
        self.kind = kind


@dataclass
class DiffLine:
    kind: str
    text: str
    no_newline: bool = False


@dataclass
class UnifiedHunk:
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: list[DiffLine] = field(default_factory=list)

    @property
    def old_lines(self) -> list[str]:
        return [line.text for line in self.lines if line.kind in {" ", "-"}]

    @property
    def new_line_count(self) -> int:
        return sum(1 for line in self.lines if line.kind in {" ", "+"})


@dataclass
class UnifiedFilePatch:
    kind: str
    path: str
    old_path: str
    new_path: str
    hunks: list[UnifiedHunk] = field(default_factory=list)


@dataclass(frozen=True)
class AppliedOperation:
    kind: str
    path: str


@dataclass
class _PreparedFile:
    operation: AppliedOperation
    target: Path
    data: bytes | None
    fuzzy_mechanisms: tuple[str, ...]


def _header_path(line: str, prefix: str) -> str:
    if not line.startswith(prefix):
        raise UnifiedDiffParseError(f"expected {prefix.strip()!r} file header")
    raw = line[len(prefix):]
    if "\t" in raw:
        raw = raw.split("\t", 1)[0]
    raw = raw.strip()
    if not raw:
        raise UnifiedDiffParseError(f"empty path in {prefix.strip()!r} header")
    if raw.startswith('"'):
        try:
            values = shlex.split(raw)
        except ValueError as exc:
            raise UnifiedDiffParseError(f"invalid quoted path {raw!r}: {exc}") from exc
        if len(values) != 1:
            raise UnifiedDiffParseError(f"ambiguous quoted path header: {raw!r}")
        raw = values[0]
    return raw


def _normalize_header_pair(old_path: str, new_path: str) -> tuple[str, str]:
    old_git = old_path.startswith("a/")
    new_git = new_path.startswith("b/")
    if old_git and (new_git or new_path == _DEV_NULL):
        old_path = old_path[2:]
    if new_git and (old_git or old_path == _DEV_NULL):
        new_path = new_path[2:]
    return old_path, new_path


def _parse_hunk(lines: list[str], index: int) -> tuple[UnifiedHunk, int]:
    match = _HUNK_RE.match(lines[index])
    if match is None:
        raise UnifiedDiffParseError(
            f"line {index + 1}: malformed unified-diff hunk header"
        )
    old_count = int(match.group("old_count") or 1)
    new_count = int(match.group("new_count") or 1)
    hunk = UnifiedHunk(
        old_start=int(match.group("old_start")),
        old_count=old_count,
        new_start=int(match.group("new_start")),
        new_count=new_count,
    )
    index += 1
    seen_old = 0
    seen_new = 0
    while index < len(lines) and (seen_old < old_count or seen_new < new_count):
        line = lines[index]
        if line == _NO_NEWLINE:
            if not hunk.lines:
                raise UnifiedDiffParseError(
                    f"line {index + 1}: newline marker has no preceding hunk line"
                )
            hunk.lines[-1].no_newline = True
            index += 1
            continue
        if not line or line[0] not in {" ", "-", "+"}:
            raise UnifiedDiffParseError(
                f"line {index + 1}: expected a space, '-' or '+' hunk prefix"
            )
        kind = line[0]
        hunk.lines.append(DiffLine(kind, line[1:]))
        if kind in {" ", "-"}:
            seen_old += 1
        if kind in {" ", "+"}:
            seen_new += 1
        if seen_old > old_count or seen_new > new_count:
            raise UnifiedDiffParseError(
                f"line {index + 1}: hunk body exceeds its declared line counts"
            )
        index += 1
    if seen_old != old_count or seen_new != new_count:
        raise UnifiedDiffParseError(
            "hunk ended before its declared old/new line counts were satisfied"
        )
    if index < len(lines) and lines[index] == _NO_NEWLINE:
        if not hunk.lines:
            raise UnifiedDiffParseError("newline marker has no preceding hunk line")
        hunk.lines[-1].no_newline = True
        index += 1
    return hunk, index


def parse_unified_diff(patch_text: str) -> list[UnifiedFilePatch]:
    """Parse one or more standard ``---``/``+++`` file patches."""
    lines = patch_text.splitlines()
    if not lines:
        raise UnifiedDiffParseError("empty unified diff")
    patches: list[UnifiedFilePatch] = []
    index = 0
    while index < len(lines):
        while index < len(lines) and not lines[index].startswith("--- "):
            if lines[index].startswith("@@ "):
                raise UnifiedDiffParseError(
                    f"line {index + 1}: hunk appears before file headers"
                )
            index += 1
        if index >= len(lines):
            break
        old_path = _header_path(lines[index], "--- ")
        index += 1
        if index >= len(lines) or not lines[index].startswith("+++ "):
            raise UnifiedDiffParseError(
                f"line {index + 1}: expected '+++' header after '---'"
            )
        new_path = _header_path(lines[index], "+++ ")
        old_path, new_path = _normalize_header_pair(old_path, new_path)
        index += 1

        if old_path == _DEV_NULL and new_path == _DEV_NULL:
            raise UnifiedDiffParseError("both file paths cannot be /dev/null")
        if old_path == _DEV_NULL:
            kind, path = "add", new_path
        elif new_path == _DEV_NULL:
            kind, path = "delete", old_path
        else:
            if old_path != new_path:
                raise UnifiedDiffParseError(
                    f"rename diffs are unsupported: {old_path!r} -> {new_path!r}"
                )
            kind, path = "update", new_path

        hunks: list[UnifiedHunk] = []
        while index < len(lines) and lines[index].startswith("@@ "):
            hunk, index = _parse_hunk(lines, index)
            hunks.append(hunk)
        if not hunks:
            raise UnifiedDiffParseError(f"file patch {path!r} has no hunks")
        if index < len(lines):
            next_line = lines[index]
            if next_line and not (
                next_line.startswith("diff --git ")
                or next_line.startswith("--- ")
                or next_line.startswith("index ")
                or next_line.startswith("new file mode ")
                or next_line.startswith("deleted file mode ")
            ):
                raise UnifiedDiffParseError(
                    f"line {index + 1}: unexpected text after hunk: {next_line!r}"
                )
        patches.append(
            UnifiedFilePatch(kind, path, old_path, new_path, hunks)
        )
    if not patches:
        raise UnifiedDiffParseError("unified diff contains no file patches")
    return patches


def _resolved_target(cwd: Path, path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise UnifiedDiffApplyError(
            f"path {path!r} is outside the working directory",
            kind="path_outside_cwd",
        )
    target = (cwd / candidate).resolve()
    try:
        target.relative_to(cwd.resolve())
    except ValueError as exc:
        raise UnifiedDiffApplyError(
            f"path {path!r} is outside the working directory",
            kind="path_outside_cwd",
        ) from exc
    return target


def _verify_parent_directory(target: Path, display_path: str) -> None:
    """Reject an add whose nearest existing parent is not a directory."""
    parent = target.parent
    while not parent.exists() and parent != parent.parent:
        parent = parent.parent
    if not parent.is_dir():
        raise UnifiedDiffApplyError(
            f"cannot write {display_path}: parent {parent} is not a directory",
            kind="parent_not_directory",
        )


def _line_offsets(lines: list[str]) -> list[int]:
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line) + 1)
    return offsets


def _window_candidates(
    text: str,
    lines: list[str],
    starts: list[int],
    width: int,
    strategy: str,
    limit: int,
) -> list[Candidate]:
    offsets = _line_offsets(lines)
    candidates = []
    for start in starts[:max(1, limit)]:
        end_line = start + width
        end = offsets[end_line] - 1 if end_line > start else offsets[start]
        end = min(len(text), max(offsets[start], end))
        candidates.append(
            Candidate(
                strategy=strategy,
                start=offsets[start],
                end=end,
                similarity=1.0,
                line_number=start + 1,
            )
        )
    return candidates


def _raise_hunk_match_error(
    *,
    path: str,
    file_lines: list[str],
    old_lines: list[str],
    starts: list[int] | None,
    strategy: str,
    candidate_count: int,
    ambiguous: bool,
) -> None:
    text = "\n".join(file_lines)
    if starts:
        candidates = _window_candidates(
            text, file_lines, starts, len(old_lines), strategy, candidate_count
        )
    else:
        candidates = rank_candidates(
            text, "\n".join(old_lines), k=candidate_count
        )
    block = format_candidates_block(text, candidates, path)
    reason = "matches more than one location" if ambiguous else "was not found"
    message = f"hunk for {path} {reason}"
    if block:
        message += "\n" + block
    raise UnifiedDiffApplyError(
        message,
        kind="hunk_ambiguous" if ambiguous else "hunk_not_found",
    )


def _locate_hunk(
    *,
    path: str,
    file_lines: list[str],
    hunk: UnifiedHunk,
    expected: int,
    candidate_count: int,
) -> tuple[int, str]:
    old_lines = hunk.old_lines
    if not old_lines:
        return min(max(0, expected), len(file_lines)), "line_number"
    width = len(old_lines)
    if (
        0 <= expected <= len(file_lines) - width
        and file_lines[expected:expected + width] == old_lines
    ):
        return expected, "line_number"
    exact_starts = [
        start
        for start in range(len(file_lines) - width + 1)
        if file_lines[start:start + width] == old_lines
    ]
    if len(exact_starts) == 1:
        return exact_starts[0], "offset"
    if len(exact_starts) > 1:
        _raise_hunk_match_error(
            path=path,
            file_lines=file_lines,
            old_lines=old_lines,
            starts=exact_starts,
            strategy="exact",
            candidate_count=candidate_count,
            ambiguous=True,
        )

    fuzzy = fuzzy_line_matches(file_lines, old_lines)
    if fuzzy is not None:
        mechanism, fuzzy_starts = fuzzy
        if len(fuzzy_starts) == 1:
            return fuzzy_starts[0], mechanism
        _raise_hunk_match_error(
            path=path,
            file_lines=file_lines,
            old_lines=old_lines,
            starts=fuzzy_starts,
            strategy=mechanism,
            candidate_count=candidate_count,
            ambiguous=True,
        )
    _raise_hunk_match_error(
        path=path,
        file_lines=file_lines,
        old_lines=old_lines,
        starts=None,
        strategy="",
        candidate_count=candidate_count,
        ambiguous=False,
    )
    raise AssertionError("unreachable")


def _apply_hunks(
    patch: UnifiedFilePatch,
    text: str,
    *,
    candidate_count: int,
) -> tuple[str, tuple[str, ...]]:
    trailing_newline = text.endswith("\n")
    file_lines = text.split("\n") if text else []
    if trailing_newline and file_lines:
        file_lines.pop()
    offset = 0
    fuzzy_mechanisms: list[str] = []
    for hunk in patch.hunks:
        if hunk.old_count == 0:
            expected = max(0, hunk.old_start + offset)
        else:
            expected = max(0, hunk.old_start - 1 + offset)
        start, mechanism = _locate_hunk(
            path=patch.path,
            file_lines=file_lines,
            hunk=hunk,
            expected=expected,
            candidate_count=candidate_count,
        )
        if mechanism not in {"line_number", "offset"}:
            fuzzy_mechanisms.append(mechanism)
        old_end = start + hunk.old_count
        touches_eof = old_end == len(file_lines)
        replacement: list[str] = []
        cursor = start
        last_output_line: DiffLine | None = None
        for line in hunk.lines:
            if line.kind == " ":
                replacement.append(file_lines[cursor])
                cursor += 1
                last_output_line = line
            elif line.kind == "-":
                cursor += 1
            else:
                replacement.append(line.text)
                last_output_line = line
        if cursor != old_end:
            raise UnifiedDiffApplyError(
                f"internal hunk count mismatch for {patch.path}",
                kind="hunk_count",
            )
        file_lines[start:old_end] = replacement
        offset += hunk.new_line_count - hunk.old_count
        if touches_eof:
            if replacement:
                trailing_newline = not (
                    last_output_line is not None
                    and last_output_line.no_newline
                )
            else:
                # A prefix line that survives a suffix-only deletion already
                # had a line terminator before the deleted range.  Preserve
                # it; only a now-empty file has no trailing newline.
                trailing_newline = bool(file_lines)
    rendered = "\n".join(file_lines)
    if trailing_newline:
        rendered += "\n"
    return rendered, tuple(fuzzy_mechanisms)


def prepare_unified_diff(
    patches: list[UnifiedFilePatch],
    cwd: str,
    *,
    candidate_count: int = 3,
) -> list[_PreparedFile]:
    """Verify and stage every file result without mutating the filesystem."""
    cwd_path = Path(cwd).resolve()
    prepared: list[_PreparedFile] = []
    seen: set[Path] = set()
    for patch in patches:
        target = _resolved_target(cwd_path, patch.path)
        if target in seen:
            raise UnifiedDiffApplyError(
                f"duplicate file patch for {patch.path}", kind="duplicate_path"
            )
        seen.add(target)
        _verify_parent_directory(target, patch.path)
        if patch.kind == "add":
            if target.exists():
                raise UnifiedDiffApplyError(
                    f"cannot add {patch.path}: file already exists",
                    kind="file_exists",
                )
            source = ""
        else:
            if not target.is_file():
                raise UnifiedDiffApplyError(
                    f"cannot {patch.kind} {patch.path}: file does not exist",
                    kind="file_not_found",
                )
            try:
                raw = target.read_bytes()
                source = raw.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise UnifiedDiffApplyError(
                    f"cannot patch {patch.path}: file is not valid UTF-8",
                    kind="binary_file",
                ) from exc
            except OSError as exc:
                raise UnifiedDiffApplyError(
                    f"cannot read {patch.path}: {exc}", kind="read_failed"
                ) from exc
        crlf = "\r\n" in source
        normalized = source.replace("\r\n", "\n") if crlf else source
        rendered, fuzzy = _apply_hunks(
            patch,
            normalized,
            candidate_count=max(1, int(candidate_count)),
        )
        if patch.kind == "delete" and rendered:
            raise UnifiedDiffApplyError(
                f"delete patch for {patch.path} does not remove the whole file",
                kind="delete_not_empty",
            )
        if crlf:
            rendered = rendered.replace("\n", "\r\n")
        operation = AppliedOperation(patch.kind, patch.path)
        prepared.append(
            _PreparedFile(
                operation=operation,
                target=target,
                data=None if patch.kind == "delete" else rendered.encode("utf-8"),
                fuzzy_mechanisms=fuzzy,
            )
        )
    return prepared


def apply_prepared_diff(prepared: list[_PreparedFile]) -> str:
    """Apply a fully verified prepared diff and return a concise summary."""
    try:
        for item in prepared:
            if item.operation.kind == "delete":
                item.target.unlink()
                continue
            item.target.parent.mkdir(parents=True, exist_ok=True)
            item.target.write_bytes(item.data or b"")
    except OSError as exc:
        raise UnifiedDiffApplyError(
            f"filesystem write failed: {exc}", kind="write_failed"
        ) from exc
    fuzzy = []
    for item in prepared:
        fuzzy.extend(item.fuzzy_mechanisms)
    if fuzzy:
        detail = "; fuzzy=" + ",".join(fuzzy)
    else:
        detail = ""
    return f"OK: applied unified diff ({len(prepared)} file(s){detail})"


def verify_and_apply_unified_diff(
    patches: list[UnifiedFilePatch],
    cwd: str,
    *,
    candidate_count: int = 3,
) -> tuple[str, tuple[AppliedOperation, ...]]:
    prepared = prepare_unified_diff(
        patches, cwd, candidate_count=candidate_count
    )
    result = apply_prepared_diff(prepared)
    return result, tuple(item.operation for item in prepared)


__all__ = [
    "AppliedOperation",
    "DiffLine",
    "UnifiedDiffApplyError",
    "UnifiedDiffParseError",
    "UnifiedFilePatch",
    "UnifiedHunk",
    "apply_prepared_diff",
    "parse_unified_diff",
    "prepare_unified_diff",
    "verify_and_apply_unified_diff",
]
