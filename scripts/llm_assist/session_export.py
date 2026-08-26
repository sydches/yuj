"""Stable, redacted Markdown export for one assistant session."""
from __future__ import annotations

import hashlib
import json
import re
import stat
from dataclasses import dataclass
from pathlib import Path

from ..llm_solver.bash_quirks import apply_redactions, load_redactions
from ..llm_solver.harness.clarifications import clarification_state
from ..llm_solver.harness.corrections import validate_correction_trace
from .store import SessionRecord
from .usage import aggregate_session_usage, render_session_usage


REPORT_FORMAT = "yuj-session-report-v1"
_TRANSCRIPT_HEADER_RE = re.compile(
    r"^=== turn (\d+) (input|output) ===\s*$", re.MULTILINE
)
_TRANSCRIPT_SEGMENT_RE = re.compile(r"transcript\.pre_seg_(\d+)\.log\Z")
_PRIVATE_KEY_RE = re.compile(
    r"-----BEGIN [^-\n]*PRIVATE KEY-----.*?"
    r"-----END [^-\n]*PRIVATE KEY-----",
    re.DOTALL,
)
_AUTH_HEADER_RE = re.compile(
    r"(?i)\b(authorization\s*:\s*(?:bearer|basic)\s+)[^\s]+"
)
_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)\b(password|passwd|secret|token|api[_-]?key)"
    r"\b\s*([:=])\s*([^\s,;]+)"
)
_URL_USERINFO_RE = re.compile(
    r"(?i)\b([a-z][a-z0-9+.-]*://)([^\s/@:]+):([^\s/@]+)@"
)
_MAX_SUMMARY_CHARS = 4_000


class SessionExportError(ValueError):
    """Saved evidence cannot produce a truthful session report."""


@dataclass(frozen=True)
class _TranscriptSegment:
    number: int
    digest: str
    first_user_text: str | None
    assistant_responses: tuple[tuple[int, str], ...]


@dataclass(frozen=True)
class _ConversationItem:
    session_number: int
    turn_number: int
    order: int
    heading: str
    text: str


class _ReportRedactor:
    def __init__(self, record: SessionRecord):
        replacements: dict[str, str] = {}
        for value, replacement in (
            (record.artifact_dir, "<private-session-path>"),
            (record.worktree_path, "<workspace>"),
            (record.cwd, "<workspace>"),
            (record.system_prompt_path, "<private-config-path>"),
            (record.credential_id, "<redacted>"),
        ):
            if value:
                replacements[str(value)] = replacement
        for value in record.config_paths:
            if value:
                replacements[str(value)] = "<private-config-path>"
        self._literal_replacements = tuple(
            sorted(replacements.items(), key=lambda item: (-len(item[0]), item[0]))
        )
        self._rules = load_redactions()

    def redact(self, value: object) -> str:
        text = apply_redactions(str(value), self._rules)
        text = _PRIVATE_KEY_RE.sub("[REDACTED:private_key]", text)
        text = _AUTH_HEADER_RE.sub(r"\1[REDACTED:authorization]", text)
        text = _SECRET_ASSIGNMENT_RE.sub(
            r"\1\2[REDACTED:assigned_secret]", text
        )
        text = _URL_USERINFO_RE.sub(r"\1[REDACTED:url_credentials]@", text)
        for protected, replacement in self._literal_replacements:
            text = text.replace(protected, replacement)
        return text


def build_session_report(
    record: SessionRecord,
    *,
    status: str,
    finish_reason: str | None,
    turns: int,
) -> str:
    """Build one stable report without writing any session artifact."""
    artifact_dir = record.artifact_path
    trace_path = artifact_dir / ".trace.jsonl"
    trace_bytes = _read_owned_bytes(trace_path, artifact_dir, required=False)
    events = _parse_trace(trace_bytes)
    segments = _load_transcript_segments(artifact_dir)
    redactor = _ReportRedactor(record)
    conversation = _conversation_items(
        artifact_dir,
        events=events,
        segments=segments,
    )
    usage = aggregate_session_usage([trace_path])

    lines = [
        "# Yuj session report",
        "",
        (
            "> Yuj removed detected secrets and private paths. Review this "
            "report before you share it."
        ),
        "",
        "## Session",
        "",
        f"- Report format: `{REPORT_FORMAT}`",
        f"- Session: `{_inline(redactor.redact(record.session_id))}`",
        f"- Label: `{_inline(redactor.redact(record.label or '-'))}`",
        f"- Status: `{_inline(redactor.redact(status))}`",
        f"- Archived: `{'yes' if record.archived_at is not None else 'no'}`",
        f"- Created: `{_inline(redactor.redact(record.created_at))}`",
        f"- Updated: `{_inline(redactor.redact(record.updated_at))}`",
        f"- Model: `{_inline(redactor.redact(record.model))}`",
        f"- Provider: `{_inline(redactor.redact(record.provider or 'unknown'))}`",
        (
            "- Authentication: `"
            f"{_inline(redactor.redact(record.auth_method or 'unknown'))}`"
        ),
        f"- Context: `{_inline(redactor.redact(record.context_mode))}`",
        f"- Turns: `{turns}`",
    ]
    if record.parent_session_id is not None:
        lines.append(
            "- Parent session: `"
            f"{_inline(redactor.redact(record.parent_session_id))}`"
        )
    if record.archived_at is not None:
        lines.append(
            f"- Archived at: `{_inline(redactor.redact(record.archived_at))}`"
        )
    if finish_reason:
        lines.append(
            f"- Finish reason: `{_inline(redactor.redact(finish_reason))}`"
        )

    lines.extend(["", "## Task", ""])
    lines.extend(_markdown_block(redactor.redact(record.prompt_text)))

    lines.extend(["", "## Conversation", ""])
    if conversation:
        for item in conversation:
            lines.extend(
                [
                    (
                        f"### {item.heading} "
                        f"(session {item.session_number}, turn {item.turn_number})"
                    ),
                    "",
                ]
            )
            lines.extend(_markdown_block(redactor.redact(item.text)))
            lines.append("")
        if lines[-1] == "":
            lines.pop()
    else:
        lines.append("No saved operator follow-up or final assistant response was found.")

    lines.extend(["", "## Tool activity", ""])
    tool_events = [event for event in events if event.get("event") == "tool_call"]
    if not tool_events:
        lines.append("No tool calls were recorded.")
    else:
        for index, event in enumerate(tool_events, start=1):
            session_number = _safe_nonnegative_int(event.get("session_number"))
            turn_number = _safe_nonnegative_int(event.get("turn_number"))
            tool_name = redactor.redact(event.get("tool_name") or "unknown")
            outcome = redactor.redact(event.get("outcome") or "unknown")
            blocked = "yes" if bool(event.get("gate_blocked")) else "no"
            request = _bounded_summary(
                redactor.redact(event.get("args_summary") or "")
            )
            result = _bounded_summary(
                redactor.redact(event.get("result_summary") or "")
            )
            lines.extend(
                [
                    (
                        f"### {index}. {_heading(tool_name)} "
                        f"(session {session_number}, turn {turn_number})"
                    ),
                    "",
                    f"- Outcome: `{_inline(outcome)}`",
                    f"- Blocked: `{blocked}`",
                    "",
                    "Request summary:",
                    "",
                ]
            )
            lines.extend(_markdown_block(request or "(empty)"))
            lines.extend(["", "Result summary:", ""])
            lines.extend(_markdown_block(result or "(empty)"))
            if index != len(tool_events):
                lines.append("")

    lines.extend(["", "## Usage", ""])
    lines.extend(f"- `{_inline(line)}`" for line in render_session_usage(usage))

    lines.extend(
        [
            "",
            "## Provenance",
            "",
            (
                "- Trace SHA-256: `"
                f"{hashlib.sha256(trace_bytes).hexdigest() if trace_bytes else 'missing'}`"
            ),
            f"- Transcript segments: `{len(segments)}`",
        ]
    )
    for segment in segments:
        lines.append(
            f"- Transcript segment {segment.number} SHA-256: `{segment.digest}`"
        )
    lines.extend(
        [
            "- Workspace path: omitted",
            "- Private artifact paths: omitted",
            "- Credential identity and values: omitted",
            "- System prompts and configuration values: omitted",
            "- Raw provider requests and responses: omitted",
            "- Model reasoning: omitted",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def _conversation_items(
    artifact_dir: Path,
    *,
    events: list[dict],
    segments: tuple[_TranscriptSegment, ...],
) -> list[_ConversationItem]:
    items: list[_ConversationItem] = []
    followups = [
        event for event in events if event.get("event") == "operator_followup"
    ]
    unused_segments = list(segments[1:])
    for event in followups:
        expected_digest = str(event.get("text_sha256") or "")
        matched: _TranscriptSegment | None = None
        for segment in unused_segments:
            text = segment.first_user_text
            if text is not None and hashlib.sha256(text.encode("utf-8")).hexdigest() == expected_digest:
                matched = segment
                break
        if matched is None:
            continue
        unused_segments.remove(matched)
        items.append(
            _ConversationItem(
                session_number=_safe_nonnegative_int(event.get("session_number")),
                turn_number=0,
                order=10,
                heading="Operator follow-up",
                text=matched.first_user_text or "",
            )
        )

    clarification = clarification_state(artifact_dir)
    if clarification.request is not None:
        request = clarification.request
        items.append(
            _ConversationItem(
                session_number=_safe_nonnegative_int(request.get("session_number")),
                turn_number=_safe_nonnegative_int(request.get("turn_number")),
                order=30,
                heading="Assistant clarification question",
                text=str(request.get("question") or ""),
            )
        )
    if clarification.answer is not None:
        consumption = clarification.consumption or {}
        session_number = _safe_nonnegative_int(
            consumption.get("session_number")
            or (
                _safe_nonnegative_int(clarification.request.get("session_number")) + 1
                if clarification.request is not None
                else 0
            )
        )
        items.append(
            _ConversationItem(
                session_number=session_number,
                turn_number=0,
                order=20,
                heading="Operator clarification answer",
                text=str(clarification.answer.get("answer") or ""),
            )
        )

    correction = validate_correction_trace(artifact_dir)
    if correction.correction is not None:
        consumption = correction.consumption or {}
        items.append(
            _ConversationItem(
                session_number=_safe_nonnegative_int(
                    consumption.get("session_number")
                    or (
                        _safe_nonnegative_int(
                            correction.correction.get("after_session_number")
                        )
                        + 1
                    )
                ),
                turn_number=0,
                order=15,
                heading="Operator correction",
                text=str(correction.correction.get("text") or ""),
            )
        )

    for segment in segments:
        for turn_number, text in segment.assistant_responses:
            items.append(
                _ConversationItem(
                    session_number=segment.number,
                    turn_number=turn_number,
                    order=40,
                    heading="Assistant response",
                    text=text,
                )
            )
    return sorted(
        items,
        key=lambda item: (
            item.session_number,
            item.turn_number,
            item.order,
            item.heading,
        ),
    )


def _load_transcript_segments(artifact_dir: Path) -> tuple[_TranscriptSegment, ...]:
    numbered: list[tuple[int, Path]] = []
    for path in artifact_dir.glob("transcript.pre_seg_*.log"):
        match = _TRANSCRIPT_SEGMENT_RE.fullmatch(path.name)
        if match is not None:
            numbered.append((int(match.group(1)), path))
    numbered.sort(key=lambda item: item[0])
    current = artifact_dir / "transcript.log"
    if current.exists() or current.is_symlink():
        next_number = max((number for number, _path in numbered), default=0) + 1
        numbered.append((next_number, current))

    segments: list[_TranscriptSegment] = []
    seen_numbers: set[int] = set()
    for number, path in numbered:
        if number <= 0 or number in seen_numbers:
            raise SessionExportError("transcript segment numbering is invalid")
        seen_numbers.add(number)
        body = _read_owned_bytes(path, artifact_dir, required=True)
        text = _decode_utf8(body, label=f"transcript segment {number}")
        first_user, assistant_responses = _parse_transcript(text)
        segments.append(
            _TranscriptSegment(
                number=number,
                digest=hashlib.sha256(body).hexdigest(),
                first_user_text=first_user,
                assistant_responses=assistant_responses,
            )
        )
    return tuple(segments)


def _parse_transcript(
    text: str,
) -> tuple[str | None, tuple[tuple[int, str], ...]]:
    headers = list(_TRANSCRIPT_HEADER_RE.finditer(text))
    bodies: list[tuple[int, str, str]] = []
    for index, header in enumerate(headers):
        start = header.end()
        end = headers[index + 1].start() if index + 1 < len(headers) else len(text)
        bodies.append(
            (int(header.group(1)), header.group(2), text[start:end].strip())
        )

    first_user: str | None = None
    assistant_responses: list[tuple[int, str]] = []
    for turn_number, kind, raw_body in bodies:
        try:
            payload = json.loads(raw_body)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(payload, dict):
            continue
        if kind == "input" and first_user is None:
            messages = payload.get("messages")
            if isinstance(messages, list):
                for message in reversed(messages):
                    if isinstance(message, dict) and message.get("role") == "user":
                        candidate = _message_text(message)
                        if candidate:
                            first_user = candidate
                            break
            continue
        if kind != "output":
            continue
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            continue
        choice = choices[0]
        message = choice.get("message") if isinstance(choice, dict) else None
        if not isinstance(message, dict) or message.get("tool_calls"):
            continue
        content = _message_text(message)
        if content:
            assistant_responses.append((turn_number, content))
    return first_user, tuple(assistant_responses)


def _message_text(message: dict) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for part in content:
        if isinstance(part, str):
            parts.append(part)
            continue
        if not isinstance(part, dict):
            continue
        kind = str(part.get("type") or "")
        if kind not in {"text", "input_text", "output_text"}:
            continue
        value = part.get("text")
        if isinstance(value, str):
            parts.append(value)
    return "\n".join(parts)


def _parse_trace(body: bytes) -> list[dict]:
    if not body:
        return []
    text = _decode_utf8(body, label="trace")
    events: list[dict] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SessionExportError(
                f"trace line {line_number} is not valid JSON"
            ) from exc
        if not isinstance(event, dict):
            raise SessionExportError(
                f"trace line {line_number} is not an object"
            )
        events.append(event)
    return events


def _read_owned_bytes(path: Path, root: Path, *, required: bool) -> bytes:
    if path.parent != root:
        raise SessionExportError("session evidence path escaped its artifact root")
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        if required:
            raise SessionExportError(f"required session evidence is missing: {path.name}")
        return b""
    except OSError as exc:
        raise SessionExportError(
            f"session evidence cannot be inspected: {path.name}"
        ) from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise SessionExportError(
            f"session evidence must be a regular owned file: {path.name}"
        )
    if metadata.st_nlink != 1:
        raise SessionExportError(
            f"session evidence must not be hard-linked: {path.name}"
        )
    try:
        return path.read_bytes()
    except OSError as exc:
        raise SessionExportError(
            f"session evidence cannot be read: {path.name}"
        ) from exc


def _decode_utf8(body: bytes, *, label: str) -> str:
    try:
        return body.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SessionExportError(f"{label} is not UTF-8") from exc


def _safe_nonnegative_int(value: object) -> int:
    return value if type(value) is int and value >= 0 else 0


def _bounded_summary(text: str) -> str:
    if len(text) <= _MAX_SUMMARY_CHARS:
        return text
    omitted = len(text) - _MAX_SUMMARY_CHARS
    return text[:_MAX_SUMMARY_CHARS] + f"\n[truncated for export: {omitted} chars omitted]"


def _markdown_block(text: str) -> list[str]:
    longest = max((len(match.group(0)) for match in re.finditer(r"`+", text)), default=0)
    fence = "`" * max(3, longest + 1)
    return [f"{fence}text", text.rstrip("\n"), fence]


def _inline(text: str) -> str:
    return " ".join(str(text).split()).replace("`", "'")


def _heading(text: str) -> str:
    return " ".join(str(text).split()).replace("#", "\\#") or "unknown"


__all__ = [
    "REPORT_FORMAT",
    "SessionExportError",
    "build_session_report",
]
