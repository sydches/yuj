"""Session-local read-before-edit ledger with trace reconstruction.

The ledger is mechanical harness state.  A successful typed read, successful
mutation, or safely classified single-file shell read records an exact file
fingerprint.  Before edit, the current bytes are compared with that record.
Observation events make the ledger reconstructible from the append-only trace;
the model never owns or updates it.
"""
from __future__ import annotations

import hashlib
import re
import shlex
import threading
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable, Iterable, Literal, Mapping

from ._tools._common import _resolve


StaleGuardMode = Literal["off", "warn", "block"]
EventSink = Callable[[dict[str, object]], None]


class StaleGuardError(RuntimeError):
    """The ledger could not obtain a stable, contained file observation."""


@dataclass(frozen=True)
class FileFingerprint:
    mtime_ns: int
    size: int
    sha256: str

    def as_trace(self) -> dict[str, object]:
        return {
            "mtime_ns": self.mtime_ns,
            "size": self.size,
            "sha256": self.sha256,
        }

    @classmethod
    def from_trace(cls, value: Mapping[str, object]) -> "FileFingerprint":
        return cls(
            mtime_ns=int(value["mtime_ns"]),
            size=int(value["size"]),
            sha256=str(value["sha256"]),
        )


@dataclass(frozen=True)
class GuardDecision:
    path: str
    reason: str
    mode: StaleGuardMode
    allowed: bool
    message: str = ""

    @property
    def blocked(self) -> bool:
        return not self.allowed


@dataclass(frozen=True)
class ShellRead:
    """A shell command proven to read one explicit file."""

    verb: str
    path: str


_ASSIGNMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=.*$", re.DOTALL)
_GLOB_CHARS = frozenset("*?[]{}")
_READ_VERBS = {"cat", "head", "tail", "sed", "grep", "egrep", "fgrep", "rg"}


def _has_unquoted_shell_control(command: str) -> bool:
    """Reject compounds, substitutions, redirections, and backgrounding."""
    quote = ""
    escaped = False
    index = 0
    while index < len(command):
        char = command[index]
        if escaped:
            escaped = False
        elif char == "\\" and quote != "'":
            escaped = True
        elif quote:
            if char == quote:
                quote = ""
        elif char in "'\"":
            quote = char
        elif char in ";&|<>\n`":
            return True
        elif char == "$" and index + 1 < len(command) and command[index + 1] == "(":
            return True
        index += 1
    return bool(quote or escaped)


def _plain_file_token(token: str) -> bool:
    return bool(
        token
        and token not in {"-", ".", ".."}
        and not token.endswith("/")
        and not any(char in token for char in _GLOB_CHARS)
    )


def _single_cat_path(args: list[str]) -> str | None:
    paths = []
    options_done = False
    for token in args:
        if token == "--" and not options_done:
            options_done = True
        elif not options_done and token.startswith("-"):
            if not re.fullmatch(r"-[AbEnsTuv]+", token):
                return None
        else:
            paths.append(token)
    return paths[0] if len(paths) == 1 and _plain_file_token(paths[0]) else None


def _single_head_tail_path(verb: str, args: list[str]) -> str | None:
    paths: list[str] = []
    index = 0
    options_done = False
    value_options = {"-n", "--lines", "-c", "--bytes"}
    if verb == "tail":
        value_options |= {
            "--pid", "-s", "--sleep-interval", "--max-unchanged-stats",
        }
    while index < len(args):
        token = args[index]
        if token == "--" and not options_done:
            options_done = True
        elif not options_done and token in value_options:
            index += 1
            if index >= len(args):
                return None
        elif not options_done and (
            token.startswith("--lines=")
            or token.startswith("--bytes=")
            or token.startswith("--pid=")
            or token.startswith("--sleep-interval=")
            or re.fullmatch(r"-[nc][+-]?\d+", token)
            or re.fullmatch(r"[+-]\d+", token)
        ):
            pass
        elif not options_done and token.startswith("-"):
            # Formatting/follow flags do not add another file operand.
            if not re.fullmatch(r"-(?:q|v|f|F|r|z)+", token):
                return None
        else:
            paths.append(token)
        index += 1
    return paths[0] if len(paths) == 1 and _plain_file_token(paths[0]) else None


def _single_sed_path(args: list[str]) -> str | None:
    quiet = False
    scripts = 0
    operands: list[str] = []
    index = 0
    options_done = False
    while index < len(args):
        token = args[index]
        if token == "--" and not options_done:
            options_done = True
        elif not options_done and token in {"-n", "--quiet", "--silent"}:
            quiet = True
        elif not options_done and token in {"-e", "--expression"}:
            index += 1
            if index >= len(args):
                return None
            scripts += 1
        elif not options_done and (
            token.startswith("-e") and token != "-e"
            or token.startswith("--expression=")
        ):
            scripts += 1
        elif not options_done and (
            token == "-i" or token.startswith("-i")
            or token == "--in-place" or token.startswith("--in-place=")
            or token in {"-f", "--file"} or token.startswith("--file=")
        ):
            return None
        elif not options_done and token.startswith("-"):
            return None
        else:
            operands.append(token)
        index += 1
    if not quiet:
        return None
    if scripts == 0:
        if not operands:
            return None
        operands.pop(0)  # the positional sed program
    return operands[0] if len(operands) == 1 and _plain_file_token(operands[0]) else None


_GREP_AGGREGATE = {
    "-c", "--count", "--count-matches", "-l", "-L",
    "--files-with-matches", "--files-without-match", "-q", "--quiet",
    "--stats", "--json",
}
_GREP_RECURSIVE = {"-r", "-R", "--recursive", "--files"}
_GREP_VALUE_OPTIONS = {
    "-e", "--regexp", "-m", "--max-count", "-A", "--after-context",
    "-B", "--before-context", "-C", "--context", "-g", "--glob",
    "-t", "--type", "-T", "--type-not", "--color", "--encoding",
}
_GREP_FLAG_OPTIONS = {
    "--line-number", "--with-filename", "--no-filename", "--ignore-case",
    "--invert-match", "--word-regexp", "--line-regexp", "--fixed-strings",
    "--extended-regexp", "--perl-regexp", "--text", "--binary-files=text",
    "--no-messages", "--only-matching", "--hidden", "--no-heading",
    "--smart-case", "--multiline", "--case-sensitive",
}


def _single_grep_path(args: list[str]) -> str | None:
    operands: list[str] = []
    pattern_supplied = False
    index = 0
    options_done = False
    while index < len(args):
        token = args[index]
        if token == "--" and not options_done:
            options_done = True
        elif not options_done and token in _GREP_AGGREGATE | _GREP_RECURSIVE:
            return None
        elif not options_done and token in {"-f", "--file"}:
            return None  # a pattern file would be a second file read
        elif not options_done and token in _GREP_VALUE_OPTIONS:
            index += 1
            if index >= len(args):
                return None
            if token in {"-e", "--regexp"}:
                pattern_supplied = True
        elif not options_done and token in _GREP_FLAG_OPTIONS:
            pass
        elif not options_done and any(
            token.startswith(prefix)
            for prefix in (
                "--regexp=", "--max-count=", "--after-context=",
                "--before-context=", "--context=", "--glob=", "--type=",
                "--type-not=", "--color=", "--encoding=",
            )
        ):
            if token.startswith("--regexp="):
                pattern_supplied = True
        elif not options_done and re.fullmatch(r"-(?:m|A|B|C)\d+", token):
            pass
        elif not options_done and token.startswith("-"):
            # Common presentation/matching flags. Unknown option shapes fail
            # closed rather than accidentally treating an option value as path.
            if not re.fullmatch(r"-[nHhivwxyFoUaIsSP]+", token):
                return None
        else:
            operands.append(token)
        index += 1
    if not pattern_supplied:
        if not operands:
            return None
        operands.pop(0)
    return operands[0] if len(operands) == 1 and _plain_file_token(operands[0]) else None


def classify_single_file_read(command: str) -> ShellRead | None:
    """Classify a non-compound shell command that reveals one file's text."""
    if not command.strip() or _has_unquoted_shell_control(command):
        return None
    try:
        tokens = shlex.split(command, posix=True)
    except ValueError:
        return None
    while tokens and _ASSIGNMENT_RE.fullmatch(tokens[0]):
        tokens.pop(0)
    if not tokens:
        return None
    verb = tokens.pop(0).rsplit("/", 1)[-1]
    if verb not in _READ_VERBS:
        return None
    if verb == "cat":
        path = _single_cat_path(tokens)
    elif verb in {"head", "tail"}:
        path = _single_head_tail_path(verb, tokens)
    elif verb == "sed":
        path = _single_sed_path(tokens)
    else:
        path = _single_grep_path(tokens)
    return ShellRead(verb=verb, path=path) if path is not None else None


class StaleFileGuard:
    """Read ledger and policy decision point for one solver session."""

    def __init__(
        self,
        *,
        cwd: str | Path,
        mode: StaleGuardMode = "warn",
        event_sink: EventSink | None = None,
    ) -> None:
        if mode not in {"off", "warn", "block"}:
            raise ValueError("stale guard mode must be off, warn, or block")
        self.cwd = Path(cwd).resolve()
        self.mode = mode
        self.event_sink = event_sink
        self._ledger: dict[str, FileFingerprint] = {}
        self._lock = threading.RLock()

    def _emit(self, event: str, **fields: object) -> None:
        if self.event_sink is not None:
            self.event_sink({"event": event, **fields})

    def _target(self, path: str) -> tuple[Path, str]:
        try:
            target = _resolve(str(self.cwd), path)
            relative = target.relative_to(self.cwd).as_posix()
        except ValueError as exc:
            raise StaleGuardError(str(exc)) from exc
        if relative in {"", "."}:
            raise StaleGuardError("stale guard path must name a file")
        return target, relative

    @staticmethod
    def _fingerprint(target: Path) -> FileFingerprint:
        for _attempt in range(3):
            before = target.stat()
            data = target.read_bytes()
            after = target.stat()
            before_key = (before.st_ino, before.st_mtime_ns, before.st_size)
            after_key = (after.st_ino, after.st_mtime_ns, after.st_size)
            if before_key == after_key and len(data) == after.st_size:
                return FileFingerprint(
                    mtime_ns=after.st_mtime_ns,
                    size=after.st_size,
                    sha256=hashlib.sha256(data).hexdigest(),
                )
        raise StaleGuardError(f"file changed while being fingerprinted: {target.name}")

    def observe(self, path: str, *, source: str) -> FileFingerprint:
        """Record a successful read or mutation and emit reconstruction data."""
        target, relative = self._target(path)
        try:
            fingerprint = self._fingerprint(target)
        except FileNotFoundError as exc:
            raise StaleGuardError(f"file not found: {relative}") from exc
        with self._lock:
            self._ledger[relative] = fingerprint
        self._emit(
            "stale_guard_observe",
            path=relative,
            source=source,
            fingerprint=fingerprint.as_trace(),
        )
        return fingerprint

    def observe_read(self, path: str) -> FileFingerprint:
        return self.observe(path, source="read")

    def observe_mutation(self, path: str, *, source: str) -> FileFingerprint:
        if source not in {"write", "edit", "apply_patch", "udiff"}:
            raise ValueError(
                "mutation source must be write, edit, apply_patch, or udiff"
            )
        return self.observe(path, source=source)

    def observe_shell_read(self, command: str) -> ShellRead | None:
        """Credit one safely classified, successful bash read."""
        classified = classify_single_file_read(command)
        if classified is None:
            return None
        target, _relative = self._target(classified.path)
        if not target.is_file():
            return None
        self.observe(classified.path, source=f"bash:{classified.verb}")
        return classified

    def forget(self, path: str, *, source: str = "apply_patch") -> None:
        """Remove a deleted path from the ledger and record that transition."""
        _target, relative = self._target(path)
        with self._lock:
            self._ledger.pop(relative, None)
        self._emit(
            "stale_guard_observe", path=relative, source=source, fingerprint=None
        )

    def check_edit(self, path: str) -> GuardDecision:
        """Decide whether an edit may run under the configured policy."""
        if self.mode == "off":
            return GuardDecision(path=path, reason="off", mode=self.mode, allowed=True)
        target, relative = self._target(path)
        with self._lock:
            expected = self._ledger.get(relative)
            current: FileFingerprint | None = None
            if expected is None:
                reason = "unread"
            else:
                try:
                    current = self._fingerprint(target)
                except FileNotFoundError:
                    reason = "missing"
                else:
                    reason = "modified" if current.sha256 != expected.sha256 else ""
                    if not reason and current != expected:
                        # Metadata-only change does not make content stale.
                        self._ledger[relative] = current
                        self._emit(
                            "stale_guard_observe",
                            path=relative,
                            source="metadata_refresh",
                            fingerprint=current.as_trace(),
                        )
            if not reason:
                return GuardDecision(
                    path=relative, reason="fresh", mode=self.mode, allowed=True
                )

        prefix = "ERROR" if self.mode == "block" else "WARNING"
        message = f"{prefix}: stale_file: read {relative} first"
        blocked = self.mode == "block"
        self._emit(
            "stale_guard",
            path=relative,
            reason=reason,
            mode=self.mode,
            blocked=blocked,
            expected=(expected.as_trace() if expected is not None else None),
            current=(current.as_trace() if current is not None else None),
        )
        return GuardDecision(
            path=relative,
            reason=reason,
            mode=self.mode,
            allowed=not blocked,
            message=message,
        )

    @classmethod
    def from_trace(
        cls,
        *,
        cwd: str | Path,
        mode: StaleGuardMode,
        events: Iterable[Mapping[str, object]],
        event_sink: EventSink | None = None,
    ) -> "StaleFileGuard":
        """Rebuild the last observation for each path from trace events."""
        guard = cls(cwd=cwd, mode=mode, event_sink=event_sink)
        for event in events:
            if event.get("event") != "stale_guard_observe":
                continue
            path = str(event.get("path", ""))
            pure = PurePosixPath(path)
            if not path or pure.is_absolute() or ".." in pure.parts:
                raise StaleGuardError(f"invalid path in stale guard trace: {path!r}")
            fingerprint = event.get("fingerprint")
            if fingerprint is None:
                guard._ledger.pop(path, None)
            elif isinstance(fingerprint, Mapping):
                guard._ledger[path] = FileFingerprint.from_trace(fingerprint)
            else:
                raise StaleGuardError(
                    f"invalid fingerprint in stale guard trace for {path}"
                )
        return guard

    def ledger_snapshot(self) -> dict[str, FileFingerprint]:
        with self._lock:
            return dict(self._ledger)


__all__ = [
    "FileFingerprint", "GuardDecision", "ShellRead", "StaleFileGuard",
    "StaleGuardError", "StaleGuardMode", "classify_single_file_read",
]
