"""Trusted host-side lifecycle hooks with trace-backed replay.

Hooks are an operator extension seam, not model tools.  Live runs execute the
configured argv outside the model-command sandbox and send one JSON object on
stdin.  Replay runs consume the normalized effects saved in ``hook`` trace
rows and never launch the command again.
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
import shlex
import shutil
import signal
import subprocess
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping

log = logging.getLogger(__name__)

HOOK_EVENTS = (
    "pre_tool",
    "post_tool",
    "pre_model",
    "session_start",
    "session_end",
    "done",
)
_HOOK_FIELDS = frozenset({"matcher", "command", "timeout_s"})
_TRACE_TEXT_LIMIT = 16_384


class HookConfigurationError(ValueError):
    """A hook table is malformed or unsafe for the selected task boundary."""


class HookReplayError(RuntimeError):
    """Recorded hook provenance conflicts with the active replay contract."""


def _bounded(text: object, limit: int = _TRACE_TEXT_LIMIT) -> str:
    value = str(text or "")
    if len(value) <= limit:
        return value
    return value[:limit] + "... [hook output truncated]"


@dataclass(frozen=True, slots=True)
class HookSpec:
    event: str
    index: int
    matcher: str
    command: tuple[str, ...]
    timeout_s: float

    @property
    def command_text(self) -> str:
        return shlex.join(self.command)

    def matches(self, payload: Mapping[str, object]) -> bool:
        target = str(payload.get("tool_name") or self.event)
        if self.matcher == "*":
            return True
        if self.matcher.startswith("re:"):
            return re.fullmatch(self.matcher[3:], target) is not None
        return target == self.matcher


@dataclass(frozen=True, slots=True)
class HookEffect:
    """Combined effect of every matching handler for one lifecycle event."""

    blocked: bool = False
    reason: str = ""
    updated_input: dict[str, object] | None = None
    additional_context: str = ""

    def context_block(self) -> str:
        if not self.additional_context:
            return ""
        return (
            '<injected-fragment source="hook">\n'
            f"{self.additional_context}\n"
            "</injected-fragment>"
        )


@dataclass(frozen=True, slots=True)
class _Invocation:
    exit_code: int | None
    elapsed_ms: int
    outcome: str
    reason: str = ""
    updated_input: dict[str, object] | None = None
    additional_context: str = ""
    replayed: bool = False

    @classmethod
    def from_trace(cls, event: Mapping[str, object]) -> "_Invocation":
        outcome = str(event.get("outcome") or "error")
        if outcome not in {
            "allow", "block", "rewrite", "annotate", "timeout", "error"
        }:
            raise HookReplayError(
                f"recorded hook has unknown outcome {outcome!r}"
            )
        updated = event.get("updated_input")
        return cls(
            exit_code=(
                int(event["exit"]) if event.get("exit") is not None else None
            ),
            elapsed_ms=int(event.get("ms") or 0),
            outcome=outcome,
            reason=str(event.get("reason") or ""),
            updated_input=dict(updated) if isinstance(updated, dict) else None,
            additional_context=str(event.get("additional_context") or ""),
            replayed=True,
        )


def _handler_rows(value: object, *, event: str) -> list[Mapping[str, object]]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        return [value]
    if not isinstance(value, (list, tuple)):
        raise HookConfigurationError(
            f"config error: hooks.{event} must be a table or array of tables."
        )
    rows: list[Mapping[str, object]] = []
    for index, row in enumerate(value):
        if not isinstance(row, Mapping):
            raise HookConfigurationError(
                f"config error: hooks.{event}[{index}] must be a table."
            )
        rows.append(row)
    return rows


def _parse_command(value: object, *, path: str) -> tuple[str, ...]:
    if isinstance(value, str):
        command = (value,)
    elif isinstance(value, (list, tuple)) and all(
        isinstance(part, str) for part in value
    ):
        command = tuple(value)
    else:
        raise HookConfigurationError(
            f"config error: {path}.command must be a string or array of strings."
        )
    if not command or any(not part or "\x00" in part for part in command):
        raise HookConfigurationError(
            f"config error: {path}.command must contain non-empty argv strings."
        )
    return command


def parse_hook_specs(
    enabled: object,
    handlers: object,
) -> dict[str, tuple[HookSpec, ...]]:
    """Validate the public ``[hooks]`` shape and return immutable specs."""
    if not isinstance(enabled, bool):
        raise HookConfigurationError(
            "config error: hooks.enabled must be a boolean."
        )
    if not isinstance(handlers, Mapping):
        raise HookConfigurationError(
            "config error: hooks event settings must be tables."
        )
    unknown_events = set(handlers) - set(HOOK_EVENTS)
    if unknown_events:
        names = ", ".join(f"hooks.{name}" for name in sorted(unknown_events))
        raise HookConfigurationError(f"config error: unknown hook event(s): {names}.")

    parsed: dict[str, tuple[HookSpec, ...]] = {}
    for event in HOOK_EVENTS:
        specs: list[HookSpec] = []
        for index, row in enumerate(_handler_rows(handlers.get(event), event=event)):
            unknown_fields = set(row) - _HOOK_FIELDS
            if unknown_fields:
                names = ", ".join(sorted(unknown_fields))
                raise HookConfigurationError(
                    f"config error: hooks.{event}[{index}] has unknown field(s): "
                    f"{names}."
                )
            matcher = row.get("matcher", "*")
            if not isinstance(matcher, str) or not matcher:
                raise HookConfigurationError(
                    f"config error: hooks.{event}[{index}].matcher must be a "
                    "non-empty string."
                )
            if matcher.startswith("re:"):
                try:
                    re.compile(matcher[3:])
                except re.error as exc:
                    raise HookConfigurationError(
                        f"config error: hooks.{event}[{index}].matcher regex is "
                        f"invalid: {exc}."
                    ) from exc
            timeout = row.get("timeout_s", 10)
            if (
                isinstance(timeout, bool)
                or not isinstance(timeout, (int, float))
                or not math.isfinite(float(timeout))
                or float(timeout) <= 0
            ):
                raise HookConfigurationError(
                    f"config error: hooks.{event}[{index}].timeout_s must be a "
                    "finite number greater than zero."
                )
            specs.append(
                HookSpec(
                    event=event,
                    index=index,
                    matcher=matcher,
                    command=_parse_command(
                        row.get("command"), path=f"hooks.{event}[{index}]"
                    ),
                    timeout_s=float(timeout),
                )
            )
        parsed[event] = tuple(specs)
    return parsed


def validate_hook_settings(enabled: object, handlers: object) -> None:
    """Config-loader entry point; validation has no filesystem effects."""
    parse_hook_specs(enabled, handlers)


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _command_path_candidates(command: tuple[str, ...], task_cwd: Path) -> set[Path]:
    """Resolve executable/path argv that could select task-owned content."""
    candidates: set[Path] = set()
    root = task_cwd.absolute()
    for index, raw in enumerate(command):
        if index > 0 and raw.startswith("-"):
            continue
        expanded = Path(raw)
        lexical: Path | None = None
        if expanded.is_absolute():
            lexical = expanded.absolute()
        elif (root / expanded).exists():
            # Popen resolves relative executable and script arguments from
            # ``cwd=task_cwd``. Check that location before consulting PATH,
            # including a bare executable selected through a relative PATH
            # entry such as ``.``.
            lexical = (root / expanded).absolute()
        elif index == 0 and "/" not in raw:
            found = shutil.which(raw)
            if found:
                lexical = Path(found).absolute()
        elif index == 0 or "/" in raw:
            lexical = (root / expanded).absolute()
        if lexical is not None:
            candidates.add(lexical)
            candidates.add(lexical.resolve(strict=False))
    return candidates


def validate_hook_sandbox_paths(
    specs: Mapping[str, Iterable[HookSpec]],
    *,
    task_cwd: str | Path,
    sandbox_required: bool,
) -> None:
    """Reject host execution with a configured path controlled by the task."""
    if not sandbox_required:
        return
    root = Path(task_cwd).absolute()
    resolved_root = root.resolve(strict=False)
    for event in HOOK_EVENTS:
        for spec in specs.get(event, ()):
            for candidate in _command_path_candidates(spec.command, root):
                if _inside(candidate, root) or _inside(candidate, resolved_root):
                    raise HookConfigurationError(
                        "config error: the hook path guard forbids "
                        f"hooks.{event}[{spec.index}] command paths inside the "
                        f"task cwd ({root})."
                    )


def _first(mapping: Mapping[str, object], *names: str) -> object | None:
    for name in names:
        if name in mapping:
            return mapping[name]
    return None


def _interpret(stdout: str, exit_code: int, *, event: str) -> _Invocation:
    stripped = stdout.strip()
    parsed: Mapping[str, object] | None = None
    if stripped:
        try:
            candidate = json.loads(stripped)
        except json.JSONDecodeError:
            candidate = None
        if isinstance(candidate, Mapping):
            parsed = candidate

    nested: Mapping[str, object] = {}
    if parsed is not None:
        value = _first(parsed, "hook_specific_output", "hookSpecificOutput")
        if isinstance(value, Mapping):
            nested = value

    reason_value = None
    if parsed is not None:
        reason_value = _first(
            parsed, "stop_reason", "stopReason", "reason", "error"
        )
    reason = _bounded(reason_value or (stripped if exit_code == 2 else ""))

    if exit_code == 2:
        return _Invocation(exit_code, 0, "block", reason or "hook exited 2")
    if exit_code != 0:
        return _Invocation(exit_code, 0, "error", reason)
    if parsed is None:
        return _Invocation(exit_code, 0, "allow")

    decision = str(_first(parsed, "decision") or "").lower()
    permission = str(
        _first(nested, "permission_decision", "permissionDecision") or ""
    ).lower()
    should_continue = _first(parsed, "continue")
    if should_continue is False or decision in {"block", "deny"} or permission in {
        "deny",
        "ask",
    }:
        return _Invocation(exit_code, 0, "block", reason or "hook denied event")

    updated = _first(parsed, "updated_input", "updatedInput")
    if updated is None:
        updated = _first(nested, "updated_input", "updatedInput")
    additional = _first(parsed, "additional_context", "additionalContext")
    if additional is None:
        additional = _first(nested, "additional_context", "additionalContext")
    if additional is None:
        additional = _first(parsed, "system_message", "systemMessage")

    if additional is not None and not isinstance(additional, str):
        return _Invocation(
            exit_code, 0, "error", "additional_context must be a string"
        )

    if updated is not None:
        if event != "pre_tool" or not isinstance(updated, dict):
            return _Invocation(
                exit_code,
                0,
                "error",
                "updated_input is valid only as an object from pre_tool",
            )
        return _Invocation(
            exit_code,
            0,
            "rewrite",
            updated_input=dict(updated),
            additional_context=_bounded(additional) if additional else "",
        )
    if additional is not None:
        return _Invocation(
            exit_code, 0, "annotate", additional_context=_bounded(additional)
        )
    return _Invocation(exit_code, 0, "allow")


class HookRunner:
    """Run matching live hooks or consume matching recorded hook effects."""

    def __init__(
        self,
        *,
        enabled: bool,
        handlers: object,
        task_cwd: str | Path,
        run_dir: str | Path,
        run_id: str,
        session_number: int,
        sandbox_required: bool,
        event_sink: Callable[[dict[str, object]], None],
        replay: bool = False,
        recorded_events: Iterable[Mapping[str, object]] = (),
    ) -> None:
        self.enabled = enabled
        self.specs = parse_hook_specs(enabled, handlers)
        self.task_cwd = Path(task_cwd).absolute()
        self.run_dir = Path(run_dir).absolute()
        self.run_id = run_id
        self.session_number = int(session_number)
        self.event_sink = event_sink
        self.replay = bool(replay)
        self.sandbox_required = bool(sandbox_required)
        if self.enabled and not self.replay:
            validate_hook_sandbox_paths(
                self.specs,
                task_cwd=self.task_cwd,
                sandbox_required=self.sandbox_required,
            )
        self._recorded: dict[tuple[object, ...], deque[Mapping[str, object]]] = (
            defaultdict(deque)
        )
        for row in recorded_events:
            if int(row.get("session_number", -1)) != self.session_number:
                continue
            self._recorded[self._replay_key_from_trace(row)].append(row)

    @staticmethod
    def _replay_key_from_trace(row: Mapping[str, object]) -> tuple[object, ...]:
        return (
            str(row.get("hook_event") or ""),
            int(row.get("turn_number") or 0),
            int(row.get("hook_index") or 0),
            str(row.get("tool_call_id") or ""),
        )

    @staticmethod
    def _replay_key(
        event: str, spec: HookSpec, payload: Mapping[str, object]
    ) -> tuple[object, ...]:
        return (
            event,
            int(payload.get("turn") or 0),
            spec.index,
            str(payload.get("tool_call_id") or ""),
        )

    def _run_live(self, spec: HookSpec, payload: Mapping[str, object]) -> _Invocation:
        # Repeat the path check at invocation time so a task cannot create or
        # retarget a configured relative script after session construction.
        validate_hook_sandbox_paths(
            {spec.event: (spec,)},
            task_cwd=self.task_cwd,
            sandbox_required=self.sandbox_required,
        )
        env = os.environ.copy()
        env.update(
            {
                "YUJ_RUN_DIR": str(self.run_dir),
                "YUJ_RUN_ID": self.run_id,
                "YUJ_TASK_CWD": str(self.task_cwd),
                "YUJ_HOOK_EVENT": spec.event,
            }
        )
        stdin_text = json.dumps(payload, sort_keys=True)
        started = time.perf_counter()
        try:
            process = subprocess.Popen(
                spec.command,
                cwd=self.task_cwd,
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                start_new_session=True,
            )
        except OSError as exc:
            elapsed = round((time.perf_counter() - started) * 1000)
            return _Invocation(None, elapsed, "error", _bounded(exc))

        try:
            stdout, stderr = process.communicate(
                stdin_text, timeout=spec.timeout_s
            )
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.communicate()
            elapsed = round((time.perf_counter() - started) * 1000)
            return _Invocation(None, elapsed, "timeout", "hook timed out")

        elapsed = round((time.perf_counter() - started) * 1000)
        invocation = _interpret(stdout, int(process.returncode), event=spec.event)
        reason = invocation.reason
        if invocation.outcome == "error" and not reason:
            reason = _bounded(stderr.strip() or f"hook exited {process.returncode}")
        return _Invocation(
            invocation.exit_code,
            elapsed,
            invocation.outcome,
            reason,
            invocation.updated_input,
            invocation.additional_context,
        )

    def _run_recorded(
        self, spec: HookSpec, payload: Mapping[str, object]
    ) -> _Invocation | None:
        key = self._replay_key(spec.event, spec, payload)
        queue = self._recorded.get(key)
        if not queue:
            # Stop/exhaustion and partial replay can create lifecycle
            # boundaries that did not occur at that turn in the source. A
            # missing row therefore means no recorded effect. It must never
            # fall back to launching the live command.
            log.debug(
                "replay has no hook row for %s turn=%s hook_index=%s "
                "tool_call_id=%r; applying no effect",
                spec.event,
                key[1],
                spec.index,
                key[3],
            )
            return None
        row = queue.popleft()
        if str(row.get("command") or "") != spec.command_text:
            raise HookReplayError(
                f"replay hook command differs for {spec.event}: "
                f"recorded={row.get('command')!r} configured={spec.command_text!r}"
            )
        return _Invocation.from_trace(row)

    def _record(
        self,
        spec: HookSpec,
        payload: Mapping[str, object],
        invocation: _Invocation,
    ) -> None:
        fields: dict[str, object] = {
            "hook_event": spec.event,
            "hook_index": spec.index,
            "matcher": spec.matcher,
            "command": spec.command_text,
            "exit": invocation.exit_code,
            "ms": invocation.elapsed_ms,
            "outcome": invocation.outcome,
        }
        tool_call_id = str(payload.get("tool_call_id") or "")
        if tool_call_id:
            fields["tool_call_id"] = tool_call_id
        tool_name = str(payload.get("tool_name") or "")
        if tool_name:
            fields["tool_name"] = tool_name
        if invocation.reason:
            fields["reason"] = _bounded(invocation.reason)
        if invocation.updated_input is not None:
            fields["updated_input"] = invocation.updated_input
        if invocation.additional_context:
            fields["additional_context"] = invocation.additional_context
        if invocation.replayed:
            fields["replayed"] = True
        self.event_sink(fields)

    def run(self, event: str, **fields: object) -> HookEffect:
        if event not in HOOK_EVENTS:
            raise ValueError(f"unknown hook event: {event!r}")
        if not self.enabled:
            return HookEffect()

        payload: dict[str, object] = {
            "event": event,
            "run_id": self.run_id,
            "session": self.session_number,
            "turn": int(fields.pop("turn", 0) or 0),
            **fields,
        }
        updated_input: dict[str, object] | None = None
        annotations: list[str] = []
        for spec in self.specs[event]:
            if not spec.matches(payload):
                continue
            invocation = (
                self._run_recorded(spec, payload)
                if self.replay
                else self._run_live(spec, payload)
            )
            if invocation is None:
                continue
            self._record(spec, payload, invocation)
            if invocation.updated_input is not None:
                updated_input = dict(invocation.updated_input)
                payload["tool_args"] = dict(updated_input)
            if invocation.additional_context:
                annotations.append(invocation.additional_context)
            if invocation.outcome == "block":
                return HookEffect(
                    blocked=True,
                    reason=invocation.reason or f"{event} hook blocked",
                    updated_input=updated_input,
                    additional_context="\n\n".join(annotations),
                )
            if invocation.outcome in {"timeout", "error"}:
                log.warning(
                    "hook %s (%s) outcome=%s: %s",
                    event,
                    spec.command_text,
                    invocation.outcome,
                    invocation.reason or "no detail",
                )
        return HookEffect(
            updated_input=updated_input,
            additional_context="\n\n".join(annotations),
        )


__all__ = [
    "HOOK_EVENTS",
    "HookConfigurationError",
    "HookEffect",
    "HookReplayError",
    "HookRunner",
    "HookSpec",
    "parse_hook_specs",
    "validate_hook_sandbox_paths",
    "validate_hook_settings",
]
