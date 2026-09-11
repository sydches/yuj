"""Mechanical .solver/state.json writer — harness-side, not model-side.

The model never maintains `.solver/state.json`. The harness does. This module
projects a sequence of lean `.trace.jsonl` telemetry events into a content-blind
state schema that `SolverStateContext` reads back into the prompt.

Write path: rebuild-from-trace. On each invocation, re-read the full trace
file and rewrite state.json from scratch. No in-memory accumulator, no drift.
state.json is a *view* over `.trace.jsonl`, nothing more.

Schema (target of the projection, consumed by SolverStateContext):

    {
      "state":     {"current_attempt": str, "last_verify": str,
                     "next_action": str, "last_rewind": object?,
                     "rewind_report": object?},
      "todos":     [{"description": str, "status": str}, ...],
      "trace":     [{"step": int, "session": int, "turn": int, "reasoning": str,
                     "action": str, "result": str, "next": str,
                     "gate_blocked": bool, "verification_status": str,
                     "path_binding": "recorded"|"unknown", "check_kind": str,
                     "write_like": bool,
                     "source_write_like": bool,
                     "source_write_paths": [str, ...]}, ...],
      "gates":     [{"event": "stream_rule_triggered"|"stream_rule_injection",
                      "session": int, "turn": int, ...}, ...],
      "evidence":  [{"step": int, "action": str, "result": str,
                     "verdict": "OK"|"FAIL"|"UNKNOWN", "gate_blocked": bool,
                     "kind": str}, ...],
      "inference": [],
      "tools":     {"lazy_loading_enabled": bool,
                     "active_limit": int|null,
                     "registered": [str, ...], "active": [str, ...],
                     "activations": [{"session": int, "turn": int,
                                      "requested": [str, ...],
                                      "activated": [str, ...],
                                      "already_active": [str, ...],
                                      "active": [str, ...]}, ...]}
    }

Content-blind by construction: the projection never inspects tool results
for task-format patterns (pytest nodeids, compiler error formats, lint
summary lines, or any other task-specific output shape). The only markers
it reads are harness-generated wire format: the `ERROR:` wrapper emitted
by `tools.py` on exception, the `[exit code: N]` suffix appended by
`bash()` on non-zero exit, and the `[harness gate]` prefix on gate-blocked
results. Permitted task observations can support useful harness assistance
when their source and task binding are established. This projection does not
derive new task-format diagnostics from result text.

Rewind is a structural exception to the otherwise linear projection. The raw
event list is never changed. Both the model-tool exploration collapse and the
operator/guardrail conversation-workspace action emit `rewind`; their distinct
field sets select the matching `last_rewind` metadata. Either form selects an
earlier persistent turn prefix in this derived view, and a model-tool row may
also retain its supplied goal/report.

Evidence population includes bash and run_tests calls, excluding blocked
requests and recorded unavailable or unresolved runners. Read,
write, edit, glob, grep return harness I/O status ("wrote N bytes",
"file not found"), not gate verdicts on task state. The filter is
structural (which tool was invoked), not content-based. The verdict
field is derived from the content-blind `classify_outcome`, which reads
only the harness's own exit-code marker and error wrapper — never
task output format.

Evidence kind distinguishes a command result, custom probe, check attempt and
recorded completed test check. Only recorded passed/failed check status can
advance the process verification phase; it does not establish task completion.
Path classification binds each row to its session's recorded environment.
It never consults the caller's live task mapping during replay.

Tool result text in the projection comes from `output_snippet` when present,
falling back to legacy `result_summary`. Verdicts prefer explicit `pass_fail`
telemetry so bounded snippets do not need to preserve tail exit markers. Native
unknown check outcomes remain unknown; only legacy rows use text fallback. The
`reasoning` field is the model's pre-tool assistant text for that turn. All
trace entries within a single (session, turn) share the same reasoning;
renderers that care about deduplication group by turn.

`gates` is a mechanical projection of stream-rule trigger and injection
telemetry. It contains rule identifiers, scope/offset/path metadata, and
delivery metadata, but never copies rule bodies. `inference` stays empty:
there is no content-blind population rule for it today.

Replay usage (offline, against any historical trace). ``resolve_trace_path``
finds the trace whether the run wrote it beside the workspace (current) or
inside it (pre-split runs):

    from scripts.llm_solver.harness.state_writer import project_from_trace
    from scripts.llm_solver._shared.telemetry_paths import resolve_trace_path
    state = project_from_trace(resolve_trace_path(Path("results/.../repos/<task>")))

Live usage (during a solve loop). The trace lives beside the workspace, never
inside it (see _shared/telemetry_paths.py); state.json stays in the workspace
because the model is its reader:

    from scripts.llm_solver.harness.state_writer import write_state_from_trace
    from scripts.llm_solver._shared.telemetry_paths import trace_path
    write_state_from_trace(trace_path(repo_dir), repo_dir / ".solver" / "state.json")
"""
from __future__ import annotations

import copy
import json
import re
from collections import Counter
from pathlib import Path

import orjson as _orjson

from .._shared.classification import classify_outcome, is_gate_blocked
from .file_changes import observed_mutation
from .bash_write_classification import (
    STATE_WRITER_MUTATION_PREFIXES,
    _SOURCE_EXT_RE,
    is_bash_workspace_mutation_like,
    is_workspace_path,
)
from .thoughts import thought_is_expired

# Display cap for the arguments in `action = tool(args_summary)`. Long
# arguments can reach it; action identity comes from the full-call hash.
_MAX_ACTION_CHARS = 120

# Evidence result cap — tighter than the full trace result cap because
# evidence entries are rendered into the model's context window on every
# turn. Enough for a short tail of a failing verification run without
# blowing the budget. Rendering still applies the larger rolling window
# for the full raw output; evidence is the compressed index.
_MAX_EVIDENCE_CHARS = 500

# Bump this on any non-additive change to the projected state.json shape.
# Readers use it to select the matching schema.
STATE_SCHEMA_VERSION = 1
STATE_SCHEMA_VERSION_IMPERATIVE = 2

_READ_ONLY_PREFIX_RE = re.compile(
    r"^(?:cd\s+\S+\s+&&\s+|env\s+[^;&|]+\s+)*"
    r"(?:cat|sed\s+-n|grep|rg|find|ls|head|tail|wc)\b"
)
from ._shell_patterns import (
    CHECK_COMMAND_RE as _VERIFICATION_RE, TEST_COMMAND_RE,
    command_from_summary, matches_command,
)
from .task_environment import recorded_task_environment
_CANDIDATE_INTENT_RE = re.compile(
    r"\b(?:change|edit|fix|implement|modify|patch|replace|rewrite|"
    r"apply|ready to apply|the fix is|fix is to|use .+ instead of)\b",
    re.IGNORECASE,
)
_EXPLORATION_INTENT_RE = re.compile(
    r"\b(?:understand|explore|examine|look at|read|check|first inspect|"
    r"first look|find where|search for|current state)\b",
    re.IGNORECASE,
)
_CONCRETE_EDIT_RE = re.compile(
    r"\b(?:change .+ to|change .+ from .+ to|replace .+ with|use .+ instead of|"
    r"the fix is|fix is to|ready to apply|apply the (?:edit|fix|patch)|"
    r"implement the fix|modify .+ to|patch .+)\b",
    re.IGNORECASE,
)
_NON_EDIT_PROGRESS_RE = re.compile(
    r"\b(?:run(?:ning)? (?:the )?(?:test|tests|pytest)|try running|"
    r"verify|verification|git diff|current state|check the current|"
    r"correct python environment|python environment|probe|reproducer)\b",
    re.IGNORECASE,
)
_APPLIED_EDIT_RE = re.compile(
    r"\b(?:already (?:made|applied|implemented|changed)|"
    r"(?:is|has|have|had|was|were) already "
    r"(?:made|applied|implemented|changed)|"
    r"(?:is|has|have|had|was|were) already been "
    r"(?:made|applied|implemented|changed)|"
    r"successfully (?:made|applied|implemented|changed)|"
    r"(?:change|changes|edit|edits|fix|patch) "
    r"(?:is|has|have|had|was|were) already "
    r"(?:made|applied|implemented|changed)|"
    r"marked as SUCCESS|need to verify|verify the (?:change|patch|fix)|"
    r"tests? pass)\b",
    re.IGNORECASE,
)
# Extension set sourced from bash_write_classification.SOURCE_FILE_EXTENSIONS
# (via _SOURCE_EXT_RE) so state projection recognizes go/rs/js/ts/etc. source
# paths, not just the Python-only short list this used to hardcode.
_FILE_TOKEN_RE = re.compile(
    r"(?<![\w/.-])[A-Za-z0-9_./+-]+\."
    rf"(?:{_SOURCE_EXT_RE})\b"
)


def _last_session(events: list[dict]) -> int | None:
    """Highest session_number observed in any event, or None."""
    best: int | None = None
    for ev in events:
        sn = ev.get("session_number")
        if isinstance(sn, int) and (best is None or sn > best):
            best = sn
    return best


def _last_turn(events: list[dict]) -> int | None:
    """Highest turn_number observed in any event, or None."""
    best: int | None = None
    for ev in events:
        tn = ev.get("turn_number")
        if isinstance(tn, int) and (best is None or tn > best):
            best = tn
    return best


def _last_turn_in_session(
    events: list[dict], session_number: int | None,
) -> int | None:
    """Highest tool-call turn in the latest session segment."""
    best: int | None = None
    for ev in events:
        if ev.get("event") != "tool_call":
            continue
        if (
            session_number is not None
            and ev.get("session_number") != session_number
        ):
            continue
        turn = ev.get("turn_number")
        if isinstance(turn, int) and (best is None or turn > best):
            best = turn
    return best


def _latest_edit_format(events: list[dict]) -> str:
    """Return the latest raw session-start dialect for state provenance."""
    value = ""
    for event in events:
        if event.get("event") != "session_start":
            continue
        candidate = event.get("edit_format")
        if isinstance(candidate, str) and candidate:
            value = candidate
    return value


def _event_turn(event: dict) -> int | None:
    """Return an event's turn across the two public trace field spellings."""
    value = event.get("turn_number", event.get("turn"))
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def active_events(events: list[dict]) -> list[dict]:
    """Project the active branch while preserving the append-only raw trace.

    Either public rewind form is an instruction to projections, never a
    deletion from ``.trace.jsonl``. Each completed turn names a persistent
    active prefix, so a later rewind can select either the current lineage or
    a turn from a previously discarded branch. Later chronological events
    then extend that selected prefix. The linked prefixes keep this
    reconstruction linear in the number of raw events instead of copying the
    whole view at every turn.
    """
    tail = None
    turn_views: dict[tuple[object, int], tuple] = {}

    def append(event: dict, previous):
        return (event, previous)

    def flatten(node) -> list[dict]:
        projected: list[dict] = []
        while node is not None:
            event, node = node
            projected.append(event)
        projected.reverse()
        return projected

    def filtered_prefix(node, session_number: object, to_turn: int):
        filtered = [
            event
            for event in flatten(node)
            if (
                event.get("session_number") != session_number
                or (
                    (_event_turn(event) is not None)
                    and int(_event_turn(event)) <= to_turn
                )
                or (
                    _event_turn(event) is None
                    and event.get("event")
                    not in {"handoff", "session_end", "session_exit"}
                )
            )
        ]
        rebuilt = None
        for event in filtered:
            rebuilt = append(event, rebuilt)
        return rebuilt

    for event in events:
        if event.get("event") != "rewind":
            tail = append(event, tail)
            turn = _event_turn(event)
            if turn is not None:
                turn_views[(event.get("session_number"), turn)] = tail
            continue
        session_number = event.get("session_number")
        to_turn = int(event.get("to_turn", -1))
        target = turn_views.get((session_number, to_turn))
        if target is None:
            # Older or hand-authored traces may lack a turn-bearing row for
            # the target. Preserve their prior best-effort filtering rule.
            target = filtered_prefix(tail, session_number, to_turn)
        tail = append(event, target)
    return flatten(tail)


def _extract_quoted_arg(action: str, name: str) -> str:
    marker = f"{name}="
    start = action.find(marker)
    if start < 0:
        return ""
    value_start = start + len(marker)
    if value_start >= len(action):
        return ""
    quote = action[value_start]
    if quote not in {"'", '"'}:
        return ""
    chars: list[str] = []
    escaped = False
    for char in action[value_start + 1:]:
        if escaped:
            chars.append(char)
            escaped = False
            continue
        if char == "\\":
            chars.append(char)
            escaped = True
            continue
        if char == quote:
            return "".join(chars)
        chars.append(char)
    return "".join(chars)


def _action_cmd(item: dict) -> str:
    if "_recorded_command" in item:
        return item["_recorded_command"]
    return _extract_quoted_arg(str(item.get("action") or ""), "cmd")


def _is_mutation_item(item: dict) -> bool:
    if "_mutation_like" in item:
        return item["_mutation_like"]
    if item.get("plan_artifact") is True:
        return False
    observed = observed_mutation(item)
    if observed is not None:
        return observed
    if item.get("source_write_like") is True:
        raw_paths = item.get("source_write_paths") or []
        if not raw_paths or any(
            is_workspace_path(str(path)) for path in raw_paths
        ):
            return True
    action = str(item.get("action") or "")
    if action.startswith(STATE_WRITER_MUTATION_PREFIXES):
        return True
    cmd = _action_cmd(item)
    return is_bash_workspace_mutation_like(cmd)


def _mutation_failed(item: dict) -> bool:
    if observed_mutation(item) is True:
        return False
    if item.get("gate_blocked") is True:
        return True
    if item.get("outcome_version") == "native_execution_v1":
        return (item.get("outcome") not in {"completed", "ok"}
                or item.get("exit_status") not in (None, 0))
    verdict = str(item.get("pass_fail") or "").strip().lower()
    if verdict:
        return verdict != "pass"
    return classify_outcome(str(item.get("result") or "")) == "FAIL"


def _is_successful_mutation_item(item: dict) -> bool:
    return _is_mutation_item(item) and not _mutation_failed(item)


def _is_read_only_item(item: dict) -> bool:
    if "_read_only" in item:
        return item["_read_only"]
    action = str(item.get("action") or "")
    if action.startswith(("read(", "grep(", "glob(", "list_files(")):
        return True
    cmd = _action_cmd(item).strip()
    # A leading read does not make a later check in the same call a read.
    return bool(cmd and _READ_ONLY_PREFIX_RE.search(cmd)
                and not matches_command(cmd, _VERIFICATION_RE, allow_partial=True))


def _is_check_item(item: dict) -> bool:
    """Recognize a recorded request for context, not completed verification."""
    if _verification_unavailable(item):
        return False
    if _is_mutation_item(item) or _is_read_only_item(item):
        return False
    cmd = _action_cmd(item)
    action = str(item.get("action") or "")
    return matches_command(cmd, _VERIFICATION_RE, allow_partial=True) or action.startswith("run_tests(")


def _is_verification_item(item: dict) -> bool:
    """A completed test check needs the producer's recorded execution status."""
    return (
        not _verification_unavailable(item)
        and not _is_mutation_item(item)
        and item.get("verification_status") in {"passed", "failed"}
    )


def _check_kind(item: dict) -> str:
    if _verification_unavailable(item):
        return "unavailable"
    if _is_verification_item(item):
        return "completed_test_check"
    command = _action_cmd(item)
    if str(item.get("verification_status") or "").startswith("custom_") or (
        _is_check_item(item) and command
        and not matches_command(command, TEST_COMMAND_RE)
    ):
        return "custom_probe"
    return "check_attempt" if _is_check_item(item) else "command_result"


def _verification_unavailable(item: dict) -> bool:
    """Honor recorded refusal facts before treating a request as a check.

    Absence of these facts does not establish execution or coverage; legacy
    command recognition and other verification states retain their own limits.
    """
    return bool(item.get("gate_blocked")) or item.get("verification_status") in {
        "selection_unresolved", "runner_unavailable",
    }


def _is_candidate_edit_reasoning(reasoning: str) -> bool:
    reasoning = str(reasoning or "").strip()
    if not reasoning or not _CANDIDATE_INTENT_RE.search(reasoning):
        return False
    if _APPLIED_EDIT_RE.search(reasoning):
        return False
    concrete = _CONCRETE_EDIT_RE.search(reasoning) is not None
    if _NON_EDIT_PROGRESS_RE.search(reasoning) and not concrete:
        return False
    if _EXPLORATION_INTENT_RE.search(reasoning) and not concrete:
        return False
    return True


def _compact_text(text: str, limit: int) -> str:
    compact = " ".join(str(text or "").split())
    if len(compact) > limit:
        return compact[: max(0, limit - 3)] + "..."
    return compact


def _normalize_path(path: str) -> str:
    from .bash_write_classification import normalize_trace_path
    return normalize_trace_path(path)


def _target_paths(trace: list[dict], pending_idx: int | None) -> list[str]:
    paths: list[str] = []
    scan_start = max(0, (pending_idx or len(trace) - 1) - 16)
    scan_end = (pending_idx + 1) if pending_idx is not None else len(trace)
    for item in trace[scan_start:scan_end]:
        if "_target_paths" in item:
            for path in item["_target_paths"]:
                if path not in paths:
                    paths.append(path)
            continue
        for raw_path in item.get("source_write_paths") or []:
            if not is_workspace_path(str(raw_path)):
                continue
            path = _normalize_path(str(raw_path))
            if path and path not in paths:
                paths.append(path)
        action = str(item.get("action") or "")
        cmd = _action_cmd(item)
        for source in (action, cmd):
            for match in _FILE_TOKEN_RE.finditer(source):
                raw_path = match.group(0)
                if not is_workspace_path(raw_path):
                    continue
                path = _normalize_path(raw_path)
                if path and path not in paths:
                    paths.append(path)
    return paths[:8]


def _project_process(trace: list[dict]) -> dict:
    steps = len(trace)
    mutation_indices = [
        idx for idx, item in enumerate(trace) if _is_successful_mutation_item(item)
    ]
    failed_mutation_indices = [
        idx for idx, item in enumerate(trace)
        if _is_mutation_item(item) and _mutation_failed(item)
    ]
    verify_indices = [idx for idx, item in enumerate(trace) if _is_verification_item(item)]

    pending_idx: int | None = None
    pending_reasoning = ""
    for idx in range(len(trace) - 1, -1, -1):
        item = trace[idx]
        if _is_successful_mutation_item(item):
            continue
        reasoning = str(item.get("reasoning") or "").strip()
        if not _is_candidate_edit_reasoning(reasoning):
            continue
        if any(_is_successful_mutation_item(next_item) for next_item in trace[idx + 1:]):
            continue
        pending_idx = idx
        pending_reasoning = _compact_text(reasoning, 900)
        break

    last_mutation_idx = mutation_indices[-1] if mutation_indices else None
    last_failed_mutation_idx = (
        failed_mutation_indices[-1] if failed_mutation_indices else None
    )
    last_verify_idx = verify_indices[-1] if verify_indices else None
    recent = trace[-16:]
    read_count = sum(1 for item in recent if _is_read_only_item(item))

    if (
        last_failed_mutation_idx is not None
        and (last_mutation_idx is None or last_failed_mutation_idx > last_mutation_idx)
    ):
        phase = "mutation_attempt_failed"
        suggested = "If the task requires this edit, inspect the recorded failure and choose a permitted retry."
    elif last_mutation_idx is None and pending_idx is not None:
        phase = "candidate_edit_pending"
        suggested = "If the proposed edit is supported by the task and evidence, apply it; otherwise continue the needed investigation."
    elif last_mutation_idx is None:
        phase = "pre_mutation_discovery"
        suggested = "Use the task requirements and available evidence to choose further inspection, a change, or a completion report."
    elif last_verify_idx is None or last_verify_idx < last_mutation_idx:
        phase = "post_mutation_unverified"
        suggested = "If the task requires checks for this change, run those checks and report their scope and outcome."
    else:
        phase = "post_verification"
        suggested = (
            "inspect the recorded check outcome and coverage; "
            "a completed check alone does not establish task completion"
        )

    read_counts = Counter()
    read_labels = {}
    for item in trace:
        identity = str(item.get("action_sha256") or "")
        if not identity or not _is_read_only_item(item):
            continue
        read_counts[identity] += 1
        read_labels.setdefault(
            identity, str(item.get("_display_action", item.get("action")) or "")
        )
    read_hotspots = []
    for identity, count in read_counts.most_common(3):
        if count < 3:
            continue
        read_hotspots.append({
            "action": _compact_text(read_labels[identity], 160),
            "action_sha256": identity,
            "count": count,
        })

    return {
        "steps": steps,
        "phase": phase,
        "last_mutation_step": (
            trace[last_mutation_idx].get("step") if last_mutation_idx is not None else None
        ),
        "last_failed_mutation_step": (
            trace[last_failed_mutation_idx].get("step")
            if last_failed_mutation_idx is not None else None
        ),
        "steps_since_mutation": (
            steps - last_mutation_idx - 1 if last_mutation_idx is not None else None
        ),
        "last_verification_step": (
            trace[last_verify_idx].get("step") if last_verify_idx is not None else None
        ),
        "steps_since_verification": (
            steps - last_verify_idx - 1 if last_verify_idx is not None else None
        ),
        "pending_edit_step": (
            trace[pending_idx].get("step") if pending_idx is not None else None
        ),
        "pending_edit_reasoning": pending_reasoning,
        "target_paths": _target_paths(trace, pending_idx),
        # Legacy key remains explicitly unknown. Request frequency does not
        # establish repeated information or lack of task progress.
        "read_loop": None,
        "read_activity": {
            "window_entries": len(recent),
            "read_like_requests": read_count,
            "basis": "recorded_request_syntax",
            "progress": "unassessed",
        },
        "read_hotspots": read_hotspots,
        "required_next_action": "",
        "suggested_next_action": suggested,
    }


def project(
    events: list[dict],
    *,
    max_result_chars: int,
    imperative_projection: bool = False,
    think_keep_turns: int | None = None,
) -> dict:
    """Project a list of trace events into the state.json schema.

    Deterministic, pure. Same input → same output. Content-blind.
    max_result_chars must be supplied by the caller (wired from
    cfg.max_output_chars) so the trace stores exactly what the model
    saw live.

    The returned dict carries a top-level `meta` block with the schema
    version and projection bookkeeping (event count, last session/turn
    seen) so a downstream reader can detect whether two state.json
    snapshots came from the same trace prefix.
    """
    raw_events = events
    logical_events = active_events(raw_events)
    state: dict = {}
    todos: list[dict] = []
    trace: list[dict] = []
    process_trace: list[dict] = []
    session_environments: dict = {}
    gates: list[dict] = []
    evidence: list[dict] = []
    current_session = _last_session(logical_events)
    current_turn = _last_turn_in_session(logical_events, current_session)
    retention_turn = current_turn if current_turn is not None else 0
    tools: dict = {
        "lazy_loading_enabled": False,
        "active_limit": None,
        "registered": [],
        "active": [],
        "activations": [],
    }

    step = 0
    for ev in logical_events:
        et = ev.get("event")
        if et == "session_start":
            session_environments[ev.get("session_number")] = ev.get("task_environment")
        if et == "tool_call":
            step += 1
            tool = ev.get("tool_name") or "?"
            args = _truncate(ev.get("args_summary") or "", _MAX_ACTION_CHARS)
            recorded_result = ev.get("output_snippet") or ev.get("result_summary") or ""
            result = _truncate(recorded_result, max_result_chars)
            reasoning = ev.get("reasoning") or ""
            if (
                tool == "think"
                and thought_is_expired(
                    ev.get("turn_number"),
                    current_turn=retention_turn,
                    keep_turns=think_keep_turns,
                    session_number=ev.get("session_number"),
                    current_session=current_session,
                )
            ):
                args = ""
                reasoning = ""
            action = f"{tool}({args})"
            # gate_blocked: prefer the event field (set by loop.py) with
            # fallback to wire-format detection for old traces that lack
            # it. Recognising the harness-generated gate marker is not
            # task parsing — the harness wrote it.
            blocked = ev.get("gate_blocked", is_gate_blocked(recorded_result))
            projected_step = {
                "step": step,
                "session": ev.get("session_number"),
                "turn": ev.get("turn_number"),
                "reasoning": reasoning,
                "action": action,
                "action_sha256": str(ev.get("action_sha256") or ""),
                "result": result,
                "next": "",
                "gate_blocked": blocked,
                "verification_status": str(ev.get("verification_status") or ""),
                "write_like": bool(ev.get("write_like")),
                "source_write_like": bool(ev.get("source_write_like")),
                "source_write_paths": list(ev.get("source_write_paths") or []),
                "plan_artifact": bool(ev.get("plan_artifact")),
                "pass_fail": str(ev.get("pass_fail") or ""),
                "output_sha256": str(ev.get("output_sha256") or ""),
                "output_full_path": str(ev.get("output_full_path") or ""),
            }
            # Preserve tool-owned outcome evidence before rendering advice.
            # Missing legacy evidence stays unknown; never infer it from text.
            if "file_changes" in ev:
                projected_step["file_changes"] = ev["file_changes"]
                projected_step["executed"] = ev.get("executed", True)
            if ev.get("outcome_version") == "native_execution_v1":
                for key in ("outcome_version", "outcome", "exit_status", "error_class"):
                    projected_step[key] = ev.get(key)
            if ev.get("parent_tool_call_id"):
                projected_step["parent_tool_call_id"] = str(
                    ev["parent_tool_call_id"]
                )
            if ev.get("cell_inner_index") is not None:
                projected_step["cell_inner_index"] = ev["cell_inner_index"]
            # Classify the recorded action before display clipping. These
            # private rows never expand the model-visible action/result limits.
            recorded_args = str(ev.get("args_summary") or "")
            if tool == "think" and not args:
                recorded_args = ""
            process_step = {
                **projected_step, "action": f"{tool}({recorded_args})",
                "_display_action": action, "result": recorded_result,
                "_recorded_command": (
                    command_from_summary(recorded_args) if tool == "bash" else ""
                ),
            }
            record = session_environments.get(ev.get("session_number"))
            with recorded_task_environment(record) as binding:
                process_step["_mutation_like"] = _is_mutation_item(process_step)
                process_step["_read_only"] = _is_read_only_item(process_step)
                process_step["_target_paths"] = _target_paths([process_step], None)
            projected_step["path_binding"] = "recorded" if binding is not None else "unknown"
            projected_step["check_kind"] = _check_kind(process_step)
            process_trace.append(process_step)
            trace.append(projected_step)
            state["current_attempt"] = action
            # Preserve eligible bash and run_tests outcomes. A request known
            # to be blocked or to lack a runner is not check evidence.
            # Presence in this index does not establish task coverage.
            # Native verdicts retain explicit uncertainty. Only legacy rows
            # use classify_outcome on retained harness-style output markers.
            #
            # bash is the subprocess boundary; run_tests emits its own
            # `<test_results status="passed|failed">` envelope (also harness-
            # generated) and is the canonical gate when language_quirks
            # registers a runner. Other tools are harness I/O, not gate
            # verdicts on task state.
            if tool in ("bash", "run_tests") and not _verification_unavailable(projected_step):
                pass_fail = str(ev.get("pass_fail") or "").strip().lower()
                verdict = (
                    "OK" if pass_fail == "pass"
                    else "FAIL" if pass_fail == "fail"
                    else "UNKNOWN" if ev.get("outcome_version") == "native_execution_v1"
                    else classify_outcome(recorded_result)
                )
                evidence.append({
                    "step": step,
                    "action": action,
                    "result": _truncate(result, _MAX_EVIDENCE_CHARS),
                    "verdict": verdict,
                    "gate_blocked": False,
                    "kind": projected_step["check_kind"],
                    "verification_status": projected_step["verification_status"],
                    **({key: projected_step[key] for key in
                        ("outcome_version", "outcome", "exit_status", "error_class")}
                       if "outcome_version" in projected_step else {}),
                })
        elif et == "session_end":
            fr = ev.get("finish_reason") or "?"
            sn = ev.get("session_number")
            turns = ev.get("turns") or 0
            state["last_verify"] = f"session {sn} ended: {fr} after {turns} turns"
        elif et == "plan_mode_enter":
            state["phase"] = "plan"
        elif et == "plan_mode_exit":
            state["phase"] = "implementation"
        elif et == "compaction":
            # Mechanical trace projection only. The model-authored summary
            # stays in the conversation and is never copied into state.json.
            last_compaction = {
                "session_number": ev.get("session_number"),
                "turn_number": ev.get("turn_number"),
                "tokens_before": ev.get("tokens_before"),
                "tokens_after": ev.get("tokens_after"),
                "first_kept_turn": ev.get("first_kept_turn"),
                "method": ev.get("method"),
                "fallback": ev.get("fallback"),
            }
            # Preserve compatibility with trace prefixes written before the
            # hook fields existed while projecting both fields from new rows.
            if "hook" in ev:
                last_compaction["hook"] = ev.get("hook")
            if "hook_outcome" in ev:
                last_compaction["hook_outcome"] = ev.get("hook_outcome")
            state["last_compaction"] = last_compaction
        elif et == "stream_rule_triggered":
            gates.append({
                "event": et,
                "session": ev.get("session_number"),
                "turn": ev.get("turn_number"),
                "rule": ev.get("rule"),
                "scope": ev.get("scope"),
                "offset": ev.get("offset"),
                "path": ev.get("path") or "",
                "tool_name": ev.get("tool_name") or "",
                "interrupt": bool(ev.get("interrupt")),
            })
        elif et == "stream_rule_injection":
            gates.append({
                "event": et,
                "session": ev.get("session_number"),
                "turn": ev.get("turn_number"),
                "rules": list(ev.get("rules") or []),
                "delivery": ev.get("delivery") or "",
                "context_mode": ev.get("context_mode") or "",
            })
        elif et == "todos":
            # Model-authored planning content enters state only through this
            # explicit trace event. Each event replaces the whole list; no
            # tool-call summary or prior state.json value is merged into it.
            latest = ev.get("todos")
            todos = copy.deepcopy(latest) if isinstance(latest, list) else []
        elif et == "session_start" and "active_tools" in ev:
            tools = {
                "lazy_loading_enabled": bool(
                    ev.get("tool_lazy_loading_enabled", False)
                ),
                "active_limit": ev.get("tool_active_limit"),
                "registered": list(ev.get("registered_tools") or []),
                "active": list(ev.get("active_tools") or []),
                "activations": [],
            }
        elif et == "tools_activated":
            tools["active"] = list(ev.get("active_tools") or [])
            tools["activations"].append({
                "session": ev.get("session_number"),
                "turn": ev.get("turn_number"),
                "requested": list(ev.get("requested") or []),
                "activated": list(ev.get("activated") or []),
                "already_active": list(ev.get("already_active") or []),
                "active": list(ev.get("active_tools") or []),
            })
        elif et == "rewind":
            if any(
                field in ev
                for field in (
                    "reason", "commit", "rewind_count", "rewind_id",
                    "delivery",
                )
            ):
                state["last_rewind"] = {
                    "session_number": ev.get("session_number"),
                    "from_turn": ev.get("from_turn"),
                    "to_turn": ev.get("to_turn"),
                    "reason": ev.get("reason"),
                    "commit": ev.get("commit"),
                    "rewind_id": ev.get("rewind_id"),
                    "delivery": ev.get("delivery"),
                }
            else:
                state["last_rewind"] = {
                    "session_number": ev.get("session_number"),
                    "from_turn": ev.get("from_turn"),
                    "to_turn": ev.get("to_turn"),
                    "report_chars": ev.get("report_chars"),
                }
                if isinstance(ev.get("goal"), str) and isinstance(
                    ev.get("report"), str
                ):
                    state["rewind_report"] = {
                        "goal": ev["goal"],
                        "report": ev["report"],
                    }
        elif et == "advisor_note":
            # Control metadata and the private note transcript are deliberately
            # outside the mechanical model-state projection.
            continue
        elif et == "hook":
            # Hook effects are raw execution/replay evidence. Only their
            # admitted downstream conversation or tool result may enter an
            # ordinary projected row.
            continue
        # Other session_start rows do not mutate projected task state.

    state.setdefault("current_attempt", "")
    state.setdefault("last_verify", "")
    state.setdefault("next_action", "")

    meta = {
        "schema_version": (
            STATE_SCHEMA_VERSION_IMPERATIVE
            if imperative_projection else STATE_SCHEMA_VERSION
        ),
        "event_count": len(raw_events),
        "last_session": _last_session(logical_events),
        "last_turn": _last_turn(logical_events),
        "edit_format": _latest_edit_format(raw_events),
    }
    if any(event.get("event") == "rewind" for event in raw_events):
        meta["projected_event_count"] = len(logical_events)
        meta["active_event_count"] = len(logical_events)

    projected = {
        # The meta block lets readers detect the schema version and prefix
        # without
        # walking every event. event_count / last_session / last_turn
        # computed once over the input list (cheap; events in-memory).
        # Top-level (sibling to state/trace/gates) so it is discoverable
        # without descending into the existing sections.
        "meta": meta,
        "state": state,
        "todos": todos,
        "tools": tools,
        "trace": trace,
        "gates": gates,
        "evidence": evidence,
        "inference": [],
    }
    if imperative_projection:
        projected["process"] = _project_process(process_trace)
    return projected


def project_from_trace(
    trace_path: Path,
    *,
    max_result_chars: int,
    imperative_projection: bool = False,
    think_keep_turns: int | None = None,
) -> dict:
    """Load `.trace.jsonl` and project it. Missing file → empty schema."""
    trace_path = Path(trace_path)
    if not trace_path.is_file():
        return {
            "state": {
                "current_attempt": "",
                "last_verify": "",
                "next_action": "",
            },
            "todos": [],
            "tools": {
                "lazy_loading_enabled": False,
                "active_limit": None,
                "registered": [],
                "active": [],
                "activations": [],
            },
            "trace": [],
            "gates": [],
            "evidence": [],
            "inference": [],
        }
    events = []
    with open(trace_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            events.append(json.loads(line))
    return project(
        events,
        max_result_chars=max_result_chars,
        imperative_projection=imperative_projection,
        think_keep_turns=think_keep_turns,
    )


def _write_state(state_path: Path, state: dict) -> None:
    """Atomically replace an explicit output using an exclusively created temp."""
    from tempfile import NamedTemporaryFile

    state_path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=state_path.parent, prefix=state_path.name + ".",
                            suffix=".tmp", delete=False) as saved:
        temporary = Path(saved.name)
        try:
            saved.write(_orjson.dumps(state, option=_orjson.OPT_INDENT_2))
            saved.close()
            temporary.replace(state_path)
        finally:
            temporary.unlink(missing_ok=True)


def write_state_from_events(
    events: list[dict],
    state_path: Path,
    *,
    max_result_chars: int,
    imperative_projection: bool = False,
    think_keep_turns: int | None = None,
) -> None:
    """Rebuild state.json from an in-memory list of trace events.

    Fast path used by the harness loop: Session accumulates trace
    entries in memory as it writes them to disk, so per-turn state
    refresh avoids a re-read + JSON parse of the whole trace file
    (which would scale O(T^2) in trace-length across a session).
    """
    state_path = Path(state_path)
    state = project(
        events,
        max_result_chars=max_result_chars,
        imperative_projection=imperative_projection,
        think_keep_turns=think_keep_turns,
    )
    _write_state(state_path, state)


def write_state_from_trace(
    trace_path: Path,
    state_path: Path,
    *,
    max_result_chars: int,
    imperative_projection: bool = False,
    think_keep_turns: int | None = None,
) -> None:
    """Rebuild state.json from the current contents of `.trace.jsonl`.

    Slow path used at session boundaries and by any caller without an
    in-memory events list. Re-reads the full trace file each call;
    O(T) per invocation. Prefer write_state_from_events when a
    session-local events list is available.
    """
    trace_path = Path(trace_path)
    state_path = Path(state_path)
    state = project_from_trace(
        trace_path,
        max_result_chars=max_result_chars,
        imperative_projection=imperative_projection,
        think_keep_turns=think_keep_turns,
    )
    _write_state(state_path, state)


def _truncate(s: str, n: int) -> str:
    if len(s) <= n:
        return s
    return s[: n - 3] + "..."


__all__ = [
    "active_events",
    "project",
    "project_from_trace",
    "write_state_from_trace",
]
