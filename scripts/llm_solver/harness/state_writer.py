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
                     "gate_blocked": bool, "write_like": bool,
                     "source_write_like": bool,
                     "source_write_paths": [str, ...]}, ...],
      "gates":     [],
      "evidence":  [{"step": int, "action": str, "result": str,
                     "verdict": "OK"|"FAIL", "gate_blocked": bool}, ...],
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
results. A harness that derived intelligence from task output would be
cheating the benchmark — moving capability from the model into the loop.

Rewind is a structural exception to the otherwise linear projection. The raw
event list is never changed. Both the model-tool exploration collapse and the
operator/guardrail conversation-workspace action emit `rewind`; their distinct
field sets select the matching `last_rewind` metadata. Either form selects an
earlier persistent turn prefix in this derived view, and a model-tool row may
also retain its supplied goal/report.

Evidence population is filtered to bash calls because bash is the
subprocess execution surface where exit-code verdicts originate. Read,
write, edit, glob, grep return harness I/O status ("wrote N bytes",
"file not found"), not gate verdicts on task state. The filter is
structural (which tool was invoked), not content-based. The verdict
field is derived from the content-blind `classify_outcome`, which reads
only the harness's own exit-code marker and error wrapper — never
task output format.

Tool result text in the projection comes from `output_snippet` when present,
falling back to legacy `result_summary`. Verdicts prefer explicit `pass_fail`
telemetry so bounded snippets do not need to preserve tail exit markers. The
`reasoning` field is the model's pre-tool assistant text for that turn. All
trace entries within a single (session, turn) share the same reasoning;
renderers that care about deduplication group by turn.

`gates` and `inference` stay empty: neither has a content-blind population
rule today. They remain in the schema as protocol placeholders for the
model to read.

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
from .bash_write_classification import (
    STATE_WRITER_MUTATION_PREFIXES,
    _SOURCE_EXT_RE,
    is_bash_legacy_mutation_like,
)
from .thoughts import thought_is_expired

# Per-entry cap for the `action` column. `action` is `tool(args_summary)`;
# args are already bounded by loop.py's _summarize_args, so this is a
# safety net, never hit in practice.
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
_VERIFICATION_RE = re.compile(
    r"\b(?:pytest|py\.test|python3?\s+-m\s+pytest|unittest|"
    r"python3?\s+-c|python3?\s+-\s*<<|"
    r"tox|nox|make\s+test|cargo\s+test|go\s+test|npm\s+test|"
    # Additive: jest/vitest (mirrors language_quirks/jest.toml
    # verification_patterns) and ctest/pnpm-test/yarn-test (ctest mirrors
    # language_quirks/ctest.toml).
    r"jest|vitest|npx\s+jest|ctest|pnpm\s+test|yarn\s+test)\b"
)
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
    r"(?<![\w/.-])(?:/testbed/)?[A-Za-z0-9_./+-]+\."
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
    """Highest turn_number observed in any tool_call event, or None."""
    best: int | None = None
    for ev in events:
        if ev.get("event") != "tool_call":
            continue
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
    return _extract_quoted_arg(str(item.get("action") or ""), "cmd")


def _is_mutation_item(item: dict) -> bool:
    if item.get("source_write_like") is True:
        return True
    action = str(item.get("action") or "")
    if action.startswith(STATE_WRITER_MUTATION_PREFIXES):
        return True
    cmd = _action_cmd(item)
    return is_bash_legacy_mutation_like(cmd)


def _mutation_failed(item: dict) -> bool:
    if item.get("gate_blocked") is True:
        return True
    verdict = str(item.get("pass_fail") or "").strip().lower()
    if verdict:
        return verdict != "pass"
    return classify_outcome(str(item.get("result") or "")) == "FAIL"


def _is_successful_mutation_item(item: dict) -> bool:
    return _is_mutation_item(item) and not _mutation_failed(item)


def _is_read_only_item(item: dict) -> bool:
    action = str(item.get("action") or "")
    if action.startswith(("read(", "grep(", "glob(", "list_files(")):
        return True
    cmd = _action_cmd(item).strip()
    return bool(cmd and _READ_ONLY_PREFIX_RE.search(cmd))


def _is_verification_item(item: dict) -> bool:
    cmd = _action_cmd(item)
    action = str(item.get("action") or "")
    return bool(cmd and _VERIFICATION_RE.search(cmd)) or action.startswith("run_tests(")


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
    path = path.strip().strip("'\"")
    if path.startswith("/testbed/"):
        path = path[len("/testbed/"):]
    if path.startswith("./"):
        path = path[2:]
    return path


def _target_paths(trace: list[dict], pending_idx: int | None) -> list[str]:
    paths: list[str] = []
    scan_start = max(0, (pending_idx or len(trace) - 1) - 16)
    scan_end = (pending_idx + 1) if pending_idx is not None else len(trace)
    for item in trace[scan_start:scan_end]:
        for raw_path in item.get("source_write_paths") or []:
            path = _normalize_path(str(raw_path))
            if path and path not in paths:
                paths.append(path)
        action = str(item.get("action") or "")
        cmd = _action_cmd(item)
        for source in (action, cmd):
            for match in _FILE_TOKEN_RE.finditer(source):
                path = _normalize_path(match.group(0))
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
    recent_mutations = sum(1 for item in recent if _is_successful_mutation_item(item))
    read_loop = len(recent) >= 8 and read_count >= max(8, len(recent) - 2) and recent_mutations == 0

    if (
        last_failed_mutation_idx is not None
        and (last_mutation_idx is None or last_failed_mutation_idx > last_mutation_idx)
    ):
        phase = "mutation_attempt_failed"
        required = "retry the source edit with a write method that succeeds"
    elif last_mutation_idx is None and pending_idx is not None:
        phase = "candidate_edit_pending"
        required = "apply the pending source edit"
    elif last_mutation_idx is None:
        phase = "pre_mutation_discovery"
        required = "read only for a new narrow fact; otherwise make a source edit or name the blocker"
    elif last_verify_idx is None or last_verify_idx < last_mutation_idx:
        phase = "post_mutation_unverified"
        required = "run targeted verification for the current diff"
    else:
        phase = "post_verification"
        required = "use the latest verification result to refine the patch or call done"

    read_hotspots = []
    for action, count in Counter(
        str(item.get("action") or "") for item in trace if _is_read_only_item(item)
    ).most_common(3):
        if count < 3:
            continue
        read_hotspots.append({"action": _compact_text(action, 160), "count": count})

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
        "read_loop": read_loop,
        "read_hotspots": read_hotspots,
        "required_next_action": required,
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
        if et == "tool_call":
            step += 1
            tool = ev.get("tool_name") or "?"
            args = _truncate(ev.get("args_summary") or "", _MAX_ACTION_CHARS)
            result = _truncate(
                ev.get("output_snippet") or ev.get("result_summary") or "",
                max_result_chars,
            )
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
            blocked = ev.get("gate_blocked", is_gate_blocked(result))
            projected_step = {
                "step": step,
                "session": ev.get("session_number"),
                "turn": ev.get("turn_number"),
                "reasoning": reasoning,
                "action": action,
                "result": result,
                "next": "",
                "gate_blocked": blocked,
                "write_like": bool(ev.get("write_like")),
                "source_write_like": bool(ev.get("source_write_like")),
                "source_write_paths": list(ev.get("source_write_paths") or []),
                "pass_fail": str(ev.get("pass_fail") or ""),
                "output_sha256": str(ev.get("output_sha256") or ""),
                "output_full_path": str(ev.get("output_full_path") or ""),
            }
            if ev.get("parent_tool_call_id"):
                projected_step["parent_tool_call_id"] = str(
                    ev["parent_tool_call_id"]
                )
            if ev.get("cell_inner_index") is not None:
                projected_step["cell_inner_index"] = ev["cell_inner_index"]
            trace.append(projected_step)
            state["current_attempt"] = action
            # Evidence: every bash or run_tests call that actually ran (not
            # gate-blocked) is a verification attempt. The verdict comes from
            # the content-blind classify_outcome, which reads only the
            # harness's own exit-code marker / ERROR: wrapper / envelope
            # status — no task-format parsing.
            #
            # bash is the subprocess boundary; run_tests emits its own
            # `<test_results status="passed|failed">` envelope (also harness-
            # generated) and is the canonical gate when language_quirks
            # registers a runner. Other tools are harness I/O, not gate
            # verdicts on task state.
            if tool in ("bash", "run_tests") and not blocked:
                pass_fail = str(ev.get("pass_fail") or "").strip().lower()
                verdict = (
                    "OK" if pass_fail == "pass"
                    else "FAIL" if pass_fail == "fail"
                    else classify_outcome(result)
                )
                evidence.append({
                    "step": step,
                    "action": action,
                    "result": _truncate(result, _MAX_EVIDENCE_CHARS),
                    "verdict": verdict,
                    "gate_blocked": False,
                })
        elif et == "session_end":
            fr = ev.get("finish_reason") or "?"
            sn = ev.get("session_number")
            turns = ev.get("turns") or 0
            state["last_verify"] = f"session {sn} ended: {fr} after {turns} turns"
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
        "gates": [],
        "evidence": evidence,
        "inference": [],
    }
    if imperative_projection:
        projected["process"] = _project_process(trace)
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
    state_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = state_path.with_suffix(state_path.suffix + ".tmp")
    # orjson with OPT_INDENT_2 ⇒ same on-disk shape as the prior
    # `json.dump(..., indent=2)`. Bytes-mode write skips the text
    # encode pass; the file is consumed by readers that handle
    # both text and bytes (state.json is JSON, encoding-stable).
    tmp.write_bytes(_orjson.dumps(state, option=_orjson.OPT_INDENT_2))
    tmp.replace(state_path)


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
    state_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = state_path.with_suffix(state_path.suffix + ".tmp")
    tmp.write_bytes(_orjson.dumps(state, option=_orjson.OPT_INDENT_2))
    tmp.replace(state_path)


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
