"""Factual repetition notices from private completed-tool receipts."""
import hashlib
import json

from ..injections import UserTurnInjection
from ..repeated_observations import observation_key
from ..state_writer import active_events
from ..time_budget import remaining_run_seconds


def _key(row):
    if row.get("gate_blocked") or row.get("executed") is False:
        return None
    action = row.get("action_sha256")
    if (not isinstance(action, str) or len(action) != 64
            or any(c not in "0123456789abcdef" for c in action)):
        return None
    observation = observation_key(row.get("observation_receipt"))
    if observation is None:
        observation = _execution_key(row.get("execution_observation"))
        if observation is None:
            return None
    inspection = row.get("inspection_evidence")
    if inspection is not None:
        if not isinstance(inspection, dict):
            return None
        # Bind the actual file snapshot and range, independently of formatting.
        required = ("namespace", "path", "sha256", "start_line", "line_count", "total_lines")
        if any(name not in inspection for name in required):
            return None
        inspection = {name: inspection[name] for name in required}
    return json.dumps([action, observation, inspection], sort_keys=True, separators=(",", ":"))


def _execution_key(record):
    if not isinstance(record, dict) or record.get("kind") != "captured_execution_observation":
        return None
    binding = record.get("binding")
    digest = record.get("captured_result_sha256")
    if (not isinstance(binding, dict) or not binding.get("task_cwd")
            or not binding.get("command_sha256")
            or type(record.get("exit_status")) is not int
            or not isinstance(digest, str) or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)):
        return None
    return json.dumps(record, sort_keys=True, separators=(",", ":"))


def repeated_observation(session, turn):
    """Compare the latest completed result with its nearest same-action record.

    Unknown or changed results for that action break the comparison. The
    receipt proves a tool observation, not delivery, usefulness or task progress.
    """
    number = getattr(session, "_session_number", None)
    rows = [row for row in active_events(getattr(session, "_trace_events", ()))
            if row.get("event") == "tool_call"
            and type(row.get("turn_number")) is int
            and row["turn_number"] <= turn
            and (number is None or row.get("session_number") == number)]
    if not rows or rows[-1]["turn_number"] != turn:
        return None
    current = rows[-1]
    key = _key(current)
    if key is None:
        return None
    for prior in reversed(rows[:-1]):
        if prior.get("session_number") != current.get("session_number"):
            break
        if prior.get("action_sha256") != current["action_sha256"]:
            continue
        if _key(prior) != key:
            return None
        return {
            "policy": "completed_observation_notice_v1",
            "prior_turn": prior["turn_number"], "current_turn": turn,
            "prior_tool_call_id": prior.get("tool_call_id", ""),
            "tool_call_id": current.get("tool_call_id", ""),
            "observation_kind": (current.get("observation_receipt") or current["execution_observation"])["kind"],
            "recorded_exit_status": (current.get("execution_observation") or {}).get("exit_status"),
            "evidence_key": _evidence_key(current, key),
            "file_revision_bound": isinstance(current.get("inspection_evidence"), dict),
            "progress": "unknown",
        }
    return None


def _evidence_key(row, key):
    return hashlib.sha256(json.dumps([row.get("session_number"), key]).encode()).hexdigest()


def restore_observation_notices(session, events):
    """Rebuild notice suppression from the existing restored event projection."""
    keys, seen = {}, set()
    for row in events:
        if row.get("session_number") != session._session_number:
            continue
        call_id = row.get("tool_call_id")
        if row.get("event") == "tool_call":
            key = _key(row)
            keys[call_id] = _evidence_key(row, key) if key else None
        elif (row.get("event") == "user_turn_injection"
              and row.get("mechanism") == "completed_observation_notice"
              and row.get("delivery") == "user_turn" and keys.get(call_id)):
            seen.add(keys[call_id])
    pending = [item for item in session._pending_user_turn_injections
               if item.mechanism != "completed_observation_notice"
               or keys.get(item.tool_call_id) == item.ctx.get("evidence_key")]
    seen.update(item.ctx["evidence_key"] for item in pending
                if item.mechanism == "completed_observation_notice")
    session._completed_observation_notices = seen
    session._pending_user_turn_injections = pending
    session._pending_user_turn_texts = {item.text for item in pending}


def record_loop_observation_notice(session, turn):
    """Serve the explicit loop guard even when adaptive detection is off."""
    if (not getattr(getattr(session, "cfg", None), "loop_detect_enabled", False)
            or getattr(getattr(session, "_plan_mode", None), "active", False)):
        return
    row = {}
    record_observation_notice(session, turn, row)
    if "repeated_observation" in row:
        session._emit("completed_observation_notice",
                      session_number=session._session_number,
                      **row["repeated_observation"])


def record_observation_notice(session, turn, row):
    fact = repeated_observation(session, turn)
    if fact is None:
        return
    cfg = session.cfg
    fact["delivery"] = "withheld"
    row["repeated_observation"] = fact
    if (turn <= getattr(cfg, "guardrails_arm_after_turn", 0)
            or (getattr(cfg, "transformations_explicit", False)
                and not getattr(cfg, "detector_activated_guardrails", True))
            or remaining_run_seconds() == 0
            or turn - int(getattr(session, "_turn_start_offset", 0) or 0) + 1
                >= getattr(cfg, "max_turns", float("inf"))):
        return
    seen = getattr(session, "_completed_observation_notices", set())
    if fact["evidence_key"] in seen:
        fact["delivery"] = "already_notified"
        return
    queue = getattr(session, "_queue_user_turn_injection", None)
    if queue is None:
        fact["delivery"] = "unavailable"
        return
    detail = " The recorded file revision and selected range also match." if fact["file_revision_bound"] else ""
    if fact["recorded_exit_status"] is not None:
        detail += (f" The recorded exit status is {fact['recorded_exit_status']}; "
                   "its meaning depends on the command's contract.")
    text = (f"Harness observation: the latest tool call at turn {turn} returned the same "
            f"completed observation as the matching call at turn {fact['prior_turn']}."
            f"{detail} This does not establish stalled progress; further inspection may still be useful.")
    if queue(UserTurnInjection(text=text, bucket="harness_observation",
                              mechanism="completed_observation_notice",
                              tool_call_id=fact["tool_call_id"],
                              ctx={**fact, "delivery": "user_turn"})):
        seen.add(fact["evidence_key"])
        session._completed_observation_notices = seen
        fact["delivery"] = "queued"
