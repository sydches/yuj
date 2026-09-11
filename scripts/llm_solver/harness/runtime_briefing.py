"""Admit optional startup observations after essential runtime guidance."""
import json

from .plan_mode import effective_model_tool_schemas
from .request_counting import has_reported_count
from .time_budget import BudgetExhausted


def admit_runtime_briefing(session):
    """Fit whole optional facts using the actual first-request projection."""
    briefing = session._runtime_briefing
    facts = briefing["facts"]
    optional = facts["runtime_observations"]
    source = json.dumps(facts, ensure_ascii=True)
    history = session.context.snapshot_messages()
    if not any(message.get("role") == "system" and source in str(message.get("content", ""))
               for message in history):
        return  # This context does not contain the startup briefing.
    counter = getattr(session, "_tokenizer", None)
    limit = int(session.cfg.context_size * session.cfg.context_fill_ratio)
    if limit <= 0:
        return  # No positive context allocation was declared.
    count_record = {}

    def measure(keep):
        nonlocal count_record
        admitted = {**facts, "runtime_observations": optional[:keep],
                    "omitted_observations": len(optional) - keep}
        target = json.dumps(admitted, ensure_ascii=True)
        messages = [
            {**message, "content": message["content"].replace(source, target)}
            if message.get("role") == "system" and isinstance(message.get("content"), str)
            else message for message in history
        ]
        # Use canonical history so changing a system fact preserves prior
        # turns and lets the existing strategy rebuild its normal projection.
        if messages != session.context.snapshot_messages() and not session.context.rewind_messages(messages):
            raise RuntimeError("context cannot apply startup briefing admission")
        if counter is None:
            return False
        try:
            count = counter.count(list(session.context.get_messages()),
                                  tools=effective_model_tool_schemas(session))
        except BudgetExhausted:
            raise
        except Exception as exc:
            count_record = {"count_precision": "unknown", "count_reason": type(exc).__name__}
            return False
        count_record = dict(getattr(counter, "last", {}) or {})
        return has_reported_count(counter, count) and count <= limit

    keep = len(optional)
    if not measure(keep):
        keep = 0
        if measure(0):
            low, high = 0, len(optional)
            while low < high:
                candidate = (low + high + 1) // 2
                if measure(candidate):
                    low = candidate
                else:
                    high = candidate - 1
            keep = low
        measure(keep)
    # Essential selection is retained even if its fit cannot be established.
    # The existing preflight and transport gates own request refusal.
    record = {"session_number": session._session_number, "optional_facts_kept": keep,
              "omitted_observations": len(optional) - keep,
              "context_allocation_tokens": limit, "request_token_count": count_record}
    briefing["report"].setdefault("briefing_admissions", []).append(record)
    session._emit("runtime_briefing_admission", **record)
