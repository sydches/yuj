"""Pure formatters for SolverStateContext — free functions, no self.

Each takes the raw state.json fragment + render parameters and returns
a string. Kept side-effect-free so they're trivially testable in
isolation.
"""
from __future__ import annotations


def format_state(state) -> str:
    if not state:
        return ""
    if isinstance(state, str):
        return state.strip()
    if isinstance(state, dict):
        parts = []
        for k in ("current_attempt", "last_verify", "next_action"):
            v = state.get(k)
            if v:
                label = k.replace("_", " ").capitalize()
                parts.append(f"{label}: {v}")
        return "\n".join(parts)
    return str(state)


def format_trace(trace, max_entries: int, trace_stub_chars: int) -> str:
    """Format the tail of the trace as action/outcome stubs only.

    The trace section is the structural breadcrumb — it shows WHAT
    happened, not the raw payload. Full tool-result content lives in
    the rolling _format_tool_results window, which has its own 30K
    char budget and handles the recency cap.

    An older version put full results in both sections. The same file
    read could then appear several times in one message. This repeated
    tens of thousands of characters.

    Now every trace entry gets a short stub (trace_stub_chars,
    default 200) regardless of recency. The model reads the trace
    section to remember what it DID; it reads the rolling window to
    see the raw RESULT of recent actions. No overlap.
    """
    if not isinstance(trace, list) or not trace:
        return ""
    tail = trace[-max_entries:]
    lines: list[str] = []
    for entry in tail:
        if isinstance(entry, dict):
            step = entry.get("step", "?")
            action = entry.get("action", "")
            result = entry.get("result", "")
            nxt = entry.get("next", "")
        else:
            step, action, result, nxt = "?", str(entry), "", ""
        stub_result = (
            result[: trace_stub_chars - 3] + "..."
            if len(result) > trace_stub_chars
            else result
        )
        lines.append(f"{step} | {action} | {stub_result} | {nxt}")
    return "\n".join(lines)


def format_list(items, max_items: int) -> str:
    if not isinstance(items, list) or not items:
        return ""
    tail = items[-max_items:]
    lines = []
    for x in tail:
        if isinstance(x, dict):
            # Structured evidence entry (DRY schema).
            lines.append(f"step {x['step']}: {x['action']} → {x.get('result', '')}")
        else:
            lines.append(str(x))
    return "\n".join(lines)


def format_todo_section(todos, max_chars: int) -> str:
    """Render a bounded, line-safe todo block for the per-turn suffix."""
    if not isinstance(todos, list) or not todos or max_chars <= 0:
        return ""
    lines: list[str] = []
    for item in todos:
        if not isinstance(item, dict):
            continue
        description = " ".join(str(item.get("description") or "").split())
        status = str(item.get("status") or "")
        if description and status:
            lines.append(f"- [{status}] {description}")
    if not lines:
        return ""

    rendered = "=== Todos ===\n" + "\n".join(lines)
    if len(rendered) <= max_chars:
        return rendered
    marker = "\n... [todo list truncated]"
    if max_chars <= len(marker):
        return rendered[:max_chars]
    return rendered[: max_chars - len(marker)].rstrip() + marker


def format_state_suffix(
    suffix: str,
    todos,
    *,
    todo_char_budget: int,
) -> str:
    """Combine the bounded projected todo block with the configured suffix."""
    parts = []
    todo_section = format_todo_section(todos, todo_char_budget)
    if todo_section:
        parts.append(todo_section)
    if suffix:
        parts.append(suffix)
    return "\n\n".join(parts)
