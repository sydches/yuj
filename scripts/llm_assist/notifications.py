"""Best-effort terminal notifications for assistant sessions."""
from __future__ import annotations

import sys
from typing import TextIO


def _attention_state(success: bool, finish_reason: str | None) -> str:
    if finish_reason == "approval_required":
        return "approval required"
    if finish_reason == "input_required":
        return "input required"
    if success:
        return "completed"
    return "failed"


def send_session_notification(
    *,
    mode: str,
    session_ref: str,
    success: bool,
    finish_reason: str | None,
    interactive: bool,
    stream: TextIO | None = None,
) -> bool:
    """Ring and print a safe state line without affecting session results."""
    if mode != "bell" or not interactive:
        return False
    try:
        destination = stream if stream is not None else sys.stderr
        state = _attention_state(success, finish_reason)
        destination.write(f"\aYuj session {session_ref}: {state}\n")
        destination.flush()
    except Exception:
        return False
    return True
