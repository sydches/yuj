"""Session-boundary integration for validated model-written handoffs."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ...server.types import Usage
from ..state_writer import write_state_from_trace
from .checkpoint_summary import count_messages
from .handoff_summary import (
    HandoffResult,
    generate_handoff,
    insert_handoff_into_resume_prompt,
)
from .model_role_runtime import consumer_role_client, record_role_usage
from .resume import _load_trace_events
from .trace_schema import emit_trace_event

if TYPE_CHECKING:
    from ...config import Config


@dataclass(frozen=True)
class BoundaryHandoff:
    """One attempted handoff and the side request's measured usage."""

    result: HandoffResult
    usage: Usage | None
    role_fields: dict[str, object] = field(
        default_factory=lambda: {"role": "main"}
    )

    @property
    def role(self) -> str:
        return str(self.role_fields["role"])


def generate_boundary_handoff(
    *,
    cfg: "Config",
    client: Any,
    task: str,
    trace_events: list[dict],
    session_number: int,
    tokenizer: Any = None,
) -> BoundaryHandoff:
    """Issue at most one bounded same-client request for a rollover.

    The leaf validator owns section/path/response-size checks. This runtime
    wrapper additionally rejects a request that cannot leave room for its
    declared response inside the configured context window.
    """
    usage: Usage | None = None
    routed = consumer_role_client(client, "weak")

    def _call_model(payload: dict[str, Any]) -> str:
        nonlocal usage
        request_tokens = count_messages(payload["messages"], tokenizer)
        if request_tokens + int(cfg.handoff_max_tokens) >= int(cfg.context_size):
            raise ValueError(
                "handoff request does not fit configured context: "
                f"{request_tokens} + {cfg.handoff_max_tokens} >= {cfg.context_size}"
            )
        response = routed.client.complete_side_request(payload)
        usage = response.usage
        record_role_usage(client, routed, response.usage)
        return response.content

    # The exact tokenizer enforces the request budget above. Bound raw trace
    # serialization as well so the chars/4 fallback cannot build an
    # arbitrarily large request before that check.
    history_chars = max(
        1,
        (int(cfg.context_size) - int(cfg.handoff_max_tokens)) * 3,
    )
    result = generate_handoff(
        model=cfg.model,
        task=task,
        trace_events=trace_events,
        tokenizer=tokenizer,
        max_tokens=int(cfg.handoff_max_tokens),
        call_model=_call_model,
        session_number=session_number,
        max_history_chars=history_chars,
    )
    return BoundaryHandoff(
        result=result,
        usage=usage,
        role_fields=routed.trace_fields(),
    )


def maybe_prepare_boundary_handoff(
    *,
    cfg: "Config",
    client: Any,
    task: str,
    trace_path: Path,
    trace_file: Any,
    state_path: Path | None,
    session_number: int,
    finish_reason: str,
    has_next_session: bool,
    tokenizer: Any = None,
) -> HandoffResult | None:
    """Generate and record a handoff only for an eligible live rollover."""
    if not (
        cfg.handoff_summary_enabled
        and finish_reason in {"context_full", "length", "max_turns"}
        and has_next_session
    ):
        return None
    attempt = generate_boundary_handoff(
        cfg=cfg,
        client=client,
        task=task,
        trace_events=_load_trace_events(trace_path),
        session_number=session_number,
        tokenizer=tokenizer,
    )
    emit_trace_event(
        trace_file,
        "handoff",
        session_number=session_number,
        tokens=attempt.result.tokens,
        valid=attempt.result.valid,
        fallback=attempt.result.fallback,
        **attempt.role_fields,
    )
    if state_path is not None:
        write_state_from_trace(
            trace_path,
            state_path,
            max_result_chars=cfg.max_output_chars,
            think_keep_turns=cfg.tools_think_keep_turns,
        )
    return attempt.result


def apply_pending_handoff(
    mechanical_prompt: str,
    *,
    task: str,
    handoff: HandoffResult | None,
) -> str:
    """Return the exact mechanical prompt unless a valid handoff exists."""
    if handoff is None:
        return mechanical_prompt
    return insert_handoff_into_resume_prompt(
        mechanical_prompt, task=task, handoff=handoff
    )


__all__ = [
    "BoundaryHandoff",
    "apply_pending_handoff",
    "generate_boundary_handoff",
    "maybe_prepare_boundary_handoff",
]
