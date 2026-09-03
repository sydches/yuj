"""Server context lookup and last-resort prompt recovery."""
from __future__ import annotations

import copy
import json
import logging
from types import MappingProxyType
from typing import TYPE_CHECKING

from ..tool_specs import GUARDRAIL_MUTATION_TOOL_NAMES
from ..compaction_hooks import (
    Cancel,
    Compaction,
    CompactionFileOps,
    CompactionPreparation,
)

if TYPE_CHECKING:
    from ..loop import Session

log = logging.getLogger(__name__)


class CompactionOverflowError(Exception):
    """Raised when compaction cannot fit the prompt within budget even
    after truncating tool messages in the retained conversation tail.
    Caller (chat_io) treats this as terminal — better to end the
    session with a debuggable reason than send and take a server 400.
    """


def _sync_checkpoint_archive(session: "Session", messages: list[dict]) -> None:
    """Keep raw canonical messages outside the compacted visible context.

    The archive is process-local harness memory. It is never projected into
    state.json or rebuilt from transcript/model summaries. After a compaction,
    the visible list begins with the synthetic checkpoint; subsequent calls
    append only messages added beyond that remembered visible length.
    """
    archive = getattr(session, "_checkpoint_raw_messages", None)
    if archive is None:
        session._checkpoint_raw_messages = [dict(message) for message in messages]
    else:
        visible_count = int(
            getattr(session, "_checkpoint_visible_message_count", len(messages))
        )
        if len(messages) > visible_count:
            archive.extend(dict(message) for message in messages[visible_count:])
    session._checkpoint_visible_message_count = len(messages)


def _latest_assistant_turn(messages: list[dict], current_turn: int) -> int:
    """Return the logical turn retained by digest's latest assistant pair."""
    if any(message.get("role") == "assistant" for message in messages):
        return max(0, int(current_turn) - 1)
    return 0


def _session_compaction_hook(session: "Session"):
    """Return the canonical config reference and its startup-resolved hook."""
    from ..compaction_hooks import resolve_compaction_hook

    reference = str(getattr(session.cfg, "compaction_hook", "") or "").strip()
    if hasattr(session, "_compaction_hook"):
        resolved_reference = str(
            getattr(session, "_compaction_hook_reference", reference) or ""
        )
        return resolved_reference, session._compaction_hook
    hook = resolve_compaction_hook(reference)
    session._compaction_hook_reference = reference
    session._compaction_hook = hook
    return reference, hook


def _build_hook_preparation(
    session: "Session",
    messages: list[dict],
    *,
    tokenizer,
    tool_schemas: list[dict] | None,
    keep_recent_tokens: int,
    tokens_before: int,
    requested_method: str,
    safety_margin: float,
    gate_min_mutations: int,
):
    """Build the detached, trace-derived input for one hook invocation."""
    from .checkpoint_summary import (
        build_mechanical_appendix,
        select_checkpoint_cut,
    )

    cut = select_checkpoint_cut(
        messages,
        tokenizer,
        keep_recent_tokens=keep_recent_tokens,
        tools=tool_schemas,
    )
    appendix = build_mechanical_appendix(session._trace_events)
    file_ops = CompactionFileOps(
        read_files=appendix.read_files,
        modified_files=appendix.modified_files,
        last_test_runner_digest=appendix.last_test_runner_digest,
        mutation_count=appendix.mutation_count,
    )
    knobs = MappingProxyType(
        {
            "compaction_method": requested_method,
            "checkpoint_keep_recent_tokens": int(
                getattr(session.cfg, "checkpoint_keep_recent_tokens", 0) or 0
            ),
            "checkpoint_max_summary_tokens": int(
                getattr(session.cfg, "checkpoint_max_summary_tokens", 4000)
            ),
            "digest_compaction_safety_margin": safety_margin,
            "digest_keep_recent_turns": int(
                getattr(session.cfg, "digest_keep_recent_turns", 8)
            ),
            "digest_compaction_gate_min_mutations": gate_min_mutations,
        }
    )
    preparation = CompactionPreparation(
        messages_to_summarize=tuple(copy.deepcopy(list(cut.head))),
        kept_tail=tuple(copy.deepcopy(list(cut.tail))),
        previous_summary=str(
            getattr(session, "_checkpoint_previous_summary", "") or ""
        ),
        file_ops=file_ops,
        tokens_before=tokens_before,
        first_kept_turn=cut.first_kept_turn,
        knobs=knobs,
    )
    return preparation, appendix


def _validate_hook_compaction(
    session: "Session",
    candidate,
    messages: list[dict],
    *,
    appendix,
    tokenizer,
    tool_schemas: list[dict] | None,
    budget: int,
    tokens_before: int,
):
    """Apply the built-in checkpoint validator to one hook replacement."""
    from .checkpoint_summary import (
        select_checkpoint_cut_at_turn,
        validate_checkpoint_candidate,
    )

    if not isinstance(candidate.summary, str):
        raise TypeError("Compaction.summary must be a string")
    first_kept_turn = candidate.first_kept_turn
    if isinstance(first_kept_turn, bool) or not isinstance(first_kept_turn, int):
        raise TypeError("Compaction.first_kept_turn must be an integer")
    previous_first_kept_turn = int(
        getattr(session, "_checkpoint_first_kept_turn", 0) or 0
    )
    if first_kept_turn < previous_first_kept_turn:
        raise ValueError(
            "Compaction.first_kept_turn cannot move behind the previous "
            f"boundary: {first_kept_turn} < {previous_first_kept_turn}"
        )
    cut = select_checkpoint_cut_at_turn(
        messages,
        tokenizer,
        first_kept_turn=first_kept_turn,
        tools=tool_schemas,
    )
    validation = validate_checkpoint_candidate(
        candidate.summary,
        prefix=cut.prefix,
        tail=cut.tail,
        appendix=appendix,
        tokenizer=tokenizer,
        budget=budget,
        tokens_before=tokens_before,
        tools=tool_schemas,
    )
    return validation, cut


def _head_tail_truncate(text: str, char_budget: int, head_ratio: float = 0.4,
                        marker: str | None = None) -> str:
    """Head+tail char truncation, mirroring _tool_filters.truncate_output.

    Same shape as the existing tool-output truncator, but takes an
    explicit char budget instead of pulling cfg.max_output_chars. Used
    by the post-compaction overflow guard to shrink tool messages in
    latest_pair when their combined size still busts the prompt budget,
    and by the pre-flight re-clip backstop (which passes an explicit
    ``marker`` notice inserted where content was removed).
    """
    if char_budget <= 0:
        return "" if marker is None else marker
    if len(text) <= char_budget:
        return text
    marker_reserve = 80 if marker is None else len(marker) + 2
    slice_budget = max(1, char_budget - marker_reserve)
    head_budget = int(slice_budget * head_ratio)
    tail_budget = slice_budget - head_budget
    head = text[:head_budget]
    last_nl = head.rfind("\n")
    if last_nl > head_budget // 2:
        head = head[: last_nl + 1]
    tail = text[-tail_budget:]
    first_nl = tail.find("\n")
    if 0 <= first_nl < tail_budget // 2:
        tail = tail[first_nl + 1:]
    omitted = len(text) - len(head) - len(tail)
    if marker is not None:
        return f"{head}\n{marker}\n{tail}"
    return f"{head}\n[... {omitted} chars omitted by compaction overflow guard ...]\n{tail}"


def _protected_correction_text(session: "Session") -> str | None:
    text = getattr(session, "_protected_correction_text", None)
    return text if isinstance(text, str) and text else None


def _ensure_protected_correction_tail(
    session: "Session", messages: list[dict]
) -> list[dict]:
    """Keep the pending correction exact and final after compaction."""
    text = _protected_correction_text(session)
    if text is None:
        return messages
    if (
        messages
        and messages[-1].get("role") == "user"
        and messages[-1].get("content") == text
    ):
        return messages
    return [*messages, {"role": "user", "content": text}]


def preflight_reclip_oversized(
    session,
    *,
    projected_tokens: int | None = None,
    budget_tokens: int | None = None,
) -> dict | None:
    """Re-clip one recent result just enough to recover an overflowing prompt.

    This is the first, local step of the last-resort fit gate. It runs only
    after the rendered context (including halflife) exceeds
    ``context_fill_ratio * context_size``. The newest batch of tool results is
    preferred; otherwise the newest non-initial user message is eligible. The
    selected message is clipped head+tail against the prompt's actual token
    deficit, rather than against a fixed fraction of the whole context.

    Clippable messages: tool results and user messages AFTER the first
    (the initial task prompt is never clipped). System and assistant
    messages stay intact — assistant messages carry tool_calls
    structure the server round-trips.

    Token counts use the bound local tokenizer when available (exact),
    chars/4 otherwise — the same accounting the pre-flight gate itself
    uses. Persisting goes through ContextManager.replace_all_messages()
    so strategy caches invalidate; strategies that cannot replace opt
    out and the caller falls through to the legacy session end.

    Returns an info dict on success, or ``None`` when no single eligible
    message can shrink the prompt enough to be useful.
    """
    cfg = session.cfg
    ctx_size = int(getattr(cfg, "context_size", 0) or 0)
    ctx = getattr(session, "context", None)
    if ctx_size <= 0 or ctx is None:
        return None
    try:
        msgs = list(ctx.get_messages())
    except Exception:
        return None
    tokenizer = getattr(session, "_tokenizer", None)

    def _count(m: dict) -> int:
        if tokenizer is not None:
            try:
                return int(tokenizer.count([m]))
            except Exception:
                pass
        return len(str(m)) // 4

    prompt_budget = int(
        budget_tokens
        if budget_tokens is not None
        else ctx_size * float(getattr(cfg, "context_fill_ratio", 0.95))
    )
    projected_pt = int(
        projected_tokens
        if projected_tokens is not None
        else _recount_tokens(msgs, tokenizer)
    )
    if prompt_budget <= 0 or projected_pt <= prompt_budget:
        return None

    protected_correction = _protected_correction_text(session)
    first_user_seen = False
    candidates: list[tuple[int, int]] = []
    for i, m in enumerate(msgs):
        role = m.get("role")
        if role == "user" and not first_user_seen:
            first_user_seen = True  # initial task prompt — never clipped
            continue
        if (
            i == len(msgs) - 1
            and role == "user"
            and protected_correction is not None
            and m.get("content") == protected_correction
        ):
            continue
        if role not in ("tool", "user"):
            continue
        content = m.get("content")
        if not isinstance(content, str) or not content:
            continue
        pt = _count(m)
        candidates.append((i, pt))
    if not candidates:
        return None

    # Parallel tool calls can append several results after one assistant
    # message. Prefer the largest result in that newest batch. If there is no
    # trailing tool batch, use the newest eligible user message.
    last_assistant = max(
        (i for i, message in enumerate(msgs) if message.get("role") == "assistant"),
        default=-1,
    )
    trailing_tools = [
        (i, pt)
        for i, pt in candidates
        if i > last_assistant and msgs[i].get("role") == "tool"
    ]
    best_i, best_pt = (
        max(trailing_tools, key=lambda item: item[1])
        if trailing_tools
        else candidates[-1]
    )
    required_reduction = projected_pt - prompt_budget
    target_pt = best_pt - required_reduction
    if target_pt <= 0:
        return None

    target = msgs[best_i]
    content = target["content"]
    notice = (
        f"[HARNESS re-clip: this message was ~{best_pt} tokens "
        f"({len(content)} chars) and the request exceeded its "
        f"{prompt_budget}-token prompt budget. Only the head and tail are "
        f"shown (~{target_pt}-token target for this result); the middle was "
        "removed and is not "
        "recoverable from context. Re-run a narrower command (file "
        "subset, grep filter, --max-count, head/tail) to see the "
        "removed part.]"
    )
    # Convert the calculated target through this message's observed token
    # density. The small one-shot margin absorbs notice/framing variance; the
    # caller re-counts once and escalates to digest if the request still does
    # not fit.
    chars_per_token = max(1.0, len(content) / best_pt)
    char_budget = max(0, int(target_pt * chars_per_token * 0.95))
    clipped = _head_tail_truncate(content, char_budget, marker=notice)
    new_target = dict(target)
    new_target["content"] = clipped
    new_pt = _count(new_target)
    if new_pt >= best_pt:
        return None
    new_msgs = list(msgs)
    new_msgs[best_i] = new_target
    if not ctx.replace_all_messages(new_msgs):
        return None  # strategy cannot persist a replacement
    from ..savings import get_ledger
    get_ledger().record_transform(
        bucket="preflight_reclip",
        layer="harness",
        mechanism="oversized_message_head_tail",
        before=content,
        after=clipped,
        surface="context_render",
        change_count=1,
        tool_call_id=str(target.get("tool_call_id", "") or ""),
        ctx={
            "index": best_i,
            "role": target.get("role", ""),
            "orig_pt": best_pt,
            "new_pt": new_pt,
            "projected_pt": projected_pt,
            "budget_pt": prompt_budget,
            "required_reduction_pt": required_reduction,
            "context_size": ctx_size,
        },
    )
    log.warning(
        "preflight re-clip: message %d (role=%s) %d -> %d tokens "
        "(projected=%d, prompt_budget=%d, ctx=%d)",
        best_i, target.get("role"), best_pt, new_pt,
        projected_pt, prompt_budget, ctx_size,
    )
    return {
        "index": best_i,
        "role": target.get("role", ""),
        "tool_call_id": target.get("tool_call_id", ""),
        "orig_pt": best_pt,
        "new_pt": new_pt,
        "projected_pt": projected_pt,
        "budget_pt": prompt_budget,
    }


def _recount_tokens(msgs: list[dict], tokenizer, tools: list[dict] | None = None) -> int:
    """Recount tokens via the same path that produced est_pt.

    Uses the bound tokenizer when available (exact, matches server —
    pass the session's tool schemas so the count includes the tool
    catalog the request carries), falls back to chars_div_4 estimate
    otherwise. Returns 0 on any unexpected failure — caller should
    treat 0 as "cannot recount" and skip the overflow guard rather
    than raise spuriously.
    """
    if tokenizer is not None:
        try:
            return int(tokenizer.count(msgs, tools=tools))
        except Exception as e:
            log.warning("recount via tokenizer failed (%s); falling back to chars/4", e)
    try:
        from ..context import chars_div_4
        return int(chars_div_4(msgs))
    except Exception:
        return 0


def get_server_ctx(session: "Session") -> int:
    """Lazy-fetch the running llama-server's n_ctx from its /props endpoint
    AND sync cfg.context_size to it on first success.

    After this returns a non-zero value the first time, cfg.context_size
    is rewritten so the fill_ratio gate measures against the live
    server window instead of a stale config knob.

    Returns 0 if the server doesn't expose /props or the request fails;
    callers fall back to cfg.context_size in that case.
    """
    if getattr(session, "_server_ctx_cache", None) is not None:
        return session._server_ctx_cache
    session._server_ctx_cache = 0
    base = (getattr(session.cfg, "base_url", "") or "").rstrip("/")
    if not base:
        return 0
    # base_url is typically http://host:port/v1 — /props sits at the root.
    if base.endswith("/v1"):
        root = base[:-3]
    else:
        root = base
    try:
        import urllib.request
        req = urllib.request.Request(root + "/props")
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            payload = json.loads(resp.read().decode("utf-8", errors="replace"))
        for k in ("n_ctx_per_slot", "n_ctx", "default_generation_settings"):
            v = payload.get(k)
            if isinstance(v, dict):
                v = v.get("n_ctx") or v.get("n_ctx_per_slot")
            if isinstance(v, int) and v > 0:
                session._server_ctx_cache = v
                if not session._server_ctx_synced:
                    log.info(
                        "server_ctx (from /props): %d  (cfg.context_size was %d, syncing)",
                        v, getattr(session.cfg, "context_size", 0) or 0,
                    )
                    try:
                        object.__setattr__(session.cfg, "context_size", v)
                    except Exception:
                        session.cfg.context_size = v  # type: ignore[attr-defined]
                    session._server_ctx_synced = True
                return v
    except Exception as e:
        log.warning("server /props query failed (%s); falling back to cfg.context_size", e)
    return 0


def maybe_compact_messages(
    session: "Session",
    messages: list[dict],
    *,
    projected_tokens: int | None = None,
) -> list[dict]:
    """Digest or checkpoint only when the rendered prompt reaches its wall.

    Threshold = context_fill_ratio - digest_compaction_safety_margin. The
    default margin is zero, so normal context rendering and halflife get the
    whole configured prompt budget before this lossy recovery runs. A positive
    margin remains an explicit earlier-compaction override; the historical
    negative disable value continues to put the threshold above the window.

    Triggers when the best available pre-flight token count crosses threshold
    × server_ctx AND the session
    has produced at least cfg.digest_compaction_gate_min_mutations
    mutations. Replaces every assistant + tool message with one
    synthetic user-role digest block rendered from .trace.jsonl.
    Preserves the leading system message and the initial user
    task message verbatim.

    Checkpoint mode retains raw canonical messages in process, asks the same
    model for a structured checkpoint, and validates structure/path coverage/
    shrinkage before use. Every failure takes the unchanged digest path.
    """
    cfg = session.cfg
    configured_method = str(getattr(cfg, "compaction_method", "digest"))
    protected_correction = _protected_correction_text(session)
    protected_archive_tracking = bool(
        protected_correction is not None
        and (
            configured_method == "checkpoint"
            or str(getattr(cfg, "compaction_hook", "") or "").strip()
        )
    )
    if protected_correction is not None:
        # The correction must reach the next model request before any other
        # model work. Use only the content-blind digest path at this boundary;
        # a checkpoint model or operator hook may run again on later turns.
        requested_method = "digest"
        hook_reference, hook = "", None
    else:
        requested_method = str(
            getattr(session, "_compaction_method_override", "")
            or configured_method
        )
        hook_reference, hook = _session_compaction_hook(session)
    if (
        requested_method == "checkpoint"
        or hook is not None
        or protected_archive_tracking
    ):
        _sync_checkpoint_archive(session, messages)
    # Route through the bound method so test mocks patching
    # `Session._get_server_ctx` continue to intercept this call site.
    ctx_size = session._get_server_ctx() or int(getattr(cfg, "context_size", 0) or 0)
    if ctx_size <= 0:
        return messages
    fill_ratio = float(getattr(cfg, "context_fill_ratio", 0.95))
    safety_margin = float(getattr(cfg, "digest_compaction_safety_margin", 0.0))
    threshold = max(0.0, fill_ratio - safety_margin)
    gate_min_mut = int(getattr(cfg, "digest_compaction_gate_min_mutations", 1))
    budget = int(threshold * ctx_size)
    # Cheap-estimate fast path. The exact tokenize via
    # local_tokenizer.count() renders the model's chat template +
    # tokenizes the entire ~25k-token message list — measured
    # 10–50 ms per call on this host, hot per turn. Most turns are
    # far below the compaction budget; running the exact count just
    # to confirm "still under threshold" is wasted work.
    #
    # Strategy: cheap chars/4 estimate first. With a 10% safety
    # margin (chars/4 underestimates non-English text), if the
    # cheap estimate is well under budget the exact count would be
    # too — skip it. Only when we're within the margin do we pay
    # for the exact count to make a sound compaction decision.
    try:
        from ..context import chars_div_4
        cheap_est = int(chars_div_4(messages))
    except Exception:
        cheap_est = None
    projected_pt = int(projected_tokens or 0)
    if (
        projected_pt <= budget
        and cheap_est is not None
        and cheap_est * 1.10 < budget
    ):
        # Comfortably under budget; no compaction needed and no
        # need to pay for exact tokenization on this turn.
        return messages

    # Pre-flight exact count of the about-to-send messages. The
    # local tokenizer matches the server's tokenizer (same vocab
    # as the GGUF) and renders the model's chat template, so this
    # is the same count the server will produce. When the
    # tokenizer is unset, fall back to chars_div_4 (the historical
    # behavior).
    tokenizer = getattr(session, "_tokenizer", None)
    from ..plan_mode import effective_model_tool_schemas
    tool_schemas = effective_model_tool_schemas(session)
    if tokenizer is not None:
        try:
            measured_pt = int(tokenizer.count(messages, tools=tool_schemas))
        except Exception as e:
            log.warning("local tokenizer count failed (%s); skipping compaction check", e)
            return messages
    elif cheap_est is not None:
        measured_pt = cheap_est
    else:
        return messages
    est_pt = max(measured_pt, projected_pt)
    if est_pt <= budget:
        return messages
    mutation_count = sum(1 for ev in session._trace_events
                         if ev.get("event") == "tool_call"
                         and str(ev.get("tool_name", ""))
                         in GUARDRAIL_MUTATION_TOOL_NAMES)
    if mutation_count < gate_min_mut:
        return messages

    compaction_fallback = ""
    compaction_role_fields: dict[str, object] = {"role": "main"}
    first_kept_turn = _latest_assistant_turn(
        messages, getattr(session, "_compaction_turn", 0)
    )
    latest_pair: list[dict] = []
    new_messages: list[dict] | None = None
    event_method = requested_method
    hook_outcome = "not_configured"
    keep_recent_tokens = int(
        getattr(cfg, "checkpoint_keep_recent_tokens", 0) or 0
    ) or max(4096, int(0.20 * ctx_size))

    if hook is not None:
        source_messages = getattr(session, "_checkpoint_raw_messages", messages)
        hook_outcome = "fallback_digest"
        preparation = None
        appendix = None
        try:
            preparation, appendix = _build_hook_preparation(
                session,
                source_messages,
                tokenizer=tokenizer,
                tool_schemas=tool_schemas,
                keep_recent_tokens=keep_recent_tokens,
                tokens_before=est_pt,
                requested_method=requested_method,
                safety_margin=safety_margin,
                gate_min_mutations=gate_min_mut,
            )
            hook_result = hook(preparation)
        except Exception as exc:  # noqa: BLE001 - digest is the fail-safe
            event_method = "hook"
            compaction_fallback = "digest"
            compaction_role_fields = {"role": None}
            log.warning(
                "compaction hook %s failed; using digest fallback: %s: %s",
                hook_reference,
                type(exc).__name__,
                exc,
            )
        else:
            if hook_result is None:
                hook_outcome = "default"
            elif isinstance(hook_result, Cancel):
                hook_outcome = "cancel"
                session._emit(
                    "compaction",
                    session_number=getattr(session, "_session_number", 0),
                    turn_number=int(
                        getattr(session, "_compaction_turn", 0) or 0
                    ),
                    tokens_before=est_pt,
                    tokens_after=est_pt,
                    first_kept_turn=preparation.first_kept_turn,
                    method="hook",
                    fallback="",
                    role=None,
                    hook=hook_reference,
                    hook_outcome=hook_outcome,
                )
                log.info(
                    "compaction hook %s canceled compaction at turn %d",
                    hook_reference,
                    getattr(session, "_compaction_turn", -1),
                )
                return messages
            elif isinstance(hook_result, Compaction):
                event_method = "hook"
                compaction_role_fields = {"role": None}
                try:
                    validation, hook_cut = _validate_hook_compaction(
                        session,
                        hook_result,
                        source_messages,
                        appendix=appendix,
                        tokenizer=tokenizer,
                        tool_schemas=tool_schemas,
                        budget=budget,
                        tokens_before=est_pt,
                    )
                except Exception as exc:  # noqa: BLE001 - digest is fail-safe
                    compaction_fallback = "digest"
                    log.warning(
                        "compaction hook %s returned an invalid replacement; "
                        "using digest fallback: %s: %s",
                        hook_reference,
                        type(exc).__name__,
                        exc,
                    )
                else:
                    if (
                        validation.valid
                        and validation.compacted_messages is not None
                    ):
                        new_messages = [
                            dict(message)
                            for message in validation.compacted_messages
                        ]
                        first_kept_turn = hook_cut.first_kept_turn
                        session._checkpoint_previous_summary = (
                            validation.model_summary
                        )
                        session._checkpoint_first_kept_turn = first_kept_turn
                        hook_outcome = "replace"
                    else:
                        compaction_fallback = "digest"
                        log.warning(
                            "compaction hook %s replacement failed checkpoint "
                            "validation; using digest fallback: %s",
                            hook_reference,
                            validation.reason,
                        )
            else:
                event_method = "hook"
                compaction_fallback = "digest"
                compaction_role_fields = {"role": None}
                log.warning(
                    "compaction hook %s returned %s, expected None, Cancel, "
                    "or Compaction; using digest fallback",
                    hook_reference,
                    type(hook_result).__name__,
                )

    # A hook replacement does not need the digest. Every built-in or fallback
    # path retains the existing requirement for a valid trace-derived digest.
    if new_messages is None:
        if session._trace_path is None or not session._trace_path.is_file():
            return messages
        try:
            # Import the content-blind rendering core from _shared.
            from ..._shared.digest_core import (
                DigestOptions,
                _load as digest_load,
                render_digest,
            )
            rows = digest_load(session._trace_path)
        except Exception as e:
            log.warning("compaction: digest load failed (%s); skipping", e)
            return messages
        if not rows:
            return messages
        digest_text = render_digest(rows, DigestOptions(
            reasoning=False,
            expand_flags=(),
            tail_threshold=10**9,
            collapse_harness=True,
            head_chars=80,
        ))
        if not digest_text.strip():
            return messages

    if (
        new_messages is None
        and event_method != "hook"
        and requested_method == "checkpoint"
    ):
        from .checkpoint_summary import generate_checkpoint
        from .model_role_runtime import consumer_role_client, record_role_usage

        routed = consumer_role_client(session, "weak")
        compaction_role_fields = routed.trace_fields()

        def _call_checkpoint(payload: dict) -> str:
            side_result = routed.client.complete_side_request(payload)
            record_role_usage(session, routed, side_result.usage)
            return side_result.content

        checkpoint = generate_checkpoint(
            model=cfg.model,
            messages=getattr(session, "_checkpoint_raw_messages", messages),
            trace_events=session._trace_events,
            tokenizer=tokenizer,
            keep_recent_tokens=keep_recent_tokens,
            max_summary_tokens=int(cfg.checkpoint_max_summary_tokens),
            budget=budget,
            call_model=_call_checkpoint,
            tools=tool_schemas,
            previous_summary=str(
                getattr(session, "_checkpoint_previous_summary", "") or ""
            ),
            previous_first_kept_turn=int(
                getattr(session, "_checkpoint_first_kept_turn", 0) or 0
            ),
            tokens_before=est_pt,
        )
        if checkpoint.valid and checkpoint.compacted_messages is not None:
            new_messages = [dict(message) for message in checkpoint.compacted_messages]
            first_kept_turn = int(checkpoint.first_kept_turn or 0)
            session._checkpoint_previous_summary = checkpoint.model_summary
            session._checkpoint_first_kept_turn = first_kept_turn
        else:
            compaction_fallback = "digest"
            if checkpoint.first_kept_turn is not None:
                first_kept_turn = int(checkpoint.first_kept_turn)
            log.warning(
                "checkpoint validation failed; using digest fallback: %s",
                checkpoint.reason,
            )

    if new_messages is None:
        system_msgs = [m for m in messages if m.get("role") == "system"]
        # First non-system user message = the original task prompt; keep it.
        initial_user = None
        for m in messages:
            if m.get("role") == "user":
                initial_user = m
                break
        # Preserve the most recent assistant + tool message pair verbatim so
        # the model gets a full-fidelity turn on the just-arrived tool result.
        last_assistant_idx = None
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "assistant":
                last_assistant_idx = i
                break
        if last_assistant_idx is not None:
            latest_pair.append(messages[last_assistant_idx])
            for m in messages[last_assistant_idx + 1:]:
                if m.get("role") == "tool":
                    latest_pair.append(m)
        compacted_block = (
            f"[HARNESS: prompt crossed {threshold:.2f} of the server context window "
            f"(est_pt={est_pt}, ctx={ctx_size}). The prior assistant + tool "
            "history has been replaced by the per-turn digest below; the "
            "most recent assistant + tool exchange is preserved verbatim "
            "after the digest. Continue from the most recent state.]\n\n"
            "=== Compacted history (one line per turn) ===\n"
            + digest_text
        )
        new_messages = list(system_msgs)
        if initial_user is not None:
            new_messages.append(initial_user)
        new_messages.append({"role": "user", "content": compacted_block})
        new_messages.extend(latest_pair)

    # ── Overflow guard ─────────────────────────────────────────────
    # A retained tail can still exceed budget. Recount; if over, truncate
    # retained tool messages while preserving assistant structure, then
    # recount once more. If still over, raise a typed error so the session
    # ends with a debuggable reason instead of a server 400.
    new_messages = _ensure_protected_correction_tail(
        session, new_messages
    )
    final_count = _recount_tokens(new_messages, tokenizer, tools=tool_schemas)
    if final_count > 0 and final_count > budget:
        tool_indices = [i for i, m in enumerate(new_messages)
                        if m.get("role") == "tool"]
        if tool_indices:
            # Shallow-copy tool dicts before mutating .content so we
            # don't poison the caller's original messages list (the
            # latest_pair entries share object refs with `messages`).
            for i in tool_indices:
                new_messages[i] = dict(new_messages[i])
            # Conservative char-per-token estimate (5×) + 500-char
            # buffer absorbs both the head/tail marker overhead AND
            # the chars_div_4 vs exact-tokenizer discrepancy when
            # tokenizer is unset. One-shot reduction; we accept a
            # small overshoot (truncate more than strictly necessary)
            # in exchange for not re-tokenizing in a loop.
            over_tokens = final_count - budget
            target_reduction_chars = over_tokens * 5 + 500
            total_tool_chars = sum(
                len(new_messages[i].get("content", ""))
                for i in tool_indices
                if isinstance(new_messages[i].get("content"), str)
            )
            new_total_tool_chars = max(0, total_tool_chars - target_reduction_chars)
            per_msg_budget = new_total_tool_chars // max(1, len(tool_indices))
            for i in tool_indices:
                content = new_messages[i].get("content")
                if isinstance(content, str):
                    new_messages[i]["content"] = _head_tail_truncate(content, per_msg_budget)
            log.warning(
                "compaction overflow guard: retained tail exceeded budget by ~%d tokens; "
                "truncated %d tool message(s) to ~%d chars each",
                over_tokens, len(tool_indices), per_msg_budget,
            )
            final_count = _recount_tokens(new_messages, tokenizer, tools=tool_schemas)
        if final_count > 0 and final_count > budget:
            raise CompactionOverflowError(
                f"compaction cannot fit prompt within budget after truncation: "
                f"final_count={final_count} > budget={budget} "
                f"(ctx={ctx_size}, threshold={threshold:.2f}, "
                f"retained_tool_msgs={len(tool_indices)})"
            )

    session._compacted = True
    session._compaction_count = int(getattr(session, "_compaction_count", 0)) + 1
    log.info(
        "compaction fired (#%d) at turn %d: threshold=%.2f, est_pt=%d, ctx=%d, "
        "mutation_count=%d, method=%s, hook_outcome=%s, "
        "old_msg_count=%d -> new=%d",
        session._compaction_count,
        getattr(session, "_compaction_turn", -1),
        threshold, est_pt, ctx_size, mutation_count,
        event_method, hook_outcome,
        len(messages), len(new_messages),
    )
    from ..savings import get_ledger, serialize_messages
    get_ledger().record_transform(
        bucket="context_compaction",
        layer="harness",
        mechanism=(
            f"{event_method}_fallback_{compaction_fallback}"
            if compaction_fallback
            else event_method
        ),
        before=serialize_messages(messages),
        after=serialize_messages(new_messages),
        surface="context_render",
        ctx={
            "encoding": "message_list_json_utf8_v1",
            "tokens_before": est_pt,
            "tokens_after": final_count,
            "message_count_before": len(messages),
            "message_count_after": len(new_messages),
            "first_kept_turn": first_kept_turn,
            "hook": hook_reference,
            "hook_outcome": hook_outcome,
        },
    )
    # Persist compaction into the context manager so subsequent turns
    # extend the compacted base instead of re-rendering the original
    # 100+-message append log. Without this, compaction is a one-off:
    # the very next turn's get_messages() returns the un-compacted log
    # again and the prompt snaps back to its original size, blowing
    # past the fail-safe gate. Routed through the public ABC method
    # ContextManager.replace_all_messages() instead of the prior
    # attribute-poke pattern (`ctx._all_messages = …` wrapped in
    # try/except AttributeError), which reached past the abstract
    # interface and silently no-op'd when private names changed.
    ctx = getattr(session, "context", None)
    if ctx is not None:
        ctx.replace_all_messages(new_messages)
    if (
        requested_method == "checkpoint"
        or hook is not None
        or protected_archive_tracking
    ):
        session._checkpoint_visible_message_count = len(new_messages)
    # Invalidate the output-dedup cache: its entries reference prior
    # turn numbers whose tool-result content has just been folded into
    # the digest. Leaving the cache intact would cause subsequent
    # identical reads to return back-references like "[harness:
    # identical to turn N's read output for X]" pointing at content
    # that is no longer in the prompt — leaving the model blind to
    # files it thinks it has cached. Clear the cache so compacted output
    # cannot refer to stale file contents.
    if hasattr(session, "_output_dedup_cache"):
        session._output_dedup_cache.clear()
    compaction_turn = int(getattr(session, "_compaction_turn", 0) or 0)
    compaction_turns = getattr(session, "_compaction_turns", None)
    if compaction_turns is None:
        compaction_turns = []
        session._compaction_turns = compaction_turns
    compaction_turns.append(compaction_turn)
    session._emit(
        "compaction",
        session_number=getattr(session, "_session_number", 0),
        turn_number=compaction_turn,
        tokens_before=est_pt,
        tokens_after=final_count,
        first_kept_turn=first_kept_turn,
        method=event_method,
        fallback=compaction_fallback,
        hook=hook_reference,
        hook_outcome=hook_outcome,
        **compaction_role_fields,
    )
    if configured_method == "checkpoint" and event_method == "checkpoint":
        from .checkpoint_summary import loop_guard_forces_digest

        if loop_guard_forces_digest(
            compaction_turns,
            keep_recent_turns=int(cfg.digest_keep_recent_turns),
        ):
            session._compaction_method_override = "digest"
            log.warning(
                "checkpoint loop guard activated after compactions at turns %s; "
                "using digest for the rest of this session",
                compaction_turns[-2:],
            )
    return new_messages
