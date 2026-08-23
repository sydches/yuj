"""Server context size lookup + validated compaction at the OOM-safe threshold."""
from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..loop import Session

log = logging.getLogger(__name__)


class CompactionOverflowError(Exception):
    """Raised when compaction cannot fit the prompt within budget even
    after truncating tool messages in the most recent assistant+tool
    pair. Caller (chat_io) treats this as terminal — better to end the
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


def preflight_reclip_oversized(session) -> dict | None:
    """Invariant backstop: re-clip the single largest oversized message
    in token space so a pre-flight overflow gets one chance to recover
    before the session ends context_full.

    The pre-flight gate fires when the projected prompt exceeds
    cfg.context_fill_ratio × context_size — in practice almost always
    because one tool result appended at the tail of the previous turn
    is enormous (single-turn-overflow death). Instead of ending the
    session outright, find the largest clippable message, and — if it
    alone exceeds half the context window — head+tail clip it to fit
    within (context_size / 2) tokens, with a visible notice inserted
    where content was removed (original token size, what is shown,
    advice to re-run a narrower command). The caller re-projects once
    and only ends the session if the projection STILL exceeds the
    window.

    Clippable messages: tool results and user messages AFTER the first
    (the initial task prompt is never clipped). System and assistant
    messages stay intact — assistant messages carry tool_calls
    structure the server round-trips.

    Token counts use the bound local tokenizer when available (exact),
    chars/4 otherwise — the same accounting the pre-flight gate itself
    uses. Persisting goes through ContextManager.replace_all_messages()
    so strategy caches invalidate; strategies that cannot replace opt
    out and the caller falls through to the legacy session end.

    Returns an info dict {index, role, tool_call_id, orig_pt, new_pt}
    on success, None when nothing qualifies.
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

    budget_pt = ctx_size // 2
    first_user_seen = False
    best_i, best_pt = -1, 0
    for i, m in enumerate(msgs):
        role = m.get("role")
        if role == "user" and not first_user_seen:
            first_user_seen = True  # initial task prompt — never clipped
            continue
        if role not in ("tool", "user"):
            continue
        content = m.get("content")
        if not isinstance(content, str) or not content:
            continue
        pt = _count(m)
        if pt > best_pt:
            best_i, best_pt = i, pt
    if best_i < 0 or best_pt <= budget_pt:
        return None  # no single offending oversized message
    target = msgs[best_i]
    content = target["content"]
    notice = (
        f"[HARNESS re-clip: this message was ~{best_pt} tokens "
        f"({len(content)} chars) — too large for the {ctx_size}-token "
        f"context window. Only the head and tail are shown "
        f"(~{budget_pt}-token budget); the middle was removed and is not "
        "recoverable from context. Re-run a narrower command (file "
        "subset, grep filter, --max-count, head/tail) to see the "
        "removed part.]"
    )
    # Convert the token budget to chars via this message's own observed
    # chars-per-token ratio; reserve the notice and a 10% safety margin
    # so the clipped message lands under budget_pt after re-count.
    chars_per_token = max(1.0, len(content) / best_pt)
    char_budget = max(0, int(budget_pt * chars_per_token * 0.9) - len(notice) - 2)
    clipped = _head_tail_truncate(content, char_budget, marker=notice)
    new_target = dict(target)
    new_target["content"] = clipped
    new_msgs = list(msgs)
    new_msgs[best_i] = new_target
    if not ctx.replace_all_messages(new_msgs):
        return None  # strategy cannot persist a replacement
    new_pt = _count(new_target)
    log.warning(
        "preflight re-clip: message %d (role=%s) %d -> %d tokens "
        "(budget=%d, ctx=%d)",
        best_i, target.get("role"), best_pt, new_pt, budget_pt, ctx_size,
    )
    return {
        "index": best_i,
        "role": target.get("role", ""),
        "tool_call_id": target.get("tool_call_id", ""),
        "orig_pt": best_pt,
        "new_pt": new_pt,
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


def maybe_compact_messages(session: "Session", messages: list[dict]) -> list[dict]:
    """Digest or validated-checkpoint compaction at the OOM-safe threshold.

    Threshold = (1 - max_tokens_fraction) - digest_compaction_safety_margin.
    This guarantees that any turn that does NOT fire compaction
    leaves room for the server to allocate max_tokens generation
    slots without exceeding ctx. Fires every time the threshold
    is crossed.

    Triggers when the exact pre-flight token count (local
    tokenizer) crosses threshold × server_ctx AND the session
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
    requested_method = str(
        getattr(session, "_compaction_method_override", "") or configured_method
    )
    if requested_method == "checkpoint":
        _sync_checkpoint_archive(session, messages)
    # Route through the bound method so test mocks patching
    # `Session._get_server_ctx` continue to intercept this call site.
    ctx_size = session._get_server_ctx() or int(getattr(cfg, "context_size", 0) or 0)
    if ctx_size <= 0:
        return messages
    max_tokens_fraction = float(getattr(cfg, "max_tokens_fraction", 0.25))
    safety_margin = float(getattr(cfg, "digest_compaction_safety_margin", 0.05))
    threshold = max(0.0, 1.0 - max_tokens_fraction - safety_margin)
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
    if cheap_est is not None and cheap_est * 1.10 < budget:
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
    tool_schemas = getattr(session, "_tool_schemas", None)
    if tokenizer is not None:
        try:
            est_pt = int(tokenizer.count(messages, tools=tool_schemas))
        except Exception as e:
            log.warning("local tokenizer count failed (%s); skipping compaction check", e)
            return messages
    elif cheap_est is not None:
        est_pt = cheap_est
    else:
        return messages
    if est_pt < budget:
        return messages
    mutation_count = sum(1 for ev in session._trace_events
                         if ev.get("event") == "tool_call"
                         and str(ev.get("tool_name", "")) in ("write", "edit", "str_replace", "create"))
    if mutation_count < gate_min_mut:
        return messages
    if session._trace_path is None or not session._trace_path.is_file():
        return messages
    try:
        # Import the content-blind rendering core from _shared.
        from ..._shared.digest_core import _load as digest_load, render_digest, DigestOptions
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

    compaction_fallback = ""
    first_kept_turn = _latest_assistant_turn(
        messages, getattr(session, "_compaction_turn", 0)
    )
    latest_pair: list[dict] = []
    new_messages: list[dict] | None = None

    if requested_method == "checkpoint":
        from .checkpoint_summary import generate_checkpoint

        keep_recent_tokens = int(
            getattr(cfg, "checkpoint_keep_recent_tokens", 0) or 0
        ) or max(4096, int(0.20 * ctx_size))

        def _call_checkpoint(payload: dict) -> str:
            side_result = session.client.complete_side_request(payload)
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
    # latest_pair was appended verbatim. When it carries 3-5 parallel
    # read() results each ≥10k chars, the post-build prompt can still
    # exceed budget. Recount; if over, truncate tool messages within
    # latest_pair (preserving the assistant message intact) and
    # recount once more. If still over, raise a typed error so the
    # session ends with a debuggable reason instead of a server 400.
    final_count = _recount_tokens(new_messages, tokenizer, tools=tool_schemas)
    if final_count > 0 and final_count > budget:
        # tool messages in new_messages all came from latest_pair —
        # the digest user message has role "user", initial user has
        # role "user", system messages have role "system".
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
                "compaction overflow guard: latest_pair exceeded budget by ~%d tokens; "
                "truncated %d tool message(s) to ~%d chars each",
                over_tokens, len(tool_indices), per_msg_budget,
            )
            final_count = _recount_tokens(new_messages, tokenizer, tools=tool_schemas)
        if final_count > 0 and final_count > budget:
            raise CompactionOverflowError(
                f"compaction cannot fit prompt within budget after truncation: "
                f"final_count={final_count} > budget={budget} "
                f"(ctx={ctx_size}, threshold={threshold:.2f}, "
                f"latest_pair_msgs={len(latest_pair)})"
            )

    session._compacted = True
    session._compaction_count = int(getattr(session, "_compaction_count", 0)) + 1
    log.info(
        "compaction fired (#%d) at turn %d: threshold=%.2f, est_pt=%d, ctx=%d, "
        "mutation_count=%d, old_msg_count=%d -> new=%d",
        session._compaction_count,
        getattr(session, "_compaction_turn", -1),
        threshold, est_pt, ctx_size, mutation_count,
        len(messages), len(new_messages),
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
    if requested_method == "checkpoint":
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
        method=requested_method,
        fallback=compaction_fallback,
        role="main",
    )
    if configured_method == "checkpoint":
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
