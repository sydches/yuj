"""Session-level digest compaction at the derived OOM-safe threshold.

threshold = (1 - max_tokens_fraction) - digest_compaction_safety_margin

Re-fires on every threshold crossing. Mutation gate via
cfg.digest_compaction_gate_min_mutations (default 0).
"""
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

from scripts.llm_solver.harness.loop import Session


def _make_fake_context() -> SimpleNamespace:
    """Stand-in for a ContextManager exposing the public method the
    harness calls after compacting (replace_all_messages). Mutates its
    own _all_messages list so tests can introspect the compacted state.

    The fake implements `ContextManager.replace_all_messages` and keeps
    its message cache in sync.
    """
    fake = SimpleNamespace(_all_messages=[], _msg_cache=None, _tok_cache=None)

    def _replace(new_messages: list[dict]) -> bool:
        fake._all_messages = list(new_messages)
        fake._msg_cache = None
        fake._tok_cache = None
        return True

    fake.replace_all_messages = _replace
    return fake


def _make_session(
    cwd: Path,
    trace_events: list[dict],
    *,
    server_ctx_value: int = 98304,
    cfg_extra: dict | None = None,
) -> Session:
    """Build a stub Session with just enough state for the compaction path."""
    sess = Session.__new__(Session)
    cfg_kwargs = {
        "base_url": "http://localhost:8080/v1",
        "context_size": 40960,
        "max_tokens_fraction": 0.25,
        "digest_compaction_safety_margin": 0.05,
        "digest_compaction_gate_min_mutations": 0,
    }
    cfg_kwargs.update(cfg_extra or {})
    sess.cfg = SimpleNamespace(**cfg_kwargs)
    sess.client = SimpleNamespace()
    sess._trace_path = cwd / ".trace.jsonl"
    sess._trace_events = trace_events
    sess._compacted = False
    sess._compaction_turn = 0
    sess._server_ctx_cache = server_ctx_value
    return sess


def _seed_trace_jsonl(cwd: Path, n_turns: int) -> None:
    rows = []
    for i in range(n_turns):
        rows.append(
            {
                "event": "tool_call",
                "turn": i,
                "name": "read",
                "args_summary": f"path='./f{i}.py'",
                "result_summary": f"line content for turn {i}\n" * 20,
                "reasoning_summary": "",
                "prompt_tokens": 1000 + i * 200,
                "completion_tokens": 50,
            }
        )
    (cwd / ".trace.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def _heavy_messages(n_pairs: int, payload_chars: int = 2000) -> list[dict]:
    msgs = [{"role": "system", "content": "sys prompt"}]
    msgs.append({"role": "user", "content": "task: do thing"})
    for i in range(n_pairs):
        msgs.append({"role": "assistant", "content": ""})
        msgs.append({"role": "tool", "content": "x" * payload_chars})
    return msgs


def test_compaction_skipped_below_threshold(tmp_path):
    _seed_trace_jsonl(tmp_path, 5)
    sess = _make_session(tmp_path, trace_events=[
        {"event": "tool_call", "tool_name": "write"},
    ])
    msgs = _heavy_messages(2, payload_chars=500)
    out = sess._maybe_compact_messages(msgs)
    assert out is msgs
    assert sess._compacted is False


def test_compaction_skipped_when_no_mutation_and_gate_set(tmp_path):
    """When gate_min_mutations is set (>=1), zero mutations blocks
    compaction. Default (gate=0) does NOT block — see
    test_compaction_mutation_gate_is_configurable for the open case."""
    _seed_trace_jsonl(tmp_path, 30)
    sess = _make_session(tmp_path, trace_events=[
        {"event": "tool_call", "tool_name": "read"},
        {"event": "tool_call", "tool_name": "glob"},
    ], cfg_extra={"digest_compaction_gate_min_mutations": 1})
    msgs = _heavy_messages(60, payload_chars=2000)
    out = sess._maybe_compact_messages(msgs)
    assert out is msgs
    assert sess._compacted is False


def test_compaction_fires_above_threshold_with_mutation(tmp_path):
    _seed_trace_jsonl(tmp_path, 30)
    sess = _make_session(tmp_path, trace_events=[
        {"event": "tool_call", "tool_name": "read"},
        {"event": "tool_call", "tool_name": "write"},
        {"event": "tool_call", "tool_name": "edit"},
    ])
    sess.context = _make_fake_context()
    # Defaults: max_tokens_fraction=0.25, safety_margin=0.05 → threshold=0.70.
    # 0.70 * 98304 = 68812 tokens; chars_div_4 uses str(msg) framing.
    # 100 pairs × 5500-char payloads → est_pt over budget.
    msgs = _heavy_messages(100, payload_chars=5500)
    out = sess._maybe_compact_messages(msgs)
    assert out is not msgs
    assert sess._compacted is True
    # Post-compaction shape: system + initial_user + digest_block + latest
    # assistant + latest tool result (verbatim — full-fidelity for one turn).
    assert len(out) == 5
    roles = [m["role"] for m in out]
    assert roles == ["system", "user", "user", "assistant", "tool"]
    assert out[0]["content"] == "sys prompt"
    assert out[1]["content"] == "task: do thing"
    assert "Compacted history" in out[2]["content"]
    # Derived threshold reflected in the compaction message text.
    assert "0.70 of the server context window" in out[2]["content"]
    # The latest assistant + tool messages from the input survive verbatim.
    assert out[3] is msgs[-2]
    assert out[4] is msgs[-1]
    assert sess.context._all_messages == out
    assert sess.context._msg_cache is None
    assert sess.context._tok_cache is None


def test_compaction_refires_each_threshold_crossing(tmp_path):
    _seed_trace_jsonl(tmp_path, 30)
    sess = _make_session(tmp_path, trace_events=[
        {"event": "tool_call", "tool_name": "write"},
    ])
    sess.context = _make_fake_context()
    msgs = _heavy_messages(100, payload_chars=5500)
    out1 = sess._maybe_compact_messages(msgs)
    assert sess._compacted is True
    assert sess._compaction_count == 1
    msgs2 = _heavy_messages(120, payload_chars=5500)
    out2 = sess._maybe_compact_messages(msgs2)
    assert out2 is not msgs2
    assert sess._compaction_count == 2
    # system + initial_user + digest + latest_assistant + latest_tool = 5
    assert len(out2) == 5


def test_threshold_derivation_from_max_tokens_fraction(tmp_path):
    """Reducing max_tokens_fraction raises the derived threshold;
    same payload that fired at 0.70 keeps firing at 0.85 only with
    a larger payload that crosses the higher budget."""
    _seed_trace_jsonl(tmp_path, 30)
    # max_tokens_fraction=0.10 + safety=0.05 → threshold=0.85.
    # 0.85 * 98304 = 83558 tokens.
    sess = _make_session(tmp_path, trace_events=[
        {"event": "tool_call", "tool_name": "write"},
    ], cfg_extra={"max_tokens_fraction": 0.10})
    sess.context = _make_fake_context()
    # 80 pairs × 4000 chars ≈ 320k chars / 4 ≈ 80k est_pt — under 0.85.
    msgs_under = _heavy_messages(80, payload_chars=4000)
    out_under = sess._maybe_compact_messages(msgs_under)
    assert sess._compacted is False
    assert out_under is msgs_under
    # 100 pairs × 5500 chars ≈ 550k / 4 ≈ 137k est_pt — over 0.85.
    msgs_over = _heavy_messages(100, payload_chars=5500)
    out_over = sess._maybe_compact_messages(msgs_over)
    assert sess._compacted is True
    assert "0.85 of the server context window" in out_over[2]["content"]


def test_threshold_derivation_from_safety_margin(tmp_path):
    """Larger safety margin lowers the derived threshold; payload
    that did NOT fire at 0.70 (margin=0.05) fires at 0.50 (margin=0.25)."""
    _seed_trace_jsonl(tmp_path, 30)
    msgs = _heavy_messages(60, payload_chars=4000)  # ~240k chars / 4 ≈ 60k est_pt

    sess_low_margin = _make_session(tmp_path, trace_events=[
        {"event": "tool_call", "tool_name": "write"},
    ])  # threshold=0.70 → 68812 budget; payload undershoots.
    sess_low_margin.context = _make_fake_context()
    out_low = sess_low_margin._maybe_compact_messages(msgs)
    assert sess_low_margin._compacted is False
    assert out_low is msgs

    sess_high_margin = _make_session(tmp_path, trace_events=[
        {"event": "tool_call", "tool_name": "write"},
    ], cfg_extra={"digest_compaction_safety_margin": 0.25})  # threshold=0.50 → 49152 budget.
    sess_high_margin.context = _make_fake_context()
    out_high = sess_high_margin._maybe_compact_messages(msgs)
    assert sess_high_margin._compacted is True
    assert "0.50 of the server context window" in out_high[2]["content"]


def test_compaction_mutation_gate_is_configurable(tmp_path):
    """Default min=0 fires regardless of mutations. Setting min=1
    blocks compaction when no mutation has occurred."""
    _seed_trace_jsonl(tmp_path, 30)
    msgs = _heavy_messages(100, payload_chars=5500)

    # Default min=0, no mutations → fires.
    sess_open = _make_session(tmp_path, trace_events=[
        {"event": "tool_call", "tool_name": "read"},
    ])
    sess_open.context = _make_fake_context()
    out_open = sess_open._maybe_compact_messages(msgs)
    assert sess_open._compacted is True

    # min=1, no mutations → blocked.
    sess_blocked = _make_session(tmp_path, trace_events=[
        {"event": "tool_call", "tool_name": "read"},
    ], cfg_extra={"digest_compaction_gate_min_mutations": 1})
    out_blocked = sess_blocked._maybe_compact_messages(msgs)
    assert out_blocked is msgs
    assert sess_blocked._compacted is False


def test_latest_pair_truncated_when_alone_exceeds_budget(tmp_path):
    """Regression: when latest_pair (most recent assistant + tool) alone
    busts the post-compaction budget, the overflow guard truncates the
    tool message(s) within latest_pair so the returned prompt fits.

    Without this guard, _maybe_compact_messages can return a prompt that
    exceeds the server context size.
    """
    from scripts.llm_solver.harness._loop.compaction import _recount_tokens
    _seed_trace_jsonl(tmp_path, 30)
    sess = _make_session(tmp_path, trace_events=[
        {"event": "tool_call", "tool_name": "write"},
    ])
    sess.context = _make_fake_context()
    # Build a normal-shape compaction trigger, then replace the LAST
    # tool message with a 400 KB blob — large enough that latest_pair
    # alone is over budget after the rest of history is digested away.
    msgs = _heavy_messages(100, payload_chars=5500)
    huge_blob = "X" * 400_000
    msgs[-1] = {"role": "tool", "content": huge_blob}

    # Sanity: pre-fix this would return msgs unchanged at oversized state.
    out = sess._maybe_compact_messages(msgs)
    assert sess._compacted is True
    assert len(out) == 5
    # The huge tool content was truncated by the overflow guard.
    last_tool = out[-1]
    assert last_tool["role"] == "tool"
    assert last_tool["content"] != huge_blob
    assert "compaction overflow guard" in last_tool["content"]
    assert len(last_tool["content"]) < len(huge_blob)
    # And the post-compaction prompt now fits within budget.
    # threshold=0.70, ctx=98304 → budget=68812 tokens.
    final_count = _recount_tokens(out, tokenizer=None)
    assert final_count <= 68812, (
        f"post-guard final_count={final_count} should be <= budget=68812"
    )


def test_compaction_overflow_raises_when_truncation_insufficient(tmp_path):
    """When even truncating tool messages to zero cannot fit the prompt
    (e.g., the digest itself exceeds budget), maybe_compact_messages
    raises CompactionOverflowError so the caller (chat_io) ends the
    session with a debuggable reason instead of sending and taking a
    server 400.
    """
    from scripts.llm_solver.harness._loop.compaction import CompactionOverflowError
    # Use a TINY ctx so the digest text alone exceeds budget.
    # threshold=0.70, ctx=2048 → budget=1433 tokens ≈ 5732 chars.
    # Seed a trace with content that renders into a digest >> 5732 chars.
    rows = []
    for i in range(200):
        rows.append({
            "event": "tool_call", "turn": i, "name": "read",
            "args_summary": f"path='./long_filename_to_pad_digest_{i}.py'",
            "result_summary": ("padding " * 50),  # ~400 chars per row
            "reasoning_summary": "", "prompt_tokens": 0, "completion_tokens": 0,
        })
    (tmp_path / ".trace.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n"
    )
    sess = _make_session(tmp_path, trace_events=[
        {"event": "tool_call", "tool_name": "write"},
    ], server_ctx_value=2048)
    sess.context = _make_fake_context()
    # Force compaction to fire by providing input over budget.
    msgs = _heavy_messages(50, payload_chars=2000)
    msgs[-1] = {"role": "tool", "content": "Y" * 10_000}
    with pytest.raises(CompactionOverflowError) as exc_info:
        sess._maybe_compact_messages(msgs)
    assert "cannot fit prompt within budget" in str(exc_info.value)


def test_compaction_falls_back_to_cfg_when_server_ctx_unknown(tmp_path):
    _seed_trace_jsonl(tmp_path, 30)
    sess = _make_session(tmp_path, trace_events=[
        {"event": "tool_call", "tool_name": "write"},
    ], server_ctx_value=0)
    sess.context = _make_fake_context()
    sess.cfg.base_url = ""  # block /props
    # cfg.context_size = 40960 → 0.70 × 40960 = 28672 tokens ≈ 114688 chars
    msgs = _heavy_messages(100, payload_chars=2500)  # ~510k chars
    out = sess._maybe_compact_messages(msgs)
    assert sess._compacted is True
    assert any("Compacted history" in m["content"] for m in out)
