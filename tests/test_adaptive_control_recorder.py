"""Tests for the prefix slot recorder.

Projects session._trace_events into prefix-only slot facts.
- min fields present for filled slots
- appending future slots does not change the result for prefix k
- only tool_call events are projected; mutation/error/test/submit facts derived
"""
from __future__ import annotations

import _ac_bootstrap  # noqa: F401  (stub-parent bootstrap; must precede harness import)
from llm_solver.harness.adaptive_control import slot_recorder as sr

MIN_FIELDS = {
    "slot_idx", "slot_presence", "slot_state", "op_kind", "obs_state",
    "contact_state", "source_mutation", "test_like_action",
    "test_execution_action", "submit_like_action", "repeat_signature",
    "evidence_refs",
}


def _ev(turn, tool, args="", result="ok", gate_blocked=False, **meta):
    ev = {"event": "tool_call", "turn_number": turn, "tool_name": tool,
          "args_summary": args, "result_summary": result, "gate_blocked": gate_blocked}
    ev.update(meta)  # write_like / source_write_like / source_write_paths
    return ev


def test_bash_source_write_recognized_via_metadata():
    # tool name is bash (not a MUTATION_TOOL) but the emitted metadata says it
    # wrote source -> must be a mutation, with target_ref from source_write_paths.
    rows = sr.recent_prefix_slots_from_events([
        _ev(1, "bash", "cat > pkg/mod.py <<EOF", source_write_like=True,
            write_like=True, source_write_paths=["pkg/mod.py"]),
    ], 1)
    r = rows[0]
    assert r["source_mutation"] == "true"
    assert r["contact_state"] == "source_write"
    assert r["slot_state"] == "edit"
    assert "pkg/mod.py" in r["target_ref"]


def test_failed_source_write_stays_failed_after_reminder_decoration():
    rows = sr.recent_prefix_slots_from_events([
        _ev(
            1,
            "bash",
            "sed -i s/a/b/ pkg/mod.py",
            "[exit code: 1]\n<system-reminder>Choose another action.</system-reminder>",
            source_write_like=True,
            write_like=True,
            source_write_paths=["pkg/mod.py"],
            exit_status=1,
            pass_fail="fail",
            outcome="error",
        ),
    ], 1)
    row = rows[0]
    assert row["slot_state"] == "tool_error"
    assert row["source_mutation"] == "false"
    assert row["effective_source_mutation"] == "false"


def test_stale_bash_domain_write_metadata_is_reprojected():
    rows = sr.recent_prefix_slots_from_events([
        _ev(
            1,
            "bash",
            "python -c \"tbl.write(sys.stdout, format='ascii.rst')\"",
            source_write_like=True,
            write_like=True,
            source_write_paths=["ascii.rst"],
        ),
    ], 1)
    r = rows[0]
    assert r["source_mutation"] == "false"
    assert r["contact_state"] == ""
    assert r["slot_state"] == "run"
    assert r["target_ref"] == ""


def test_tool_name_fallback_when_no_metadata():
    # an edit tool with no action-metadata still counts as a mutation (fallback)
    rows = sr.recent_prefix_slots_from_events([_ev(1, "edit", "x.py")], 1)
    assert rows[0]["source_mutation"] == "true"


def test_min_fields_present():
    slots = sr.recent_prefix_slots_from_events([_ev(1, "bash", "ls")], 1)
    assert len(slots) == 1
    assert MIN_FIELDS <= set(slots[0])
    assert slots[0]["slot_presence"] == "filled"
    assert slots[0]["slot_idx"] == 1


def test_prefix_only_is_stable_under_future_append():
    events = [_ev(1, "read"), _ev(2, "bash", "pytest tests/")]
    before = sr.recent_prefix_slots_from_events(events, 2)
    events.append(_ev(3, "edit", "patch"))  # hidden future
    after = sr.recent_prefix_slots_from_events(events, 2)
    assert before == after  # prefix k=2 unchanged by a later turn
    assert len(before) == 2


def test_fact_derivation():
    rows = sr.recent_prefix_slots_from_events([
        _ev(1, "edit", "x.py", "ok"),            # mutation
        _ev(2, "bash", "pytest -q", "2 passed"), # test-like run
        _ev(3, "bash", "ls", "ERROR: boom"),     # tool error
        _ev(4, "done", "submit"),                # submit
    ], 9)
    by_idx = {r["slot_idx"]: r for r in rows}
    assert by_idx[1]["source_mutation"] == "true" and by_idx[1]["slot_state"] == "edit"
    assert by_idx[2]["test_like_action"] == "true"
    assert by_idx[2]["test_execution_action"] == "true"
    assert by_idx[3]["obs_state"] == "tool_error" and by_idx[3]["slot_state"] == "tool_error"
    assert by_idx[4]["submit_like_action"] == "true"


def test_environment_preamble_does_not_make_test_file_read_an_execution():
    from llm_solver.harness.adaptive_control import watch

    rows = sr.recent_prefix_slots_from_events([
        _ev(
            33,
            "bash",
            "cmd=\"source /opt/env/bin/activate; cd /work && "
            "sed -n '303,370p' tests/test_regression.py\"",
            "def test_regression(): pass",
            exit_status=0,
            pass_fail="pass",
            outcome="ok",
        ),
    ], 33)

    assert rows[0]["test_like_action"] == "true"  # legacy broad fact
    assert rows[0]["test_execution_action"] == "false"
    assert watch.material_progress(rows[0]) is False


def test_environment_preamble_preserves_real_test_execution_progress():
    from llm_solver.harness.adaptive_control import watch

    rows = sr.recent_prefix_slots_from_events([
        _ev(
            34,
            "bash",
            "cmd='source /opt/env/bin/activate; cd /work && "
            "python -m pytest tests/test_regression.py -q'",
            "2 passed in 0.10s",
            exit_status=0,
            pass_fail="pass",
            outcome="ok",
        ),
    ], 34)

    assert rows[0]["test_execution_action"] == "true"
    assert watch.material_progress(rows[0]) is True


def test_non_tool_events_ignored():
    events = [{"event": "session_start", "turn_number": 0}, _ev(1, "bash", "ls")]
    assert len(sr.recent_prefix_slots_from_events(events, 5)) == 1


def test_repeat_signature_identical_for_identical_actions():
    rows = sr.recent_prefix_slots_from_events([_ev(1, "bash", "ls"), _ev(2, "bash", "ls")], 2)
    assert rows[0]["repeat_signature"] == rows[1]["repeat_signature"]


if __name__ == "__main__":
    test_min_fields_present()
    test_prefix_only_is_stable_under_future_append()
    test_fact_derivation()
    test_non_tool_events_ignored()
    test_repeat_signature_identical_for_identical_actions()
    print("Recorder tests passed")
