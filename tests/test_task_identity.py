"""Identity follows selected records, never workspace spelling or task text."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from _config_helpers import make_config
from llm_solver.harness.loop import Session, SessionResult, TaskSpec, solve_task
from llm_solver.harness.task_identity import resolve_task_identity


def start(identity, number=1):
    return {"event": "session_start", "session_number": number,
            "task_identity": identity.record(), "instance_id": identity.instance_id,
            "attempt_id": identity.attempt_id(number)}


def test_shared_identity_survives_relocation_and_duplicate_labels(tmp_path):
    cfg = make_config()
    identity = resolve_task_identity(cfg)
    for name in ["first/project", "second/renamed"]:
        (tmp_path / name).mkdir(parents=True)
    sessions = [Session(cfg, MagicMock(), "sys", "task", str(tmp_path / name),
                        task_identity=identity, session_number=n)
                for n, name in enumerate(["first/project", "second/renamed"], 1)]
    assert {s.instance_id for s in sessions} == {identity.instance_id}
    assert len({s.attempt_id for s in sessions}) == 2
    assert all(s.task_identity is identity for s in sessions)


@pytest.mark.parametrize("number", [0, 2])
def test_explicit_trace_resume_retains_task_but_allocates_new_invocation(number):
    cfg = make_config(adaptive_control_source_instance_id="declared-task")
    original = resolve_task_identity(cfg)
    resumed = resolve_task_identity(make_config(), prior_events=[start(original, number)])
    assert resumed.task_id == original.task_id
    assert resumed.instance_id == original.instance_id == "declared-task"
    assert resumed.invocation_id != original.invocation_id
    assert resumed.parent_invocation_id == original.invocation_id
    assert resumed.linkage == "selected_trace"


@pytest.mark.parametrize("events", [[], [{"event": "session_start", "session_number": 1}]])
def test_old_or_empty_selected_trace_has_explicit_unknown_linkage(events):
    identity = resolve_task_identity(make_config(), prior_events=events)
    assert identity.linkage == "legacy_trace_unknown"
    assert identity.instance_id.startswith("local-task:")
    assert identity.parent_invocation_id == ""


def test_transcript_resume_has_explicit_unknown_linkage():
    identity = resolve_task_identity(make_config(), transcript_resume=True)
    assert identity.linkage == "transcript_unknown"
    assert not identity.parent_invocation_id


def test_newer_legacy_session_does_not_borrow_older_native_identity():
    original = resolve_task_identity(make_config())
    resumed = resolve_task_identity(make_config(), prior_events=[start(original),
        {"event": "session_start", "session_number": 2}])
    assert resumed.linkage == "legacy_trace_unknown"
    assert resumed.task_id != original.task_id


def test_conflicting_declared_resume_identity_is_rejected():
    original = resolve_task_identity(make_config(adaptive_control_source_instance_id="a"))
    with pytest.raises(ValueError, match="conflicts"):
        resolve_task_identity(make_config(adaptive_control_source_instance_id="b"),
                              prior_events=[start(original)])


@pytest.mark.parametrize("field,value", [("version", True), ("version", 2),
    ("task_id", "directory-name"), ("invocation_id", ""),
    ("parent_invocation_id", None), ("declared_instance_id", 7),
    ("linkage", "guessed"), ("linkage", "selected_trace")])
def test_malformed_native_identity_is_not_silently_replaced(field, value):
    record = start(resolve_task_identity(make_config()))
    record["task_identity"][field] = value
    with pytest.raises(ValueError, match="identity"):
        resolve_task_identity(make_config(), prior_events=[record])


@pytest.mark.parametrize("field", ["task_identity", "instance_id", "attempt_id", "lineage"])
def test_inconsistent_or_incomplete_native_record_is_rejected(field):
    record = start(resolve_task_identity(make_config()))
    if field == "task_identity":
        del record[field]["declared_instance_id"]
    elif field == "lineage":
        record["task_identity"]["parent_invocation_id"] = record["task_identity"]["invocation_id"]
        record["task_identity"]["linkage"] = "selected_trace"
    else:
        record[field] = "different"
    with pytest.raises(ValueError, match="identity"):
        resolve_task_identity(make_config(), prior_events=[record])


def test_session_cannot_relabel_shared_identity(tmp_path):
    identity = resolve_task_identity(make_config(adaptive_control_source_instance_id="a"))
    with pytest.raises(ValueError, match="conflicts"):
        Session(make_config(adaptive_control_source_instance_id="b"), MagicMock(),
                "sys", "task", str(tmp_path), task_identity=identity)


@pytest.mark.parametrize("tail", ["", '{"event":'])
def test_driver_persists_shared_identity_and_resumes_a_relocated_copy(tmp_path, tail):
    from llm_solver.harness._loop._driver_setup import resolve_run_paths

    task = tmp_path / "first" / "project"
    task.mkdir(parents=True)
    artifacts = tmp_path / "records"
    seen = []

    def no_model(session):
        seen.append(session)
        return SessionResult(0, "max_turns", False)

    cfg = make_config(max_sessions=2, auto_commit=False)
    client = MagicMock()
    with (patch.object(Session, "run", no_model),
          patch("llm_solver.harness.loop._auto_commit"),
          patch("llm_solver.harness._loop.driver.write_run_metrics")):
        solve_task(task, cfg, client, task_spec=TaskSpec("task"), artifacts_dir=artifacts)
        source_trace = seen[0]._trace_path
        events = [json.loads(line) for line in source_trace.read_text().splitlines()]
        starts = [event for event in events if event["event"] == "session_start"]
        assert len(starts) == 2
        assert seen[0].task_identity is seen[1].task_identity
        for event, session in zip(starts, seen):
            assert event["task_identity"] == session.task_identity.record()
            assert event["instance_id"] == session.instance_id
            assert event["attempt_id"] == session.attempt_id

        relocated = tmp_path / "second" / "renamed"
        relocated.mkdir(parents=True)
        copied = tmp_path / "selected-copy"
        target = resolve_run_paths(relocated, copied)[2]
        source_text = source_trace.read_text()
        if tail:
            # The existing recovery policy repairs nonterminal sessions.
            lines = source_text.splitlines()
            last_start = max(i for i, line in enumerate(lines)
                             if json.loads(line)["event"] == "session_start")
            source_text = "\n".join(lines[:last_start + 1]) + "\n"
        target.write_text(source_text + tail)
        solve_task(relocated, cfg, client, task_spec=TaskSpec("task"),
                   artifacts_dir=copied, resume_from_artifacts=True)
        assert seen[2].instance_id == seen[0].instance_id
        assert seen[2].task_identity.parent_invocation_id == seen[0].task_identity.invocation_id
        assert seen[2].task_identity is seen[3].task_identity
        assert len({session.attempt_id for session in seen}) == 4
        assert seen[2]._session_number == 3

        # Reusing a directory without requesting resume is a fresh invocation.
        solve_task(task, cfg, client, task_spec=TaskSpec("task"), artifacts_dir=artifacts)
        assert seen[4].task_identity.task_id != seen[0].task_identity.task_id
    client.chat.assert_not_called()


def test_branch_capture_distinguishes_attempts_with_same_declared_source(tmp_path):
    from llm_solver.harness.adaptive_control import branch_bundle

    task = tmp_path / "task"
    task.mkdir()
    (task / "source.py").write_text("value = 1\n")
    baseline = tmp_path / "baseline.toml"
    baseline.write_text("[loop]\nmax_turns = 2\n")
    cfg = make_config(
        adaptive_control_source_instance_id="declared-task",
        adaptive_control_branch_bundle_enabled=True,
        adaptive_control_branch_bundle_root=str(tmp_path / "bundles"),
        adaptive_control_branch_bundle_source_run_id="declared-run",
        adaptive_control_baseline_config_paths=(str(baseline),),
    )
    decision = SimpleNamespace(diagnosis_status="active_confirmed", active_hurdle_mode="fixture",
        detector_id="fixture", detector_status="active_confirmed", basis_refs=["T1"])
    sessions = [Session(cfg, MagicMock(), "system", "task", str(task)) for _ in range(2)]
    with patch.object(branch_bundle, "_git_commit", return_value="fixture"):
        rows = [branch_bundle.maybe_capture(session, decision, 1, "fixture") for session in sessions]
        repeat = branch_bundle.maybe_capture(sessions[0], decision, 1, "fixture")
    assert all(row["status"] == "created" and not row["reason"] for row in rows)
    assert rows[0]["branch_point_id"] != rows[1]["branch_point_id"]
    assert repeat["status"] == "exists"
    sessions[0].attempt_id = ""
    with patch.object(branch_bundle, "_copy_repo_snapshot") as snapshot:
        missing = branch_bundle.maybe_capture(sessions[0], decision, 1, "fixture")
    assert missing["reason"] == "task_or_attempt_identity_missing"
    snapshot.assert_not_called()
    sessions[0].attempt_id = sessions[0].task_identity.attempt_id(sessions[0]._session_number)
    for session, row in zip(sessions, rows):
        manifest = json.loads((Path(row["path"]) / "branch_manifest.json").read_text())
        assert manifest["instance_id"] == session.instance_id == "declared-task"
        assert manifest["attempt_id"] == session.attempt_id
        assert manifest["task_identity"] == session.task_identity.record()
        assert manifest["branch_identity_version"] == "attempt_v2"


def test_historical_branch_id_calculation_is_still_available():
    import hashlib
    from llm_solver.harness.adaptive_control.branch_bundle import branch_point_id

    assert branch_point_id("run", "task", 1, "signal", "detector", "policy") == (
        hashlib.sha256(b"run|task|1|signal|detector|policy").hexdigest()[:24])


def test_native_session_identity_is_used_by_control_ledger(tmp_path):
    from llm_solver.harness.adaptive_control.llm_detector_apply import _append_detector_control_ledger

    path = tmp_path / "ledger.jsonl"
    cfg = make_config(adaptive_control_ledger_path=str(path))
    identity = resolve_task_identity(cfg)
    sessions = [Session(cfg, MagicMock(), "sys", "task", str(tmp_path),
                        task_identity=identity, session_number=number) for number in (1, 2)]
    for session in sessions:
        _append_detector_control_ledger(session=session, turn=0,
            verdict=SimpleNamespace(hurdle_family="fixture", evidence_refs=[]),
            controller_version="fixture", chosen=None, apply_status="blocked",
            blocked_reason="fixture", result=None)
    records = [json.loads(line) for line in path.read_text().splitlines()]
    for record, session in zip(records, sessions):
        assert record["instance_id"] == session.instance_id == identity.instance_id
        assert record["attempt_id"] == session.attempt_id
    assert len(records) == 2
    assert records[0]["attempt_id"] != records[1]["attempt_id"]
