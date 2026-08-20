import json
from pathlib import Path
from unittest.mock import patch

from scripts.llm_assist.__main__ import main
from scripts.llm_assist.progress import TraceFollower
from scripts.llm_assist.runner import (
    approval_request_path,
    load_approval_request,
    load_approval_decisions,
    load_interrupt_marker,
    prepare_smoke_repo,
    resolve_served_model,
    resolve_smoke_model,
    save_approval_request,
    session_trace_tail,
    session_turn_tail,
)
from scripts.llm_assist.store import SessionLock, SessionLockedError, SessionStore


def test_session_store_round_trip(tmp_path: Path):
    store = SessionStore(tmp_path)

    record = store.create_session(
        cwd=tmp_path / "work",
        model="qwen3-8b",
        prompt_text="Fix the failing test",
        prompt_source="inline",
        context_mode="full",
        system_prompt_path=None,
        config_paths=[],
    )

    fetched = store.get_session(record.session_id)
    assert fetched is not None
    assert fetched.cwd == str((tmp_path / "work").resolve())
    assert fetched.status == "created"
    assert fetched.context_mode == "full"

    store.update_session(
        record.session_id,
        status="paused",
        last_finish_reason="max_turns",
    )
    updated = store.get_session(record.session_id)
    assert updated is not None
    assert updated.status == "paused"
    assert updated.last_finish_reason == "max_turns"

    sessions = store.list_sessions(limit=5)
    assert [item.session_id for item in sessions] == [record.session_id]

    assert store.get_active_session(tmp_path / "work") is None
    store.set_active_session(tmp_path / "work", record.session_id)
    active = store.get_active_session(tmp_path / "work")
    assert active is not None
    assert active.session_id == record.session_id
    assert record.session_id in store.list_active_session_ids()
    store.clear_active_session(tmp_path / "work", session_id=record.session_id)
    assert store.get_active_session(tmp_path / "work") is None

    lock = store.acquire_session_lock(record.session_id)
    assert lock.session_id == record.session_id
    assert store.get_session_lock(record.session_id) is not None
    assert record.session_id in store.list_locked_session_ids()
    store.release_session_lock(record.session_id)
    assert store.get_session_lock(record.session_id) is None


def test_session_trace_tail_formats_recent_events(tmp_path: Path):
    artifact_dir = tmp_path / "session"
    artifact_dir.mkdir(parents=True)
    trace_path = artifact_dir / ".trace.jsonl"
    events = [
        {"event": "session_start", "session_number": 1},
        {
            "event": "tool_call",
            "session_number": 1,
            "turn_number": 0,
            "tool_name": "bash",
            "args_summary": "cmd='pytest -q tests/test_app.py'",
            "result_summary": "1 passed",
        },
        {
            "event": "session_end",
            "session_number": 1,
            "finish_reason": "stop",
            "turns": 2,
        },
    ]
    trace_path.write_text("".join(json.dumps(event) + "\n" for event in events))

    lines = session_trace_tail(artifact_dir, limit=2)

    assert lines == [
        "tool_call turn=0 bash(cmd='pytest -q tests/test_app.py') => 1 passed",
        "session_end session=1 finish_reason=stop turns=2",
    ]


def test_session_turn_tail_groups_reasoning_and_tools(tmp_path: Path):
    artifact_dir = tmp_path / "session"
    artifact_dir.mkdir(parents=True)
    trace_path = artifact_dir / ".trace.jsonl"
    events = [
        {"event": "session_start", "session_number": 1},
        {
            "event": "tool_call",
            "session_number": 1,
            "turn_number": 0,
            "tool_name": "read",
            "args_summary": "path='calc.py'",
            "result_summary": "def add(a, b): return a - b",
            "reasoning": "Inspect the implementation before editing.",
            "gate_blocked": False,
        },
        {
            "event": "tool_call",
            "session_number": 1,
            "turn_number": 0,
            "tool_name": "bash",
            "args_summary": "cmd='pytest -q tests/test_calc.py'",
            "result_summary": "1 failed",
            "reasoning": "Inspect the implementation before editing.",
            "gate_blocked": False,
        },
        {
            "event": "tool_call",
            "session_number": 1,
            "turn_number": 1,
            "tool_name": "done",
            "args_summary": "",
            "result_summary": "Session ended by model.",
            "reasoning": "",
            "gate_blocked": False,
        },
    ]
    trace_path.write_text("".join(json.dumps(event) + "\n" for event in events))

    lines = session_turn_tail(artifact_dir, limit=2)

    assert lines == [
        "turn 0 (session 1)",
        "  reasoning: Inspect the implementation before editing.",
        "  tool: read(path='calc.py')",
        "    result: def add(a, b): return a - b",
        "  tool: bash(cmd='pytest -q tests/test_calc.py')",
        "    result: 1 failed",
        "turn 1 (session 1)",
        "  tool: done()",
        "    result: Session ended by model.",
    ]


def test_show_command_prints_session_details_and_trace_tail(tmp_path, capsys):
    store = SessionStore(tmp_path)
    record = store.create_session(
        cwd=tmp_path / "work",
        model="qwen3-8b",
        prompt_text="Fix the failing test",
        prompt_source="inline",
        context_mode="full",
        system_prompt_path=None,
        config_paths=[],
    )
    artifact_dir = Path(record.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    trace_path = artifact_dir / ".trace.jsonl"
    events = [
        {"event": "session_start", "session_number": 1},
        {
            "event": "tool_call",
            "session_number": 1,
            "turn_number": 0,
            "tool_name": "read",
            "args_summary": "path='calc.py'",
            "result_summary": "def add(a, b): return a - b",
            "reasoning": "Read the buggy implementation first.",
        },
        {
            "event": "session_end",
            "session_number": 1,
            "finish_reason": "stop",
            "turns": 1,
        },
    ]
    trace_path.write_text("".join(json.dumps(event) + "\n" for event in events))
    store.update_session(record.session_id, status="completed", last_finish_reason="stop")

    with patch("scripts.llm_assist.__main__.SessionStore", return_value=store):
        rc = main(["show", record.session_id, "--trace-lines", "2"])

    captured = capsys.readouterr()
    assert rc == 0
    assert f"session_id: {record.session_id}" in captured.out
    assert f"session_ref: {record.short_id}" in captured.out
    assert "status: completed" in captured.out
    assert "finish_reason: stop" in captured.out
    assert "approval: none" in captured.out
    assert "lock: none" in captured.out
    assert "interrupt: none" in captured.out
    assert "recent_turns:" in captured.out
    assert "turn 0 (session 1)" in captured.out
    assert "reasoning: Read the buggy implementation first." in captured.out
    assert "trace_tail:" in captured.out
    assert "tool_call turn=0 read(path='calc.py')" in captured.out
    assert "session_end session=1 finish_reason=stop turns=1" in captured.out


def test_approve_command_marks_pending_request_approved(tmp_path, capsys):
    store = SessionStore(tmp_path)
    record = store.create_session(
        cwd=tmp_path / "work",
        model="qwen3-8b",
        prompt_text="Fix the failing test",
        prompt_source="inline",
        context_mode="full",
        system_prompt_path=None,
        config_paths=[],
    )
    save_approval_request(
        Path(record.artifact_dir),
        {
            "status": "pending",
            "tool_name": "bash",
            "cmd": "rm -rf build",
            "args_summary": "cmd='rm -rf build'",
            "reason": "destructive file deletion via rm",
        },
    )

    with patch("scripts.llm_assist.__main__.SessionStore", return_value=store):
        rc = main(["approve", record.session_id])

    captured = capsys.readouterr()
    approval = load_approval_request(Path(record.artifact_dir))
    assert rc == 0
    assert approval is not None
    assert approval["status"] == "approved"
    assert f"approved: {record.session_id}" in captured.out
    assert f"session_ref: {record.short_id}" in captured.out


def test_approve_always_records_session_decision(tmp_path, capsys):
    store = SessionStore(tmp_path)
    record = store.create_session(
        cwd=tmp_path / "work",
        model="qwen3-8b",
        prompt_text="Fix the failing test",
        prompt_source="inline",
        context_mode="full",
        system_prompt_path=None,
        config_paths=[],
    )
    save_approval_request(
        Path(record.artifact_dir),
        {
            "status": "pending",
            "tool_name": "bash",
            "cmd": "rm -rf build",
            "args_summary": "cmd='rm -rf build'",
            "reason": "destructive file deletion via rm",
        },
    )

    with patch("scripts.llm_assist.__main__.SessionStore", return_value=store):
        rc = main(["approve", record.session_id, "--always"])

    captured = capsys.readouterr()
    decisions = load_approval_decisions(Path(record.artifact_dir))
    assert rc == 0
    assert decisions["bash:rm -rf build"] == "approved"
    assert "always approve" in captured.out


def test_reject_command_marks_pending_request_rejected(tmp_path, capsys):
    store = SessionStore(tmp_path)
    record = store.create_session(
        cwd=tmp_path / "work",
        model="qwen3-8b",
        prompt_text="Fix the failing test",
        prompt_source="inline",
        context_mode="full",
        system_prompt_path=None,
        config_paths=[],
    )
    save_approval_request(
        Path(record.artifact_dir),
        {
            "status": "pending",
            "tool_name": "bash",
            "cmd": "rm -rf build",
            "args_summary": "cmd='rm -rf build'",
            "reason": "destructive file deletion via rm",
        },
    )

    with patch("scripts.llm_assist.__main__.SessionStore", return_value=store):
        rc = main(["reject", record.session_id, "--reason", "too broad", "--always"])

    captured = capsys.readouterr()
    approval = load_approval_request(Path(record.artifact_dir))
    decisions = load_approval_decisions(Path(record.artifact_dir))
    assert rc == 0
    assert approval is not None
    assert approval["status"] == "rejected"
    assert approval["rejection_reason"] == "too broad"
    assert decisions["bash:rm -rf build"] == "rejected"
    assert f"rejected: {record.session_id}" in captured.out
    assert "always reject" in captured.out


def test_resume_rejects_pending_approval_request(tmp_path):
    store = SessionStore(tmp_path)
    record = store.create_session(
        cwd=tmp_path / "work",
        model="qwen3-8b",
        prompt_text="Fix the failing test",
        prompt_source="inline",
        context_mode="full",
        system_prompt_path=None,
        config_paths=[],
    )
    save_approval_request(
        Path(record.artifact_dir),
        {
            "status": "pending",
            "tool_name": "bash",
            "cmd": "rm -rf build",
            "args_summary": "cmd='rm -rf build'",
            "reason": "destructive file deletion via rm",
        },
    )
    store.update_session(record.session_id, status="paused", last_finish_reason="approval_required")

    with patch("scripts.llm_assist.__main__.SessionStore", return_value=store):
        try:
            main(["resume", record.session_id])
        except SystemExit as exc:
            assert "pending approval request" in str(exc)
        else:
            raise AssertionError("expected SystemExit")


def test_resume_rejects_locked_session(tmp_path):
    store = SessionStore(tmp_path)
    record = store.create_session(
        cwd=tmp_path / "work",
        model="qwen3-8b",
        prompt_text="Fix the failing test",
        prompt_source="inline",
        context_mode="full",
        system_prompt_path=None,
        config_paths=[],
    )
    store.update_session(record.session_id, status="paused", last_finish_reason="max_turns")
    foreign_lock = SessionLock(
        session_id=record.session_id,
        owner_host="other-host",
        owner_pid=4242,
        acquired_at="2026-04-25T00:00:00+00:00",
    )

    with patch("scripts.llm_assist.__main__.SessionStore", return_value=store), \
            patch.object(store, "acquire_session_lock", side_effect=SessionLockedError(foreign_lock)):
        try:
            main(["resume", record.session_id])
        except SystemExit as exc:
            assert "already locked by pid 4242 on other-host" in str(exc)
        else:
            raise AssertionError("expected SystemExit")


def test_show_command_prints_pending_approval_request(tmp_path, capsys):
    store = SessionStore(tmp_path)
    record = store.create_session(
        cwd=tmp_path / "work",
        model="qwen3-8b",
        prompt_text="Fix the failing test",
        prompt_source="inline",
        context_mode="full",
        system_prompt_path=None,
        config_paths=[],
    )
    save_approval_request(
        Path(record.artifact_dir),
        {
            "status": "pending",
            "tool_name": "bash",
            "cmd": "rm -rf build",
            "args_summary": "cmd='rm -rf build'",
            "reason": "destructive file deletion via rm",
        },
    )

    with patch("scripts.llm_assist.__main__.SessionStore", return_value=store):
        rc = main(["show", record.session_id, "--trace-lines", "0", "--turns", "0"])

    captured = capsys.readouterr()
    assert rc == 0
    assert "status: approval_pending" in captured.out
    assert f"session_ref: {record.short_id}" in captured.out
    assert "approval: pending" in captured.out
    assert "approval_reason: destructive file deletion via rm" in captured.out
    assert "approval_action: bash(cmd='rm -rf build')" in captured.out
    assert "interrupt: none" in captured.out


def test_show_command_reports_running_for_resumed_active_session(tmp_path, capsys):
    store = SessionStore(tmp_path)
    record = store.create_session(
        cwd=tmp_path / "work",
        model="qwen3-8b",
        prompt_text="Fix the failing test",
        prompt_source="inline",
        context_mode="full",
        system_prompt_path=None,
        config_paths=[],
    )
    # SQLite row still reflects the prior paused state after approval.
    store.update_session(record.session_id, status="paused", last_finish_reason="approval_required")

    artifact_dir = Path(record.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    trace_path = artifact_dir / ".trace.jsonl"
    events = [
        {"event": "session_start", "session_number": 1},
        {
            "event": "session_end",
            "session_number": 1,
            "finish_reason": "approval_required",
            "turns": 2,
        },
        {"event": "session_start", "session_number": 2},
    ]
    trace_path.write_text("".join(json.dumps(e) + "\n" for e in events))

    with patch("scripts.llm_assist.__main__.SessionStore", return_value=store):
        rc = main(["show", record.session_id, "--trace-lines", "0", "--turns", "0"])

    captured = capsys.readouterr()
    assert rc == 0
    assert "status: running" in captured.out
    assert f"session_ref: {record.short_id}" in captured.out
    assert "current_session: 2" in captured.out
    assert "finish_reason: approval_required" not in captured.out


def test_show_command_preserves_completed_finish_reason(tmp_path, capsys):
    store = SessionStore(tmp_path)
    record = store.create_session(
        cwd=tmp_path / "work",
        model="qwen3-8b",
        prompt_text="Fix the failing test",
        prompt_source="inline",
        context_mode="full",
        system_prompt_path=None,
        config_paths=[],
    )
    store.update_session(record.session_id, status="completed", last_finish_reason="stop")
    artifact_dir = Path(record.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    trace_path = artifact_dir / ".trace.jsonl"
    events = [
        {"event": "session_start", "session_number": 1},
        {
            "event": "session_end",
            "session_number": 1,
            "finish_reason": "stop",
            "turns": 3,
        },
    ]
    trace_path.write_text("".join(json.dumps(e) + "\n" for e in events))

    with patch("scripts.llm_assist.__main__.SessionStore", return_value=store):
        rc = main(["show", record.session_id, "--trace-lines", "0", "--turns", "0"])

    captured = capsys.readouterr()
    assert rc == 0
    assert "status: completed" in captured.out
    assert f"session_ref: {record.short_id}" in captured.out
    assert "finish_reason: stop" in captured.out
    assert "current_session: 1" in captured.out


def test_prepare_smoke_repo_creates_bugged_repo(tmp_path: Path):
    repo = prepare_smoke_repo(tmp_path / "assist-smoke")

    assert repo == (tmp_path / "assist-smoke").resolve()
    assert (repo / "calc.py").read_text() == "def add(a, b):\n    return a - b\n"
    assert "assert add(2, 3) == 5" in (repo / "tests" / "test_calc.py").read_text()


def test_resolve_smoke_model_prefers_exact_served_id_when_alias_missing():
    with patch("scripts.llm_assist.runner._default_model", return_value="qwen3-8b"), \
            patch("scripts.llm_assist.runner.load_config", return_value=object()), \
            patch("scripts.llm_assist.runner._load_profile", return_value=None), \
            patch("scripts.llm_assist.runner.LlamaClient") as client_cls:
        client_cls.return_value.health_check.return_value = [
            "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"
        ]

        model, served = resolve_smoke_model([])

    assert model == "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"
    assert served == ["Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"]


def test_resolve_served_model_returns_exact_id_when_alias_present():
    with patch("scripts.llm_assist.runner._default_model", return_value="qwen3-8b"), \
            patch("scripts.llm_assist.runner.load_config", return_value=object()), \
            patch("scripts.llm_assist.runner._load_profile", return_value=None), \
            patch("scripts.llm_assist.runner.LlamaClient") as client_cls:
        client_cls.return_value.health_check.return_value = [
            "qwen3-8b",
            "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf",
        ]

        model, served = resolve_served_model([])

    assert model == "qwen3-8b"
    assert "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf" in served


def test_resolve_served_model_falls_back_to_first_served_id():
    with patch("scripts.llm_assist.runner._default_model", return_value="qwen3-8b"), \
            patch("scripts.llm_assist.runner.load_config", return_value=object()), \
            patch("scripts.llm_assist.runner._load_profile", return_value=None), \
            patch("scripts.llm_assist.runner.LlamaClient") as client_cls:
        client_cls.return_value.health_check.return_value = [
            "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf",
            "fallback-model.gguf",
        ]

        model, served = resolve_served_model([], requested_model="qwen3-8b")

    assert model == "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"
    assert served[0] == "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"


def test_resolve_served_model_honors_explicit_remote_model_when_models_list_differs():
    with patch("scripts.llm_assist.runner.load_config", return_value=object()), \
            patch("scripts.llm_assist.runner._load_profile", return_value=None), \
            patch("scripts.llm_assist.runner.LlamaClient") as client_cls:
        client_cls.return_value.health_check.return_value = [
            "provider-default-model",
        ]

        model, served = resolve_served_model(
            [],
            requested_model="provider/exact-model",
            config_overrides={
                "provider": "openai-compatible",
                "base_url": "https://openrouter.ai/api/v1",
                "api_key": "$ENV:OPENROUTER_API_KEY",
            },
        )

    assert model == "provider/exact-model"
    assert served == ["provider-default-model"]


def test_trace_follower_prints_new_events(tmp_path: Path):
    artifact_dir = tmp_path / "session"
    artifact_dir.mkdir(parents=True)
    trace_path = artifact_dir / ".trace.jsonl"
    trace_path.write_text("")

    rendered: list[str] = []
    follower = TraceFollower(artifact_dir, print_fn=rendered.append, poll_interval=0.01)
    follower.start()

    events = [
        {"event": "session_start", "session_number": 1},
        {
            "event": "tool_call",
            "session_number": 1,
            "turn_number": 0,
            "tool_name": "read",
            "args_summary": "path='calc.py'",
            "result_summary": "def add(a, b): return a - b",
        },
        {
            "event": "approval_request",
            "session_number": 1,
            "turn_number": 1,
            "tool_name": "bash",
            "args_summary": "cmd='rm -rf build'",
            "reason": "destructive file deletion via rm",
        },
        {
            "event": "session_end",
            "session_number": 1,
            "finish_reason": "approval_required",
            "turns": 2,
        },
    ]
    with open(trace_path, "a") as f:
        for event in events:
            f.write(json.dumps(event) + "\n")
            f.flush()

    follower.stop()

    assert any(line.startswith("session_start session=1") for line in rendered)
    assert any("tool_call turn=0 read" in line for line in rendered)
    assert any("approval_request turn=1 bash" in line for line in rendered)
    assert any("session_end session=1 finish_reason=approval_required" in line for line in rendered)


def test_trace_follower_does_not_duplicate_existing_events(tmp_path: Path):
    artifact_dir = tmp_path / "session"
    artifact_dir.mkdir(parents=True)
    trace_path = artifact_dir / ".trace.jsonl"

    historical = [
        {"event": "session_start", "session_number": 1},
        {
            "event": "tool_call",
            "session_number": 1,
            "turn_number": 0,
            "tool_name": "read",
            "args_summary": "path='old.py'",
            "result_summary": "stale contents",
        },
        {
            "event": "session_end",
            "session_number": 1,
            "finish_reason": "stop",
            "turns": 1,
        },
    ]
    trace_path.write_text("".join(json.dumps(e) + "\n" for e in historical))

    rendered: list[str] = []
    follower = TraceFollower(artifact_dir, print_fn=rendered.append, poll_interval=0.01)
    follower.start()

    new_event = {
        "event": "tool_call",
        "session_number": 2,
        "turn_number": 0,
        "tool_name": "edit",
        "args_summary": "path='new.py'",
        "result_summary": "ok",
    }
    with open(trace_path, "a") as f:
        f.write(json.dumps(new_event) + "\n")
        f.flush()

    follower.stop()

    assert not any("old.py" in line for line in rendered)
    assert any("tool_call turn=0 edit" in line and "new.py" in line for line in rendered)


def test_cmd_run_prints_progress_before_final_result(tmp_path, capsys):
    store = SessionStore(tmp_path / "assist-home")
    work_dir = tmp_path / "work"
    work_dir.mkdir()

    def fake_run_session(store_obj, record, *, resume):
        artifact_dir = Path(record.artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        trace_path = artifact_dir / ".trace.jsonl"
        events = [
            {"event": "session_start", "session_number": 1},
            {
                "event": "tool_call",
                "session_number": 1,
                "turn_number": 0,
                "tool_name": "bash",
                "args_summary": "cmd='pytest -q'",
                "result_summary": "1 passed",
            },
            {
                "event": "session_end",
                "session_number": 1,
                "finish_reason": "stop",
                "turns": 1,
            },
        ]
        # Write in a loop with small delays so the follower's poll sees them.
        import time as _t
        with open(trace_path, "a") as f:
            for event in events:
                f.write(json.dumps(event) + "\n")
                f.flush()
                _t.sleep(0.02)
        store_obj.update_session(record.session_id, status="completed", last_finish_reason="stop")
        return True, "stop"

    with patch("scripts.llm_assist.__main__.SessionStore", return_value=store), \
            patch(
                "scripts.llm_assist.__main__.resolve_served_model",
                return_value=("exact-served.gguf", ["exact-served.gguf"]),
            ), \
            patch("scripts.llm_assist.__main__.run_session", side_effect=fake_run_session):
        rc = main([
            "run",
            "--cwd", str(work_dir),
            "--prompt-text", "do it",
        ])

    captured = capsys.readouterr()
    assert rc == 0
    # Progress output from the follower precedes the closing "status: completed"
    # block printed by _print_session_result.
    idx_progress = captured.out.find("tool_call turn=0 bash")
    idx_status = captured.out.find("status: completed")
    idx_start = captured.out.find("starting:")
    assert "ref:" in captured.out
    assert idx_start != -1
    assert idx_progress != -1 and idx_status != -1
    assert idx_start < idx_progress
    assert idx_progress < idx_status
    assert "summary:" in captured.out
    assert "last_test: pytest -q" in captured.out
    assert "last_test_result: pass" in captured.out
    sessions = store.list_sessions(limit=1)
    assert sessions, "expected a persisted session record"
    assert store.get_session_lock(sessions[0].session_id) is None


def test_cmd_run_keyboard_interrupt_marks_session_interrupted_and_resumable(tmp_path, capsys):
    store = SessionStore(tmp_path / "assist-home")
    work_dir = tmp_path / "work"
    work_dir.mkdir()

    def fake_run_session(store_obj, record, *, resume):
        artifact_dir = Path(record.artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / ".trace.jsonl").write_text(
            json.dumps({
                "event": "session_start",
                "session_number": 1,
            }) + "\n"
        )
        raise KeyboardInterrupt

    with patch("scripts.llm_assist.__main__.SessionStore", return_value=store), \
            patch(
                "scripts.llm_assist.__main__.resolve_served_model",
                return_value=("exact-served.gguf", ["exact-served.gguf"]),
            ), \
            patch("scripts.llm_assist.__main__.run_session", side_effect=fake_run_session):
        rc = main([
            "run",
            "--cwd", str(work_dir),
            "--prompt-text", "do it",
        ])

    captured = capsys.readouterr()
    sessions = store.list_sessions(limit=1)
    assert rc == 130
    assert sessions, "expected a persisted session record"
    record = sessions[0]
    assert record.status == "paused"
    assert record.last_finish_reason == "interrupted"
    assert "interrupted: session paused cleanly" in captured.out
    assert f"resume with: yuj resume {record.short_id}" in captured.out
    assert load_interrupt_marker(Path(record.artifact_dir)) is not None
    assert store.get_session_lock(record.session_id) is None


def test_code_alias_routes_to_run(tmp_path, capsys):
    store = SessionStore(tmp_path / "assist-home")
    work_dir = tmp_path / "work"
    work_dir.mkdir()

    def fake_run_session(store_obj, record, *, resume):
        artifact_dir = Path(record.artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / ".trace.jsonl").write_text(
            json.dumps({
                "event": "session_end",
                "session_number": 1,
                "finish_reason": "stop",
                "turns": 1,
            }) + "\n"
        )
        store_obj.update_session(record.session_id, status="completed", last_finish_reason="stop")
        return True, "stop"

    with patch("scripts.llm_assist.__main__.SessionStore", return_value=store), \
            patch(
                "scripts.llm_assist.__main__.resolve_served_model",
                return_value=("exact-served.gguf", ["exact-served.gguf"]),
            ), \
            patch("scripts.llm_assist.__main__.run_session", side_effect=fake_run_session) as run_mock:
        rc = main([
            "code",
            "--cwd", str(work_dir),
            "--prompt-text", "do it",
        ])

    assert rc == 0
    assert run_mock.called is True
    sessions = store.list_sessions(limit=1)
    assert sessions and sessions[0].status == "completed"


def test_code_alias_help_exits_cleanly(capsys):
    try:
        main(["code", "--help"])
    except SystemExit as exc:
        assert exc.code == 0
    else:
        raise AssertionError("expected SystemExit")
    captured = capsys.readouterr()
    assert "usage: yuj code" in captured.out
    assert "--cwd" in captured.out
    assert "--prompt-text" in captured.out


def test_root_help_lists_status_and_current_commands(capsys):
    try:
        main(["--help"])
    except SystemExit as exc:
        assert exc.code == 0
    else:
        raise AssertionError("expected SystemExit")
    captured = capsys.readouterr()
    assert "status" in captured.out
    assert "current" in captured.out
    assert "setup" in captured.out


def test_setup_writes_local_provider_config(tmp_path, monkeypatch, capsys):
    config_path = tmp_path / "config.local.toml"
    monkeypatch.setenv("YUJ_CONFIG_LOCAL", str(config_path))

    rc = main([
        "setup",
        "--provider", "openai",
        "--model", "gpt-5.4",
        "--api-key", "test-secret",
        "--force",
    ])

    captured = capsys.readouterr()
    text = config_path.read_text()
    assert rc == 0
    assert f"wrote: {config_path}" in captured.out
    assert 'provider = "openai-compatible"' in text
    assert 'base_url = "https://api.openai.com/v1"' in text
    assert 'api_key = "test-secret"' in text
    assert 'name = "gpt-5.4"' in text


def test_setup_defaults_to_local_without_prompts_in_a_script(
    tmp_path, monkeypatch,
):
    config_path = tmp_path / "config.local.toml"
    monkeypatch.setenv("YUJ_CONFIG_LOCAL", str(config_path))
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)

    with patch(
        "scripts.llm_assist.__main__.getpass.getpass",
        side_effect=AssertionError("setup must not prompt"),
    ):
        rc = main([
            "setup",
            "--model", "local-model",
            "--force",
        ])

    text = config_path.read_text()
    assert rc == 0
    assert 'provider = "openai-compatible"' in text
    assert 'base_url = "http://localhost:8080/v1"' in text
    assert 'api_key = "local"' in text
    assert 'name = "local-model"' in text


def test_setup_can_store_api_key_env_reference(tmp_path, monkeypatch):
    config_path = tmp_path / "config.local.toml"
    monkeypatch.setenv("YUJ_CONFIG_LOCAL", str(config_path))

    rc = main([
        "setup",
        "--provider", "anthropic",
        "--model", "claude-sonnet-4-5",
        "--api-key-env", "ANTHROPIC_API_KEY",
        "--force",
    ])

    text = config_path.read_text()
    assert rc == 0
    assert 'provider = "anthropic"' in text
    assert 'api_key = "$ENV:ANTHROPIC_API_KEY"' in text
    assert "claude-sonnet-4-5" in text


def test_anthropic_adapter_preserves_tool_turns():
    from scripts.llm_assist._anthropic import (
        _anthropic_to_openai_response,
        _to_anthropic_payload,
    )

    payload = {
        "model": "claude-sonnet-4-5",
        "messages": [
            {"role": "system", "content": "Use tools."},
            {"role": "user", "content": "Read calc.py"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_0_0",
                    "type": "function",
                    "function": {
                        "name": "read",
                        "arguments": '{"path": "calc.py"}',
                    },
                }],
            },
            {
                "role": "tool",
                "tool_call_id": "call_0_0",
                "content": "def add(a, b): return a - b",
            },
        ],
        "tools": [{
            "type": "function",
            "function": {
                "name": "read",
                "description": "Read a file",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                },
            },
        }],
        "max_tokens": 128,
    }

    anthropic_payload = _to_anthropic_payload(payload)
    assert anthropic_payload["system"] == "Use tools."
    assert anthropic_payload["messages"][1]["content"][0]["type"] == "tool_use"
    assert anthropic_payload["messages"][2]["content"][0]["type"] == "tool_result"

    compat = _anthropic_to_openai_response({
        "content": [{
            "type": "tool_use",
            "id": "toolu_1",
            "name": "edit",
            "input": {"path": "calc.py", "old_str": "-", "new_str": "+"},
        }],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 10, "output_tokens": 5},
    })
    assert compat.choices[0].finish_reason == "tool_calls"
    tool_call = compat.choices[0].message.tool_calls[0]
    assert tool_call.id == "toolu_1"
    assert tool_call.function.name == "edit"
    assert json.loads(tool_call.function.arguments)["path"] == "calc.py"


def test_models_command_lists_provider_models(capsys):
    class Cfg:
        provider = "openai-compatible"
        base_url = "http://localhost:8080/v1"
        model = "served-a"
        api_key = "local"
        timeout_connect = 1
        timeout_read = 1

    with patch("scripts.llm_assist.__main__._load_assistant_config", return_value=Cfg()), \
            patch("scripts.llm_assist.__main__._make_client") as client_factory:
        client_factory.return_value.health_check.return_value = ["served-a", "served-b"]
        rc = main(["models"])

    captured = capsys.readouterr()
    assert rc == 0
    assert "provider: openai-compatible" in captured.out
    assert "served-a *" in captured.out
    assert "served-b" in captured.out


def test_doctor_reports_model_failure(capsys, tmp_path, monkeypatch):
    class Cfg:
        provider = "openai-compatible"
        base_url = "http://localhost:8080/v1"
        model = "missing-model"
        api_key = "local"
        timeout_connect = 1
        timeout_read = 1

    monkeypatch.setenv("YUJ_CONFIG_LOCAL", str(tmp_path / "config.local.toml"))
    with patch("scripts.llm_assist.__main__._load_assistant_config", return_value=Cfg()), \
            patch("scripts.llm_assist.__main__._make_client") as client_factory:
        client_factory.return_value.health_check.return_value = ["served-a"]
        rc = main(["doctor"])

    captured = capsys.readouterr()
    assert rc == 1
    assert "config: ok" in captured.out
    assert "models: ok (1 returned)" in captured.out
    assert "selected_model: fail" in captured.out


def test_code_uses_positional_prompt_and_current_dir_by_default(tmp_path, capsys, monkeypatch):
    store = SessionStore(tmp_path / "assist-home")
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    monkeypatch.chdir(work_dir)

    def fake_run_session(store_obj, record, *, resume):
        artifact_dir = Path(record.artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / ".trace.jsonl").write_text(
            json.dumps({
                "event": "session_end",
                "session_number": 1,
                "finish_reason": "stop",
                "turns": 1,
            }) + "\n"
        )
        store_obj.update_session(record.session_id, status="completed", last_finish_reason="stop")
        return True, "stop"

    with patch("scripts.llm_assist.__main__.SessionStore", return_value=store), \
            patch(
                "scripts.llm_assist.__main__.resolve_served_model",
                return_value=("exact-served.gguf", ["exact-served.gguf"]),
            ), \
            patch("scripts.llm_assist.__main__.run_session", side_effect=fake_run_session):
        rc = main(["code", "fix", "the", "failing", "test"])

    captured = capsys.readouterr()
    assert rc == 0
    assert "starting:" in captured.out
    sessions = store.list_sessions(limit=1)
    assert sessions, "expected a persisted session record"
    assert sessions[0].cwd == str(work_dir.resolve())
    assert sessions[0].prompt_text == "fix the failing test"
    assert sessions[0].prompt_source == "inline-positional"
    assert sessions[0].context_mode == "halflife"
    assert sessions[0].config_paths[0].endswith("configs/regimes/treatment.toml")
    assert store.get_active_session_id(work_dir) == sessions[0].session_id


def test_code_no_treatment_uses_plain_package_and_full_context(tmp_path, monkeypatch):
    store = SessionStore(tmp_path / "assist-home")
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    user_overlay = tmp_path / "user.toml"
    user_overlay.write_text("[loop]\nmax_turns = 12\n")
    monkeypatch.chdir(work_dir)

    def fake_run_session(store_obj, record, *, resume):
        artifact_dir = Path(record.artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / ".trace.jsonl").write_text(
            json.dumps({
                "event": "session_end",
                "session_number": 1,
                "finish_reason": "stop",
                "turns": 1,
            }) + "\n"
        )
        store_obj.update_session(record.session_id, status="completed", last_finish_reason="stop")
        return True, "stop"

    with patch("scripts.llm_assist.__main__.SessionStore", return_value=store), \
            patch(
                "scripts.llm_assist.__main__.resolve_served_model",
                return_value=("exact-served.gguf", ["exact-served.gguf"]),
            ), \
            patch("scripts.llm_assist.__main__.run_session", side_effect=fake_run_session):
        rc = main([
            "code",
            "--no-treatment",
            "--config", str(user_overlay),
            "fix the test",
        ])

    assert rc == 0
    record = store.list_sessions(limit=1)[0]
    assert record.context_mode == "full"
    assert record.config_paths[0].endswith(
        "configs/regimes/baselines/plain_long_solve.toml"
    )
    assert Path(record.config_paths[1]) == user_overlay


def test_run_persists_exact_served_model_id(tmp_path, capsys):
    store = SessionStore(tmp_path / "assist-home")
    work_dir = tmp_path / "work"
    work_dir.mkdir()

    def fake_run_session(store_obj, record, *, resume):
        artifact_dir = Path(record.artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / ".trace.jsonl").write_text(
            json.dumps({
                "event": "session_end",
                "session_number": 1,
                "finish_reason": "stop",
                "turns": 1,
            }) + "\n"
        )
        store_obj.update_session(record.session_id, status="completed", last_finish_reason="stop")
        return True, "stop"

    with patch("scripts.llm_assist.__main__.SessionStore", return_value=store), \
            patch(
                "scripts.llm_assist.__main__.resolve_served_model",
                return_value=("exact-served-id.gguf", ["exact-served-id.gguf"]),
            ), \
            patch("scripts.llm_assist.__main__.run_session", side_effect=fake_run_session):
        rc = main([
            "run",
            "--cwd", str(work_dir),
            "--prompt-text", "do the thing",
            "--model", "qwen3-8b",
        ])

    assert rc == 0
    sessions = store.list_sessions(limit=1)
    assert sessions, "expected a persisted session record"
    assert sessions[0].model == "exact-served-id.gguf"


def test_run_provider_preset_persists_session_overlay(tmp_path, monkeypatch):
    store = SessionStore(tmp_path / "assist-home")
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret-value")
    seen_config_paths: list[str] = []

    def fake_run_session(store_obj, record, *, resume):
        seen_config_paths.extend(record.config_paths)
        artifact_dir = Path(record.artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / ".trace.jsonl").write_text(
            json.dumps({
                "event": "session_end",
                "session_number": 1,
                "finish_reason": "stop",
                "turns": 1,
            }) + "\n"
        )
        store_obj.update_session(record.session_id, status="completed", last_finish_reason="stop")
        return True, "stop"

    with patch("scripts.llm_assist.__main__.SessionStore", return_value=store), \
            patch(
                "scripts.llm_assist.__main__.resolve_served_model",
                return_value=("openrouter/model", ["openrouter/model"]),
            ) as resolve_mock, \
            patch("scripts.llm_assist.__main__.run_session", side_effect=fake_run_session):
        rc = main([
            "run",
            "--cwd", str(work_dir),
            "--prompt-text", "do it",
            "--provider", "openrouter",
            "--model", "openrouter/model",
        ])

    assert rc == 0
    resolve_mock.assert_called_once()
    assert resolve_mock.call_args.kwargs["config_overrides"] == {
        "provider": "openai-compatible",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": "$ENV:OPENROUTER_API_KEY",
    }
    assert seen_config_paths, "expected persisted provider overlay in config paths"
    overlay = Path(seen_config_paths[-1])
    assert overlay.name == "provider.toml"
    text = overlay.read_text()
    assert 'base_url = "https://openrouter.ai/api/v1"' in text
    assert 'api_key = "$ENV:OPENROUTER_API_KEY"' in text
    assert "secret-value" not in text


def test_custom_provider_requires_base_url(tmp_path):
    store = SessionStore(tmp_path / "assist-home")
    work_dir = tmp_path / "work"
    work_dir.mkdir()

    with patch("scripts.llm_assist.__main__.SessionStore", return_value=store):
        try:
            main([
                "run",
                "--cwd", str(work_dir),
                "--prompt-text", "do it",
                "--provider", "custom",
            ])
        except SystemExit as exc:
            assert "--provider custom requires --base-url" in str(exc)
        else:
            raise AssertionError("expected SystemExit")


def test_show_without_id_prefers_latest_session_in_current_cwd(tmp_path, capsys, monkeypatch):
    store = SessionStore(tmp_path / "assist-home")
    repo_a = tmp_path / "repo-a"
    repo_b = tmp_path / "repo-b"
    repo_a.mkdir()
    repo_b.mkdir()
    monkeypatch.chdir(repo_a)

    first = store.create_session(
        cwd=repo_b,
        model="other-model",
        prompt_text="other repo",
        prompt_source="inline",
        context_mode="full",
        system_prompt_path=None,
        config_paths=[],
    )
    second = store.create_session(
        cwd=repo_a,
        model="local-model",
        prompt_text="current repo",
        prompt_source="inline",
        context_mode="full",
        system_prompt_path=None,
        config_paths=[],
    )
    artifact_dir = Path(second.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / ".trace.jsonl").write_text(
        json.dumps({
            "event": "session_end",
            "session_number": 1,
            "finish_reason": "stop",
            "turns": 1,
        }) + "\n"
    )

    with patch("scripts.llm_assist.__main__.SessionStore", return_value=store):
        rc = main(["show", "--trace-lines", "0", "--turns", "0"])

    captured = capsys.readouterr()
    assert rc == 0
    assert first.session_id not in captured.out
    assert f"session_id: {second.session_id}" in captured.out


def test_show_without_id_prefers_active_session_over_newer_session_in_current_cwd(tmp_path, capsys, monkeypatch):
    store = SessionStore(tmp_path / "assist-home")
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)

    first = store.create_session(
        cwd=repo,
        model="older-model",
        prompt_text="older",
        prompt_source="inline",
        context_mode="full",
        system_prompt_path=None,
        config_paths=[],
    )
    second = store.create_session(
        cwd=repo,
        model="newer-model",
        prompt_text="newer",
        prompt_source="inline",
        context_mode="full",
        system_prompt_path=None,
        config_paths=[],
    )
    store.set_active_session(repo, first.session_id)
    artifact_dir = Path(first.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / ".trace.jsonl").write_text(
        json.dumps({
            "event": "session_end",
            "session_number": 1,
            "finish_reason": "stop",
            "turns": 1,
        }) + "\n"
    )

    with patch("scripts.llm_assist.__main__.SessionStore", return_value=store):
        rc = main(["show", "--trace-lines", "0", "--turns", "0"])

    captured = capsys.readouterr()
    assert rc == 0
    assert f"session_id: {first.session_id}" in captured.out
    assert second.session_id not in captured.out


def test_show_command_reports_interrupted_session_from_marker(tmp_path, capsys):
    store = SessionStore(tmp_path)
    record = store.create_session(
        cwd=tmp_path / "work",
        model="qwen3-8b",
        prompt_text="Fix the failing test",
        prompt_source="inline",
        context_mode="full",
        system_prompt_path=None,
        config_paths=[],
    )
    store.update_session(record.session_id, status="paused", last_finish_reason="interrupted")
    artifact_dir = Path(record.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / ".trace.jsonl").write_text(
        json.dumps({
            "event": "session_start",
            "session_number": 1,
        }) + "\n"
    )
    (artifact_dir / "shell_interrupt.json").write_text(
        json.dumps({
            "finish_reason": "interrupted",
            "interrupted_at": "2026-04-25T00:00:00+00:00",
        }) + "\n"
    )

    with patch("scripts.llm_assist.__main__.SessionStore", return_value=store):
        rc = main(["show", record.session_id, "--trace-lines", "0", "--turns", "0"])

    captured = capsys.readouterr()
    assert rc == 0
    assert "status: paused" in captured.out
    assert f"session_ref: {record.short_id}" in captured.out
    assert "finish_reason: interrupted" in captured.out
    assert "interrupt: interrupted at 2026-04-25T00:00:00+00:00" in captured.out


def test_status_command_prints_next_action_for_pending_approval(tmp_path, capsys):
    store = SessionStore(tmp_path / "assist-home")
    record = store.create_session(
        cwd=tmp_path / "work",
        model="qwen3-8b",
        prompt_text="Fix the failing test",
        prompt_source="inline",
        context_mode="full",
        system_prompt_path=None,
        config_paths=[],
    )
    store.update_session(record.session_id, status="paused", last_finish_reason="approval_required")
    save_approval_request(
        Path(record.artifact_dir),
        {
            "status": "pending",
            "tool_name": "bash",
            "cmd": "rm -rf build",
            "args_summary": "cmd='rm -rf build'",
            "reason": "destructive file deletion via rm",
        },
    )

    with patch("scripts.llm_assist.__main__.SessionStore", return_value=store):
        rc = main(["status", record.short_id])

    captured = capsys.readouterr()
    assert rc == 0
    assert "approval: pending" in captured.out
    assert f"next: yuj approve {record.short_id}" in captured.out


def test_status_command_prints_next_action_for_running_session(tmp_path, capsys):
    store = SessionStore(tmp_path / "assist-home")
    record = store.create_session(
        cwd=tmp_path / "work",
        model="qwen3-8b",
        prompt_text="Fix the failing test",
        prompt_source="inline",
        context_mode="full",
        system_prompt_path=None,
        config_paths=[],
    )
    artifact_dir = Path(record.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / ".trace.jsonl").write_text(
        json.dumps({
            "event": "session_start",
            "session_number": 1,
        }) + "\n"
    )

    with patch("scripts.llm_assist.__main__.SessionStore", return_value=store):
        rc = main(["status", record.short_id])

    captured = capsys.readouterr()
    assert rc == 0
    assert "status: running" in captured.out
    assert f"next: yuj show {record.short_id}" in captured.out


def test_current_alias_routes_to_status_latest(tmp_path, capsys, monkeypatch):
    store = SessionStore(tmp_path / "assist-home")
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)
    record = store.create_session(
        cwd=repo,
        model="qwen3-8b",
        prompt_text="Fix the failing test",
        prompt_source="inline",
        context_mode="full",
        system_prompt_path=None,
        config_paths=[],
    )

    with patch("scripts.llm_assist.__main__.SessionStore", return_value=store):
        rc = main(["current"])

    captured = capsys.readouterr()
    assert rc == 0
    assert f"session_id: {record.session_id}" in captured.out
    assert f"session_ref: {record.short_id}" in captured.out


def test_show_accepts_short_session_ref(tmp_path, capsys):
    store = SessionStore(tmp_path)
    record = store.create_session(
        cwd=tmp_path / "work",
        model="qwen3-8b",
        prompt_text="Fix the failing test",
        prompt_source="inline",
        context_mode="full",
        system_prompt_path=None,
        config_paths=[],
    )
    artifact_dir = Path(record.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / ".trace.jsonl").write_text(
        json.dumps({
            "event": "session_end",
            "session_number": 1,
            "finish_reason": "stop",
            "turns": 1,
        }) + "\n"
    )

    with patch("scripts.llm_assist.__main__.SessionStore", return_value=store):
        rc = main(["show", record.short_id, "--trace-lines", "0", "--turns", "0"])

    captured = capsys.readouterr()
    assert rc == 0
    assert f"session_id: {record.session_id}" in captured.out
    assert f"session_ref: {record.short_id}" in captured.out


def test_resume_without_id_prefers_latest_resumable_session_in_current_cwd(tmp_path, monkeypatch):
    store = SessionStore(tmp_path / "assist-home")
    repo_a = tmp_path / "repo-a"
    repo_b = tmp_path / "repo-b"
    repo_a.mkdir()
    repo_b.mkdir()
    monkeypatch.chdir(repo_a)

    local_completed = store.create_session(
        cwd=repo_a,
        model="done-model",
        prompt_text="done",
        prompt_source="inline",
        context_mode="full",
        system_prompt_path=None,
        config_paths=[],
    )
    store.update_session(local_completed.session_id, status="completed", last_finish_reason="stop")

    local_paused = store.create_session(
        cwd=repo_a,
        model="paused-model",
        prompt_text="resume me",
        prompt_source="inline",
        context_mode="full",
        system_prompt_path=None,
        config_paths=[],
    )
    store.update_session(local_paused.session_id, status="paused", last_finish_reason="max_turns")

    remote_paused = store.create_session(
        cwd=repo_b,
        model="remote-model",
        prompt_text="remote",
        prompt_source="inline",
        context_mode="full",
        system_prompt_path=None,
        config_paths=[],
    )
    store.update_session(remote_paused.session_id, status="paused", last_finish_reason="max_turns")

    selected: list[str] = []

    def fake_run_session(store_obj, record, *, resume):
        selected.append(record.session_id)
        artifact_dir = Path(record.artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / ".trace.jsonl").write_text(
            json.dumps({
                "event": "session_end",
                "session_number": 2,
                "finish_reason": "stop",
                "turns": 2,
            }) + "\n"
        )
        store_obj.update_session(record.session_id, status="completed", last_finish_reason="stop")
        return True, "stop"

    with patch("scripts.llm_assist.__main__.SessionStore", return_value=store), \
            patch("scripts.llm_assist.__main__.run_session", side_effect=fake_run_session):
        rc = main(["resume"])

    assert rc == 0
    assert selected == [local_paused.session_id]


def test_resume_without_id_prefers_active_resumable_session_over_newer_resumable(tmp_path, monkeypatch):
    store = SessionStore(tmp_path / "assist-home")
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)

    first = store.create_session(
        cwd=repo,
        model="older-model",
        prompt_text="resume older",
        prompt_source="inline",
        context_mode="full",
        system_prompt_path=None,
        config_paths=[],
    )
    store.update_session(first.session_id, status="paused", last_finish_reason="max_turns")
    second = store.create_session(
        cwd=repo,
        model="newer-model",
        prompt_text="resume newer",
        prompt_source="inline",
        context_mode="full",
        system_prompt_path=None,
        config_paths=[],
    )
    store.update_session(second.session_id, status="paused", last_finish_reason="max_turns")
    store.set_active_session(repo, first.session_id)

    selected: list[str] = []

    def fake_run_session(store_obj, record, *, resume):
        selected.append(record.session_id)
        artifact_dir = Path(record.artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / ".trace.jsonl").write_text(
            json.dumps({
                "event": "session_end",
                "session_number": 2,
                "finish_reason": "stop",
                "turns": 2,
            }) + "\n"
        )
        store_obj.update_session(record.session_id, status="completed", last_finish_reason="stop")
        return True, "stop"

    with patch("scripts.llm_assist.__main__.SessionStore", return_value=store), \
            patch("scripts.llm_assist.__main__.run_session", side_effect=fake_run_session):
        rc = main(["resume"])

    assert rc == 0
    assert selected == [first.session_id]


def test_resume_accepts_short_session_ref(tmp_path):
    store = SessionStore(tmp_path / "assist-home")
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    record = store.create_session(
        cwd=work_dir,
        model="paused-model",
        prompt_text="resume me",
        prompt_source="inline",
        context_mode="full",
        system_prompt_path=None,
        config_paths=[],
    )
    store.update_session(record.session_id, status="paused", last_finish_reason="max_turns")

    selected: list[str] = []

    def fake_run_session(store_obj, chosen, *, resume):
        selected.append(chosen.session_id)
        artifact_dir = Path(chosen.artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / ".trace.jsonl").write_text(
            json.dumps({
                "event": "session_end",
                "session_number": 2,
                "finish_reason": "stop",
                "turns": 1,
            }) + "\n"
        )
        store_obj.update_session(chosen.session_id, status="completed", last_finish_reason="stop")
        return True, "stop"

    with patch("scripts.llm_assist.__main__.SessionStore", return_value=store), \
            patch("scripts.llm_assist.__main__.run_session", side_effect=fake_run_session):
        rc = main(["resume", record.short_id])

    assert rc == 0
    assert selected == [record.session_id]


def test_approve_without_id_prefers_latest_pending_request_in_current_cwd(tmp_path, capsys, monkeypatch):
    store = SessionStore(tmp_path / "assist-home")
    repo_a = tmp_path / "repo-a"
    repo_b = tmp_path / "repo-b"
    repo_a.mkdir()
    repo_b.mkdir()
    monkeypatch.chdir(repo_a)

    local_record = store.create_session(
        cwd=repo_a,
        model="local-model",
        prompt_text="local",
        prompt_source="inline",
        context_mode="full",
        system_prompt_path=None,
        config_paths=[],
    )
    remote_record = store.create_session(
        cwd=repo_b,
        model="remote-model",
        prompt_text="remote",
        prompt_source="inline",
        context_mode="full",
        system_prompt_path=None,
        config_paths=[],
    )
    save_approval_request(
        Path(local_record.artifact_dir),
        {
            "status": "pending",
            "tool_name": "bash",
            "cmd": "rm -rf build",
            "args_summary": "cmd='rm -rf build'",
            "reason": "destructive file deletion via rm",
        },
    )
    save_approval_request(
        Path(remote_record.artifact_dir),
        {
            "status": "pending",
            "tool_name": "bash",
            "cmd": "rm -rf other",
            "args_summary": "cmd='rm -rf other'",
            "reason": "destructive file deletion via rm",
        },
    )

    with patch("scripts.llm_assist.__main__.SessionStore", return_value=store):
        rc = main(["approve"])

    captured = capsys.readouterr()
    approval = load_approval_request(Path(local_record.artifact_dir))
    assert rc == 0
    assert approval is not None
    assert approval["status"] == "approved"
    assert f"approved: {local_record.session_id}" in captured.out


def test_sessions_marks_active_session_and_current_cwd(tmp_path, capsys, monkeypatch):
    store = SessionStore(tmp_path / "assist-home")
    repo = tmp_path / "repo"
    other = tmp_path / "other"
    repo.mkdir()
    other.mkdir()
    monkeypatch.chdir(repo)

    local_record = store.create_session(
        cwd=repo,
        model="local-model",
        prompt_text="local",
        prompt_source="inline",
        context_mode="full",
        system_prompt_path=None,
        config_paths=[],
    )
    remote_record = store.create_session(
        cwd=other,
        model="remote-model",
        prompt_text="remote",
        prompt_source="inline",
        context_mode="full",
        system_prompt_path=None,
        config_paths=[],
    )
    store.set_active_session(repo, local_record.session_id)
    store.set_active_session(other, remote_record.session_id)
    store.acquire_session_lock(local_record.session_id)

    with patch("scripts.llm_assist.__main__.SessionStore", return_value=store):
        rc = main(["sessions"])

    captured = capsys.readouterr()
    assert rc == 0
    assert "session_id" in captured.out
    assert f"{local_record.session_id}  created" in captured.out
    assert "active,locked,cwd" in captured.out
    assert f"{remote_record.session_id}  created" in captured.out
    assert "active" in captured.out


def test_run_model_resolution_failure_exits_cleanly(tmp_path):
    store = SessionStore(tmp_path / "assist-home")
    work_dir = tmp_path / "work"
    work_dir.mkdir()

    class APIConnectionError(Exception):
        pass

    with patch("scripts.llm_assist.__main__.SessionStore", return_value=store), \
            patch(
                "scripts.llm_assist.__main__.resolve_served_model",
                side_effect=APIConnectionError("boom"),
            ):
        try:
            main([
                "run",
                "--cwd", str(work_dir),
                "--prompt-text", "do it",
            ])
        except SystemExit as exc:
            message = str(exc)
            assert "could not reach the local model server" in message
            assert "/v1/models" in message
        else:
            raise AssertionError("expected SystemExit")


def test_smoke_command_fails_when_repo_not_fixed(tmp_path, capsys):
    store = SessionStore(tmp_path / "assist-home")
    smoke_root = tmp_path / "repo"

    def fake_run_session(store_obj, record, *, resume):
        artifact_dir = Path(record.artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / ".trace.jsonl").write_text(
            json.dumps({
                "event": "session_end",
                "session_number": 1,
                "finish_reason": "stop",
                "turns": 1,
            }) + "\n"
        )
        store_obj.update_session(record.session_id, status="completed", last_finish_reason="stop")
        return True, "stop"

    with patch("scripts.llm_assist.__main__.SessionStore", return_value=store), \
            patch("scripts.llm_assist.__main__.resolve_smoke_model", return_value=("exact-model", ["exact-model"])), \
            patch("scripts.llm_assist.__main__.run_session", side_effect=fake_run_session):
        rc = main([
            "smoke",
            "--root", str(smoke_root),
            "--assist-home", str(tmp_path / "assist-home"),
        ])

    captured = capsys.readouterr()
    assert rc == 1
    assert "smoke acceptance failed" in captured.out
    assert "calc.py does not contain the fixed 'return a + b' body" in captured.out
    assert f"smoke_repo: {smoke_root.resolve()}" in captured.out
    assert "session_id:" in captured.out


def test_smoke_command_succeeds_when_repo_fixed_and_tests_pass(tmp_path, capsys):
    store = SessionStore(tmp_path / "assist-home")
    smoke_root = tmp_path / "repo"

    def fake_run_session(store_obj, record, *, resume):
        artifact_dir = Path(record.artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / ".trace.jsonl").write_text(
            json.dumps({
                "event": "session_end",
                "session_number": 1,
                "finish_reason": "stop",
                "turns": 1,
            }) + "\n"
        )
        # Apply the fix the assistant would have applied.
        (Path(record.cwd) / "calc.py").write_text(
            "def add(a, b):\n"
            "    return a + b\n"
        )
        store_obj.update_session(record.session_id, status="completed", last_finish_reason="stop")
        return True, "stop"

    with patch("scripts.llm_assist.__main__.SessionStore", return_value=store), \
            patch("scripts.llm_assist.__main__.resolve_smoke_model", return_value=("exact-model", ["exact-model"])), \
            patch("scripts.llm_assist.__main__.run_session", side_effect=fake_run_session):
        rc = main([
            "smoke",
            "--root", str(smoke_root),
            "--assist-home", str(tmp_path / "assist-home"),
        ])

    captured = capsys.readouterr()
    assert rc == 0
    assert "smoke acceptance failed" not in captured.out
    assert "status: completed" in captured.out


def test_smoke_command_fails_when_pending_approval_exists(tmp_path, capsys):
    store = SessionStore(tmp_path / "assist-home")
    smoke_root = tmp_path / "repo"

    def fake_run_session(store_obj, record, *, resume):
        artifact_dir = Path(record.artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / ".trace.jsonl").write_text(
            json.dumps({
                "event": "session_end",
                "session_number": 1,
                "finish_reason": "approval_required",
                "turns": 1,
            }) + "\n"
        )
        (Path(record.cwd) / "calc.py").write_text(
            "def add(a, b):\n"
            "    return a + b\n"
        )
        save_approval_request(
            Path(record.artifact_dir),
            {
                "status": "pending",
                "tool_name": "bash",
                "cmd": "rm -rf build",
                "args_summary": "cmd='rm -rf build'",
                "reason": "destructive file deletion via rm",
            },
        )
        store_obj.update_session(record.session_id, status="paused", last_finish_reason="approval_required")
        return False, "approval_required"

    with patch("scripts.llm_assist.__main__.SessionStore", return_value=store), \
            patch("scripts.llm_assist.__main__.resolve_smoke_model", return_value=("exact-model", ["exact-model"])), \
            patch("scripts.llm_assist.__main__.run_session", side_effect=fake_run_session):
        rc = main([
            "smoke",
            "--root", str(smoke_root),
            "--assist-home", str(tmp_path / "assist-home"),
        ])

    captured = capsys.readouterr()
    assert rc == 1
    assert "smoke acceptance failed" in captured.out
    assert "pending approval request" in captured.out


def test_smoke_command_bootstraps_repo_and_runs_session(tmp_path, capsys):
    store = SessionStore(tmp_path / "assist-home")
    smoke_root = tmp_path / "repo"

    def fake_run_session(store_obj, record, *, resume):
        artifact_dir = Path(record.artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / ".trace.jsonl").write_text(
            json.dumps({
                "event": "session_end",
                "session_number": 1,
                "finish_reason": "stop",
                "turns": 3,
            }) + "\n"
        )
        (Path(record.cwd) / "calc.py").write_text(
            "def add(a, b):\n"
            "    return a + b\n"
        )
        store_obj.update_session(record.session_id, status="completed", last_finish_reason="stop")
        return True, "stop"

    with patch("scripts.llm_assist.__main__.SessionStore", return_value=store), \
            patch("scripts.llm_assist.__main__.resolve_smoke_model", return_value=("exact-model", ["exact-model"])), \
            patch("scripts.llm_assist.__main__.run_session", side_effect=fake_run_session):
        rc = main([
            "smoke",
            "--root", str(smoke_root),
            "--assist-home", str(tmp_path / "assist-home"),
        ])

    captured = capsys.readouterr()
    assert rc == 0
    assert f"smoke_repo: {smoke_root.resolve()}" in captured.out
    assert "served_models: exact-model" in captured.out
    assert "status: completed" in captured.out
    assert (smoke_root / "calc.py").exists()
