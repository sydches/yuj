import json
from pathlib import Path

from scripts.llm_solver.config import load_config
from scripts.llm_solver.harness.loop import build_resume_prompt_from_trace


def test_build_resume_prompt_from_trace_uses_last_session_actions(tmp_path: Path):
    trace_path = tmp_path / ".trace.jsonl"
    events = [
        {"event": "session_start", "session_number": 1},
        {
            "event": "tool_call",
            "session_number": 1,
            "turn_number": 0,
            "tool_name": "bash",
            "args_summary": "cmd='pytest -q tests/test_app.py'",
            "result_summary": "1 failed",
            "prompt_tokens": 100,
            "completion_tokens": 20,
        },
        {
            "event": "session_end",
            "session_number": 1,
            "finish_reason": "max_turns",
            "turns": 7,
            "total_prompt_tokens": 777,
        },
    ]
    trace_path.write_text("".join(json.dumps(event) + "\n" for event in events))

    cfg = load_config()
    prompt = build_resume_prompt_from_trace(
        trace_path,
        cfg,
        task_description="Fix the failing application test.",
    )

    assert prompt is not None
    assert "Previous session ended after 7 turns: max_turns." in prompt
    assert "bash(cmd='pytest -q tests/test_app.py')" in prompt
    assert "Fix the failing application test." in prompt
