"""Process facts survive the string-only model-facing tool pipeline."""

from types import SimpleNamespace
from unittest.mock import patch

from _config_helpers import make_config
from scripts.llm_solver.harness._loop.trace_output import build_tool_call_trace_fields
from scripts.llm_solver.harness.tools import dispatch


def test_bash_dispatch_exports_exit_status_and_pre_reminder_hash(tmp_path):
    cfg = make_config(sandbox_bash=False)
    metadata = {}
    with patch(
        "scripts.llm_solver.harness.tools._run_in_sandbox",
        return_value=("", 1, False),
    ):
        result = dispatch(
            "bash",
            {"cmd": "false"},
            cwd=str(tmp_path),
            cfg=cfg,
            execution_metadata=metadata,
        )

    assert "[exit code: 1]" in result
    assert metadata["exit_status_known"] is True
    assert metadata["exit_status"] == 1
    assert metadata["timed_out"] is False
    assert len(metadata["output_sha256"]) == 64

    decorated = result + "\n<system-reminder>Choose a different action.</system-reminder>"
    session = SimpleNamespace(
        cfg=SimpleNamespace(trace_result_summary_chars=1200),
        cwd=str(tmp_path),
        _sink_counter=0,
        _session_number=1,
    )
    fields = build_tool_call_trace_fields(
        session,
        tool_name="bash",
        args_summary="cmd='false'",
        result=decorated,
        turn=3,
        gate_blocked=False,
        execution_metadata=metadata,
    )
    assert fields["exit_status"] == 1
    assert fields["pass_fail"] == "fail"
    assert fields["execution_output_sha256"] == metadata["output_sha256"]
    assert fields["output_sha256"] != fields["execution_output_sha256"]
