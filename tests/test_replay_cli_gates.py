"""Tests for replay CLI routing and config-parity gates."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from llm_solver.__main__ import main  # noqa: E402


def _source_run(tmp_path: Path, model="qwen-test", mode="compound") -> Path:
    """Minimal valid replay source: session.json + sha-true config layer +
    transcript + trace."""
    run = tmp_path / "source_cell"
    (run / "harness_run" / "transcripts").mkdir(parents=True)
    (run / "host_task").mkdir()
    overlay = tmp_path / "layer.toml"
    overlay.write_text("[loop]\nmax_turns = 3\n")
    sha = hashlib.sha256(overlay.read_bytes()).hexdigest()
    (run / "harness_run" / "session.json").write_text(json.dumps({
        "model": model, "context_mode": mode,
        "config_paths": [str(overlay)],
        "config_path_hashes": {str(overlay): sha}}))
    (run / "harness_run" / "transcripts" / "host_task.log").write_text(
        "=== turn 001 input ===\n{}\n=== turn 001 output ===\n"
        '{"choices": [{"message": {"role": "assistant"}, "finish_reason": "stop"}]}')
    (run / "host_task" / ".trace.jsonl").write_text(json.dumps(
        {"event": "session_start", "session_number": 1,
         "context_contract": {"mode": mode}}) + "\n")
    return run


def test_replay_refuses_user_config(tmp_path):
    src = _source_run(tmp_path)
    overlay = tmp_path / "user.toml"
    overlay.write_text("[loop]\nmax_turns = 9\n")
    with pytest.raises(SystemExit) as e:
        main([str(tmp_path / "rd1"), "--replay-from", str(src),
              "--config", str(overlay)])
    assert e.value.code == 2


def test_replay_refuses_user_model(tmp_path):
    src = _source_run(tmp_path)
    with pytest.raises(SystemExit) as e:
        main([str(tmp_path / "rd2"), "--replay-from", str(src),
              "--model", "qwen3.5-9b"])
    assert e.value.code == 2


def test_replay_refuses_missing_provenance(tmp_path):
    bare = tmp_path / "no_session_json"
    bare.mkdir()
    rc = main([str(tmp_path / "rd3"), "--replay-from", str(bare)])
    assert rc == 2


def test_replay_refuses_drifted_config_layer(tmp_path):
    src = _source_run(tmp_path)
    (tmp_path / "layer.toml").write_text("[loop]\nmax_turns = 4\n")  # drift
    rc = main([str(tmp_path / "rd4"), "--replay-from", str(src)])
    assert rc == 2


def test_replay_refuses_context_mode_mismatch(tmp_path):
    src = _source_run(tmp_path, mode="compound")
    rc = main([str(tmp_path / "rd5"), "--replay-from", str(src),
               "--context", "yuj"])
    assert rc == 2


def test_replay_without_task_is_refused_not_multitask(tmp_path):
    """A replay without --task returns 2 instead of using multi-task mode."""
    src = _source_run(tmp_path)
    rc = main([str(tmp_path / "rd6"), "--replay-from", str(src)])
    assert rc == 2


def test_replay_startup_reaches_client_with_trace(tmp_path, monkeypatch):
    """Startup passes the source trace and recorded mode to ReplayClient."""
    import llm_solver.__main__ as m
    from llm_solver.server import replay_client as rc
    src = _source_run(tmp_path, mode="compound")
    captured = {}
    real_init = rc.ReplayClient.__init__

    def _spy_init(self, transcript_path, stop_turn=0, strict_fidelity=True,
                  source_trace_path=None):
        captured["transcript"] = str(transcript_path)
        captured["trace"] = str(source_trace_path)
        raise RuntimeError("reached_client_build")

    monkeypatch.setattr(rc.ReplayClient, "__init__", _spy_init)
    with pytest.raises(RuntimeError, match="reached_client_build"):
        m.main([str(tmp_path / "rd7"), "--replay-from", str(src),
                "--task", str(tmp_path)])
    assert captured["transcript"].endswith("host_task.log")
    assert captured["trace"].endswith(".trace.jsonl")


def test_replay_extra_config_is_accepted(tmp_path, monkeypatch):
    """The capture seam: measurement overlays ride along with the adopted
    recording config instead of being refused (parity enforced by the
    fidelity gate, not by refusal)."""
    import llm_solver.__main__ as m
    from llm_solver.server import replay_client as rc
    src = _source_run(tmp_path, mode="compound")
    capture = tmp_path / "capture.toml"
    capture.write_text("[adaptive_control]\nbranch_bundle_enabled = true\n"
                       f"branch_bundle_root = \"{tmp_path / 'bundles'}\"\n")

    def _spy_init(self, *a, **k):
        raise RuntimeError("reached_client_build")

    monkeypatch.setattr(rc.ReplayClient, "__init__", _spy_init)
    with pytest.raises(RuntimeError, match="reached_client_build"):
        m.main([str(tmp_path / "rd8"), "--replay-from", str(src),
                "--task", str(tmp_path),
                "--replay-extra-config", str(capture)])
