"""A bundle digest must not hide which task entries its copier omitted."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from _branch_fixture import _cfg
from llm_solver.harness.adaptive_control import branch_bundle
from llm_solver.harness.context import FullTranscript
from llm_solver.harness.guardrails import GuardrailState


def test_capture_retains_permitted_runner_state_and_ordinary_fixtures(tmp_path, monkeypatch):
    repo = tmp_path / "task"
    files = {
        "source.py": "source",
        ".pytest_cache/v/cache/lastfailed": "runner continuation state",
        ".tox/bin/python": "environment fixture",
        "fixtures/.pytest_cache/expected.txt": "ordinary fixture",
        "fixtures/example.pyc": "ordinary fixture",
        ".cache/jest/state.json": "another runner's state",
        ".tool_output/fixture.pyc": "output fixture",
    }
    for name, content in files.items():
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    baseline = tmp_path / "baseline.toml"
    baseline.write_text("[loop]\nmax_turns = 2\n")
    context = FullTranscript()
    context.add_user("fixture")
    session = SimpleNamespace(cfg=_cfg(tmp_path / "bundles", baseline),
        cwd=str(repo), instance_id="fixture-task", attempt_id="fixture-attempt",
        context=context, _guards=GuardrailState(), _trace_events=[],
        _session_number=1, _trace_path=tmp_path / "trace", _state_path=tmp_path / "state")
    decision = SimpleNamespace(diagnosis_status="active_confirmed", active_hurdle_mode="fixture",
        detector_id="fixture", detector_status="active_confirmed", basis_refs=[])
    monkeypatch.setattr(branch_bundle, "_git_commit", lambda: "fixture")
    result = branch_bundle.maybe_capture(session, decision, 1, "fixture")
    assert result["status"] == "created"
    bundle = Path(result["path"])
    report_path = bundle / "snapshot_exclusions.json"
    report = json.loads(report_path.read_text())
    assert report["retention_status"] == "unverified"
    assert report["policy"] == "preserve_permitted_entries_v1"
    assert report["exclusions"] == []
    assert sorted(p.relative_to(bundle / "repo_snapshot").as_posix()
                  for p in (bundle / "repo_snapshot").rglob("*") if p.is_file()) == sorted(files)
    for name, content in files.items():
        assert (bundle / "repo_snapshot" / name).read_text() == content
    assert (bundle / "sunk_outputs/fixture.pyc").read_text() == "output fixture"
    manifest = json.loads((bundle / "branch_manifest.json").read_text())
    assert manifest["snapshot_retention_status"] == "unverified"
    assert manifest["snapshot_exclusions_path"] == report_path.name
    integrity = json.loads((bundle / "integrity.json").read_text())
    assert integrity["file_sha256"][report_path.name] == hashlib.sha256(report_path.read_bytes()).hexdigest()
