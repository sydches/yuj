"""Run-parameter capture contract for session.json.

The measurement command writes session.json to the run directory at startup.
The file records: started_at, run_dir, model, context_mode,
system_prompt_path, config_paths (resolved abs paths of every overlay
loaded), config hashes, regime identity when an overlay lives under
configs/regimes, resolved config hash, model_runtime, git_commit/branch,
cli_overrides. Reproducibility hook — when a run regresses, this is the
file that answers "what settings were active here?" without re-deriving them.
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest


def test_session_meta_writes_on_dry_run(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    # Write a stub config so load_config has something to merge.
    overlay_dir = tmp_path / "configs" / "regimes"
    overlay_dir.mkdir(parents=True)
    overlay = overlay_dir / "test_overlay.toml"
    overlay.write_text("[model]\nname = \"test-model\"\n")

    # Run llm_solver in dry-run mode (no llama-server needed).
    cmd = [
        sys.executable, "-m", "scripts.llm_solver", str(run_dir),
        "--dry-run",
        "--config", str(overlay),
        "--context", "compound_selective",
    ]
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=30,
        cwd=str(Path(__file__).resolve().parent.parent),
    )
    # Dry-run may exit 0 or 1 depending on whether prompts were collected;
    # the session.json hook fires before either path.
    session_path = run_dir / "session.json"
    assert session_path.is_file(), (
        f"session.json not written. stdout={result.stdout!r} "
        f"stderr={result.stderr!r}"
    )
    meta = json.loads(session_path.read_text())
    assert "started_at" in meta
    assert meta["harness_version"] == "8.0.14"
    assert "run_dir" in meta
    assert "model" in meta
    assert "context_mode" in meta
    assert meta["context_mode"] == "compound_selective"
    assert meta["transformations"]["halflife_context"] is False
    assert "config_paths" in meta
    # The overlay path we passed must appear, fully resolved.
    assert any(str(overlay.resolve()) == p for p in meta["config_paths"]), (
        f"expected {overlay.resolve()} in {meta['config_paths']}"
    )
    overlay_sha = hashlib.sha256(overlay.read_bytes()).hexdigest()
    assert meta["config_path_hashes"][str(overlay.resolve())] == overlay_sha
    assert meta["config_layers"][0]["sha256"] == overlay_sha
    assert meta["resolved_config_sha256"]
    assert len(meta["resolved_config_sha256"]) == 64
    assert meta["regime"]["name"] == "test_overlay"
    assert meta["regime"]["overlay_sha256"] == overlay_sha
    assert meta["model_runtime"]["wire_model"] == "test-model"
    assert len(meta["model_runtime_sha256"]) == 64
    assert "system_prompt_path" in meta
    assert meta["system_prompt_path"] is None  # we didn't pass --system-prompt
    assert "git_commit" in meta
    assert "git_branch" in meta
    assert "cli_overrides" in meta
