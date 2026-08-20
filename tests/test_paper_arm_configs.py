"""Public contract for the paper settings and released result files."""

import hashlib
from pathlib import Path

import pytest

from scripts.llm_solver.config import PROJECT_ROOT, load_config


CONTROL = PROJECT_ROOT / "configs/regimes/baselines/plain_long_solve.toml"
TREATMENT = PROJECT_ROOT / "configs/regimes/treatment.toml"


@pytest.mark.parametrize(
    ("relative_path", "expected_sha256"),
    [
        (
            "paper/results/task_outcomes.tsv",
            "68ca128d2bc7f1184984a1e6483bd1e5f7b7d8b32a1b11d2944eb47c6f3a5370",
        ),
        (
            "paper/results/task_outcomes.md",
            "48ee4c2e61b7ce4ea02f9c7b22e0b41e165e3549f6d92106a6cd34aa8365eb1c",
        ),
        (
            "paper/provenance/analysis_exclusions.tsv",
            "eea1a8a7a349eb4414268f4c523b4bf12aee6b44fed3c0cae939da62d28c1d51",
        ),
        (
            "paper/provenance/cell_provenance.json",
            "2d8168a8c7d1d81f0a2c8fc51c18cead1d5b3d5cc2398a4b06a0e17ca5dde44b",
        ),
    ],
)
def test_published_paper_files_match_declared_hashes(
    relative_path: str,
    expected_sha256: str,
):
    data = (PROJECT_ROOT / relative_path).read_bytes()
    assert hashlib.sha256(data).hexdigest() == expected_sha256


def test_control_and_treatment_select_the_declared_runtime_packages():
    control = load_config(user_config=CONTROL)
    treatment = load_config(user_config=TREATMENT)

    assert control.max_turns == treatment.max_turns == 250
    assert control.max_sessions == treatment.max_sessions == 1
    assert control.sandbox_bash is treatment.sandbox_bash is True
    assert control.sandbox_required is treatment.sandbox_required is True

    assert control.bash_transforms_universal_enabled is False
    assert control.bash_transforms_task_format_enabled is False
    assert control.bash_quirks_forbidden_enabled is False
    assert control.preflight_reclip_enabled is False
    assert control.adaptive_control_enabled is False
    assert control.llm_hurdle_detector_enabled is False

    assert treatment.bash_transforms_universal_enabled is True
    assert treatment.bash_transforms_task_format_enabled is True
    assert treatment.bash_quirks_forbidden_enabled is True
    assert treatment.preflight_reclip_enabled is True
    assert treatment.adaptive_control_enabled is True
    assert treatment.llm_hurdle_detector_enabled is True
    assert treatment.llm_hurdle_detector_backend == "trace_nets"
    assert treatment.adaptive_control_delivery == "user_turn"
    assert treatment.guardrails_arm_after_turn == 10
    assert treatment.adaptive_control_baseline_config_paths == (
        "configs/regimes/baselines/plain_long_solve.toml",
    )


@pytest.mark.parametrize(
    ("filename", "loop_streak", "reread_gap"),
    [
        ("verified_20480.toml", 5, 7),
        ("verified_43008.toml", 0, 3),
        ("featurebench_47104.toml", 0, 8),
        ("pro_49152.toml", 0, 7),
    ],
)
def test_paper_threshold_overlays(filename: str, loop_streak: int, reread_gap: int):
    threshold = PROJECT_ROOT / "configs/paper/thresholds" / filename
    cfg = load_config(user_config=[TREATMENT, threshold])

    assert cfg.loop_detect_threshold == loop_streak
    assert cfg.trace_nets_reread_min_gap == reread_gap


def test_published_arm_inputs_are_machine_neutral():
    paths = [
        CONTROL,
        TREATMENT,
        *sorted((PROJECT_ROOT / "configs/treatment").rglob("*")),
        *sorted((PROJECT_ROOT / "configs/paper").rglob("*")),
    ]
    for path in paths:
        if not Path(path).is_file():
            continue
        text = Path(path).read_text()
        assert "/home/" not in text
        assert "/mnt/" not in text
        assert "studies/" not in text
