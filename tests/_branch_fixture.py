"""Configuration fixture for runtime branch artifact tests."""
from types import SimpleNamespace


def _cfg(root, baseline):
    return SimpleNamespace(
        adaptive_control_branch_bundle_enabled=True,
        adaptive_control_branch_bundle_root=str(root),
        adaptive_control_branch_bundle_source_run_id="waveX",
        adaptive_control_branch_bundle_max_per_attempt=1,
        adaptive_control_branch_watch_policy_id="prefix_rewind_watch_v1",
        adaptive_control_source_instance_id="",
        adaptive_control_source_run_dir="",
        adaptive_control_detector_version="prefix_detector_v1",
        adaptive_control_policy_version="scout_v1_waveX",
        adaptive_control_watch_window_turns=5,
        adaptive_control_baseline_config_paths=(str(baseline),),
        profile_name="",
        model="fake-model",
    )
