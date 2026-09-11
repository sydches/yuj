"""Unknown task facts must not silently select pytest at any fallback site."""
from pathlib import Path

import pytest

from scripts.llm_solver.config import load_config
from scripts.llm_solver.harness._loop._driver_setup import resolve_task_format
from scripts.llm_solver.harness._loop.session_io import _load_bash_transforms
from scripts.llm_solver.language_quirks import _discovery, detect_runner


TREATMENT = Path(__file__).resolve().parents[1] / "configs/regimes/treatment.toml"


def assert_generic_transforms(cfg):
    output_control, _, _, _, _, output_parser = _load_bash_transforms(cfg)
    assert output_control is None
    assert output_parser is None


def test_unresolved_loader_does_not_invent_pytest():
    cfg = load_config(TREATMENT)
    assert cfg.analysis_task_format == "auto"
    assert cfg.bash_transforms_task_format_enabled
    assert_generic_transforms(cfg)


@pytest.mark.parametrize("error", [OSError("unreadable"), RuntimeError("discovery failed")])
def test_driver_discovery_failure_keeps_generic_analysis(tmp_path, monkeypatch, error):
    (tmp_path / "pytest.ini").write_text("[pytest]\n")

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(_discovery, "inspect_runner_candidates", fail)
    cfg = resolve_task_format(load_config(TREATMENT), tmp_path)
    assert cfg.analysis_task_format == "generic"
    assert_generic_transforms(cfg)


@pytest.mark.parametrize("contents", [None, "invalid ini", "x" * 65537])
def test_unusable_declaration_stays_generic_through_all_sites(tmp_path, contents):
    if contents is not None:
        (tmp_path / "pytest.ini").write_text(contents)
    assert detect_runner(tmp_path) == "generic"
    cfg = resolve_task_format(load_config(TREATMENT), tmp_path)
    assert cfg.analysis_task_format == "generic"
    assert_generic_transforms(cfg)


@pytest.mark.parametrize("files,expected,passed_marker", [
    ({"pytest.ini": "[pytest]\n"}, "pytest", "PASSED"),
    ({"go.mod": "module example.org/task\n"}, "go", "--- PASS"),
    ({"Cargo.toml": '[package]\nname="example"\n'}, "cargo", "ok"),
    ({"pytest.ini": "[pytest]\n", "go.mod": "module example.org/task\n"}, "generic", None),
])
def test_recorded_declarations_control_resolution_and_transform_loading(
    tmp_path, monkeypatch, files, expected, passed_marker,
):
    for name, contents in files.items():
        (tmp_path / name).write_text(contents)
    assert detect_runner(tmp_path) == expected
    # Real declaration evidence in a startup-report fixture; runtime
    # availability is deliberately not established by this check.
    report = {"runner_declarations": _discovery.inspect_runner_candidates(tmp_path)}

    def unexpected_read(*args, **kwargs):
        pytest.fail("resolution must consume the supplied observations")

    monkeypatch.setattr(_discovery, "inspect_runner_candidates", unexpected_read)
    cfg = resolve_task_format(load_config(TREATMENT), tmp_path, runtime_observations=report)
    assert cfg.analysis_task_format == expected
    assert cfg.runtime_test_selection == {
        "status": "unavailable", "task_root": str(tmp_path.resolve()),
    }
    output_control, _, _, _, _, _ = _load_bash_transforms(cfg)
    if passed_marker is None:
        assert_generic_transforms(cfg)
    else:
        assert output_control is not None
        assert output_control.passed_marker == passed_marker
