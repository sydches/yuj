"""Accepted CLI spellings retain context-policy authority before execution."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.llm_solver import config, __main__ as measurement
from scripts.llm_assist import __main__ as assistant
from test_replay_cli_gates import _source_run

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def offline_cli(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "_LOCAL_CONFIG", tmp_path / "absent.toml")
    monkeypatch.setattr(measurement, "_build_client", lambda *a, **k: pytest.fail("client construction"))
    monkeypatch.setattr(measurement, "_build_run_metadata", lambda **kwargs: {})
    monkeypatch.setattr(measurement, "_write_session_json", lambda *a, **k: None)


@pytest.mark.parametrize("arm,options,expected", [
    ("a09", [], "halflife"),
    ("a09", ["--context", "halflife"], "halflife"),
    ("a09", ["--conte=halflife"], "halflife"),
    ("a09", ["--context", "full"], "reject"),
    ("a09", ["--context=full"], "reject"),
    ("a09", ["--conte", "full"], "reject"),
    ("a09", ["--cont", "compact"], "reject"),
    ("a09", ["--conte=full"], "reject"),
    ("a00", [], "full"),
    ("a00", ["--cont", "full"], "full"),
    ("a00", ["--cont", "halflife"], "reject"),
    (None, [], "full"),
    (None, ["--cont", "compact"], "compact"),
])
def test_parsed_choice_controls_transformation_conflicts(tmp_path, capsys, arm, options, expected):
    args = [str(tmp_path / "run"), "--task", str(tmp_path), "--dry-run", *options]
    if arm:
        args += ["--config", str(ROOT / f"configs/transformation_screen/arms/{arm}.toml")]
    if expected == "reject":
        with pytest.raises(SystemExit) as error:
            measurement.main(args)
        assert error.value.code == 2
        assert "conflicts with transformations.halflife_context=" in capsys.readouterr().err
    else:
        assert measurement.main(args) == 0
        assert f"Context: {expected}\n" in capsys.readouterr().out


@pytest.mark.parametrize("options,accepted", [
    ([], True), (["--conte=halflife"], True),
    (["--context", "full"], False), (["--conte", "full"], False),
])
def test_replay_adopts_only_an_omitted_context(tmp_path, capsys, options, accepted):
    source = _source_run(tmp_path, mode="halflife")
    result = measurement.main([str(tmp_path / "run"), "--task", str(tmp_path),
                               "--dry-run", "--replay-from", str(source), *options])
    assert result == (0 if accepted else 2)
    if accepted:
        assert "Context: halflife\n" in capsys.readouterr().out


@pytest.mark.parametrize("treatment,expected", [(True, "halflife"), (False, "full")])
def test_assistant_package_default_stays_separate(treatment, expected):
    args = SimpleNamespace(treatment=treatment, context=None, config=[])
    assert assistant._effective_run_settings(args)[1] == expected
