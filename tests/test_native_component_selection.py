"""Automatic candidates come from the runner's effective collection."""
import hashlib
import os
from pathlib import Path
import shlex
import shutil
import sys

import pytest

from _config_helpers import make_config
from scripts.llm_solver.harness._guardrails.state import GuardrailState
from scripts.llm_solver.harness._guardrails.verification import resolve_component_verification_target
from scripts.llm_solver.harness.tools import dispatch


@pytest.mark.parametrize("sandbox", [False, True])
@pytest.mark.parametrize("case,expected", [
    ("default", "tests/test_core.py"),
    ("custom_config", "checks/verify_core.py"),
    ("config_subdirectory", "tests/test_core.py"),
    ("changed_config", "checks/verify_core.py"),
    ("hook", None),
    ("ambiguous", None),
    ("multiple_sources", "tests/test_core.py"),
])
def test_native_collection_controls_the_actual_component_run(tmp_path, case, expected, sandbox):
    if sandbox and not shutil.which("bwrap"):
        pytest.skip("requires bwrap")
    (tmp_path / "core.py").write_text("VALUE = 2\n")
    for directory in ("tests", "checks"):
        (tmp_path / directory).mkdir()
        (tmp_path / directory / "__init__.py").write_text("")
    (tmp_path / "tests/test_core.py").write_text("def test_decoy(): assert True\n")
    (tmp_path / "checks/verify_core.py").write_text("def test_declared(): assert False\n")
    config = tmp_path / "pytest.ini"
    config.write_text("[pytest]\ntestpaths = tests\npython_files = test_*.py\n")
    command = shlex.join([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"])
    source_paths = ("core.py",)
    if case == "multiple_sources":
        (tmp_path / "first.py").write_text("VALUE = 0\n")
        source_paths = ("first.py", "core.py")
    state = GuardrailState(post_mutation_source_paths=source_paths)
    target = resolve_component_verification_target(state, tmp_path, runner="pytest")
    assert target.path == ""  # No conventional decoy is forced before collection.
    if case == "custom_config":
        (tmp_path / "custom.ini").write_text("[pytest]\ntestpaths = checks\npython_files = verify_*.py\n")
        command += " -c custom.ini"
    elif case == "config_subdirectory":
        (tmp_path / "cfg").mkdir()
        config = tmp_path / "cfg/pytest.ini"
        config.write_text("[pytest]\ntestpaths = ../tests\n")
        command += " -c cfg/pytest.ini"
        (tmp_path / "tests/test_core.py").write_text(
            'from pathlib import Path\ndef test_body(): Path("body_ran").write_text("yes")\n')
    elif case == "changed_config":
        config.write_text("[pytest]\ntestpaths = checks\npython_files = verify_*.py\n")
    elif case == "hook":
        (tmp_path / "conftest.py").write_text("collect_ignore = ['tests/test_core.py']\n")
    elif case == "ambiguous":
        config.write_text("[pytest]\ntestpaths = tests checks\n")
        (tmp_path / "checks/test_core.py").write_text("def test_other(): assert False\n")
    cfg = make_config(
        sandbox_bash=sandbox, sandbox_required=sandbox,
        sandbox_env_set={"PATH": str(Path(sys.executable).parent) + os.pathsep + os.defpath},
        tools_run_tests_enabled=True, analysis_task_format="pytest",
        runtime_test_selection={"status": "selected", "task_root": str(tmp_path),
                                "selected": {"runner": "pytest", "base_cmd": command}},
    )
    facts = {}
    dispatch("run_tests", {"_component_source": list(target.native_sources)},
             cwd=str(tmp_path), cfg=cfg, execution_metadata=facts)
    selection = facts["runner_request"]["component_selection"]
    assert facts["runner_request"]["command_sha256"] == hashlib.sha256(command.encode()).hexdigest()
    if expected:
        assert selection["status"] == "selected"
        assert selection["source"] == "core.py"
        assert selection["association"] == "naming_hint"
        assert selection["candidates"] == [expected]
        assert selection["path_root"] == str(tmp_path)
        assert selection["collection_root"] == str(config.parent)
        assert selection["items"]
        if case != "config_subdirectory":
            assert all(item.startswith(expected + "::") for item in selection["items"])
        else:
            assert (tmp_path / "body_ran").read_text() == "yes"
        assert facts["exit_status"] == (0 if case in {"default", "multiple_sources", "config_subdirectory"} else 1)
    else:
        assert selection["status"] == ("ambiguous" if case == "ambiguous" else "no_candidate")
        assert selection["items"] == []
        assert facts["verification_status"] != "passed"
    assert not list((tmp_path / ".tool_output").glob("yuj_component_*"))
