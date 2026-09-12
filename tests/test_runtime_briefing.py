"""Brief only the task's observed setup, independently of language and paths."""
import json

import pytest

from scripts.llm_solver.harness.runtime_briefing import build_runtime_briefing, render_runtime_briefing


@pytest.mark.parametrize("language,runner,executable,command", [
    ("Python", "unittest", "/workspace/env/bin/python", "/workspace/env/bin/python -m unittest"),
    ("Go", "go", "/tools/go", "/tools/go test"),
    ("Rust", "cargo", "/tools/cargo", "/tools/cargo test"),
    ("JavaScript", "npm", "/tools/pnpm", "/tools/pnpm run test"),
])
def test_selected_setup_excludes_unrelated_observations(language, runner, executable, command):
    report = {
        "observations": [
            {"source": "project_file", "status": "observed", "language_hint": language},
            {"source": "executable_locations", "not_found": ["UNRELATED_MISSING_TOOL"]},
            {"source": "task_directory_scan", "files": 999, "sha256": "INVENTORY_HASH"},
            {"source": "conda_environments", "status": "observed",
             "environment_candidates": ["/unrelated/environment"]},
        ],
        "host_root": "/HOST_PATH",
        "docker_host": "DOCKER_ADDRESS",
        "container_id": "CONTAINER_ID",
        "runner_selection": {"status": "selected", "selected": {
            "runner": runner, "executable": executable,
            "base_cmd": command, "version_output": "observed-version",
        }},
    }
    briefing = build_runtime_briefing("/workspace/project", report)
    assert briefing == {
        "working_directory": "/workspace/project", "language": language,
        "test_runner": runner, "test_runner_executable": executable,
        "test_runner_version": "observed-version", "test_command": command,
    }
    rendered = render_runtime_briefing(briefing)
    assert f"Language: {language}" in rendered
    assert f"Run tests with: {command}" in rendered
    assert "HOST_PATH" not in rendered and "CONTAINER_ID" not in rendered


def test_runtime_and_manager_describe_only_the_selected_environment():
    report = {"facts": [{"source": "conda_environments", "status": "observed",
                         "environment_candidates": ["/envs/base", "/envs/task-env"]}],
              "runner_selection": {"status": "selected", "selected": {
                  "runner": "pytest", "language": "Python", "executable": "/envs/task-env/bin/python",
                  "runtime": {"version": "3.9.20", "executable": "/envs/task-env/bin/python",
                              "prefix": "/envs/task-env", "runner_version": "7.4.0"},
                  "base_cmd": "/envs/task-env/bin/python -m pytest"}}}
    briefing = build_runtime_briefing("/work/project", report)
    assert briefing["runtime_version"] == "3.9.20"
    assert briefing["runtime_executable"] == "/envs/task-env/bin/python"
    assert briefing["environment_manager"] == "conda"
    assert briefing["environment_path"] == "/envs/task-env"
    assert briefing["test_runner_version"] == "7.4.0"
    assert "/envs/base" not in json.dumps(briefing)


def test_missing_selection_does_not_invent_a_language_runtime_or_command():
    briefing = build_runtime_briefing("/restricted/subfolder", {
        "runner_selection": {"status": "ambiguous", "candidates": [
            {"runner": "pytest", "executable": "/candidate/python"}]}})
    assert briefing == {"working_directory": "/restricted/subfolder", "test_runner_status": "ambiguous"}
    assert render_runtime_briefing(briefing).endswith("Test runner: ambiguous")
    assert build_runtime_briefing("/restricted/subfolder", None) == {"working_directory": "/restricted/subfolder"}
