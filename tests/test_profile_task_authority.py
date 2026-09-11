"""Model reminders cannot invent task restrictions or execution history."""
import importlib.util
from pathlib import Path

import pytest

from _config_helpers import make_config
from llm_solver.server.client import LlamaClient
from llm_solver.server.profile_loader import load_profile


ROOT = Path(__file__).resolve().parents[1]
PROFILES = tuple(sorted(path.parent.name for path in (ROOT / "profiles").glob("*/profile.toml")
                        if not path.parent.name.startswith('_')))
CLAIMS = (
    "Never modify existing test files", "Pretest verdict is in context",
    "Do not list/read/grep first", "Test suite is the only verdict",
    "Last action must be a test run",
)
TASK = (
    "Update the existing test for the new interface, but preserve LICENSE. "
    "No pretest ran. Inspect source first. Also explain the change in the documentation."
)


@pytest.mark.parametrize("name", PROFILES)
@pytest.mark.parametrize("system", [
    "You are a coding assistant.",
    "Commandments: follow the task.",
    'Explain the quoted word "Commandments".',
])
def test_prepared_requests_preserve_task_authority(name, system):
    profile = load_profile(name, ROOT / "profiles")
    client = LlamaClient(make_config(profile_name=name, model=name), profile)
    messages = [{"role": "system", "content": system}, {"role": "user", "content": TASK}]
    try:
        first = client.prepare_chat_request(messages, [])
        second = client.prepare_chat_request(first["messages"], [])
        for request in (first, second):
            text = "\n".join(str(m.get("content", "")) for m in request["messages"])
            assert TASK in text
            assert not any(claim in text for claim in CLAIMS)
            assert text.count("## Operational rules") <= 1
            if "Commandments" in system:
                assert "completion criteria" in text
                assert "actual pretest results when supplied" in text
    finally:
        client.client.close()


@pytest.mark.parametrize("name", PROFILES)
def test_direct_module_fallback_respects_the_same_authority(name):
    path = ROOT / "profiles" / name / "denormalize/behavioral.py"
    spec = importlib.util.spec_from_file_location("direct_behavioral", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    messages = [{"role": "system", "content": "Commandments: follow the task."},
                {"role": "user", "content": TASK}]
    result = module.apply(messages)
    assert not any(claim in result[0]["content"] for claim in CLAIMS)
    assert "explicit edit restrictions" in result[0]["content"]
    assert result[1]["content"] == TASK
