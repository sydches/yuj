"""047: current execution evidence reaches completion and default guidance."""
from dataclasses import replace
import io
import json
from pathlib import Path
import shlex
import sys
import tomllib
from unittest.mock import MagicMock

import pytest

from _config_helpers import make_config
from llm_solver.harness.loop import Session
from llm_solver.server.types import ToolCall, TurnResult, Usage


def run_completion(root, cfg, *, implicit, checks, stale=False):
    (root / "core.py").write_text("VALUE = 1\n")
    cfg = replace(cfg, max_turns=len(checks) + 3, sandbox_bash=False,
                  sandbox_env_set={**cfg.sandbox_env_set, "PYTHONDONTWRITEBYTECODE": "1"},
                  done_loop_abort_after=0, loop_detect_enabled=False,
                  duplicate_guard_enabled=False, allow_implicit_done=True)
    calls = [ToolCall("write", "write", {"path": "core.py", "content": "VALUE = 2\n"})]
    calls += [ToolCall(str(i), "bash", {"cmd": shlex.quote(sys.executable) + " " + cmd})
              for i, cmd in enumerate(checks)]
    requests = []

    def chat(messages, *args, **kwargs):
        index = len(requests)
        requests.append(json.loads(json.dumps(messages)))
        if index == len(calls) and stale:
            # An external source edit after the passing check. Flags alone
            # remain true; completion must consult the recorded revisions.
            assert session._guards.verified_since_mutation
            (root / "core.py").write_text("VALUE = 9\n")
        tools = ([calls[index]] if index < len(calls) else
                 [] if implicit else [ToolCall("done", "done", {"summary": "finished"})])
        return TurnResult(content=None if tools else "Finished", tool_calls=tools,
                          finish_reason="tool_calls" if tools else "stop",
                          usage=Usage(prompt_tokens=10, completion_tokens=5))

    client = MagicMock()
    client.chat.side_effect = chat
    client.build_assistant_message.return_value = {"role": "assistant", "content": None}
    trace = io.StringIO()
    session = Session(cfg, client, "Follow the task.", "Set VALUE to 2 and check it.",
                      str(root), trace_file=trace)
    result = session.run()
    return result, session._guards, requests, [json.loads(line) for line in trace.getvalue().splitlines()]


@pytest.mark.parametrize("implicit", [False, True])
@pytest.mark.parametrize("case,enabled,accepted", [
    ("failed", True, False), ("passed", True, True),
    ("custom_only", True, True),
    ("stale", True, False), ("failed", False, True),
])
def test_completion_consults_current_evidence(tmp_path, implicit, case, enabled, accepted):
    (tmp_path / "test_arithmetic.py").write_text(
        "import unittest\nclass Arithmetic(unittest.TestCase):\n"
        "    def test_sum(self): self.assertEqual(1 + 1, 2)\n")
    expected = 3 if case == "failed" else 2
    checks = ["-m unittest test_arithmetic -q", f"-c 'import core; assert core.VALUE == {expected}'"]
    if case == "custom_only":
        checks = checks[1:]
    cfg = make_config(done_guard_enabled=False,
                      post_mutation_verification_gate_after=3 if enabled else 0)
    result, state, _, rows = run_completion(tmp_path, cfg, implicit=implicit,
                                           checks=checks, stale=case == "stale")
    assert state.formal_verification_passed_since_mutation is (case != "custom_only")
    assert state.verified_since_mutation is (case != "failed")
    assert result.done is accepted
    assert any(row.get("verification_status") == "passed" for row in rows) is (case != "custom_only")
    assert any(row.get("verification_status") == ("custom_failed" if case == "failed" else "custom_passed")
               for row in rows)


@pytest.mark.parametrize("implicit", [False, True])
@pytest.mark.parametrize("heuristic,accepted", [(False, True), (True, False)])
def test_unavailable_target_waiver_preserves_independent_guard(tmp_path, implicit, heuristic, accepted):
    cfg = make_config(done_guard_enabled=heuristic, post_mutation_verification_gate_after=3)
    result, state, _, _ = run_completion(tmp_path, cfg, implicit=implicit,
        checks=["-c 'import core; assert core.VALUE == 3'"] * 3)
    assert state.post_mutation_automatic_verification_unavailable
    assert not state.verified_since_mutation
    assert result.done is accepted


@pytest.mark.parametrize("explicit", [False, True])
def test_public_loader_prompts_and_delivered_rejection(tmp_path, monkeypatch, explicit):
    from llm_solver import config
    from llm_solver._shared.paths import default_config_path, local_config_path

    fields = ("post_mutation_verification_nudge", "post_mutation_verification_gate",
              "done_reject_no_formal_verification")
    root = Path(__file__).resolve().parents[1]
    source = (root / "config.toml").read_text()
    defaults = tomllib.loads(source)["prompts"]
    expected = {key: f"Owner-selected {key}" if explicit else defaults[key] for key in fields}
    source = source.replace('patterns_file = "security/patterns.toml"',
                            f'patterns_file = "{root / "security/patterns.toml"}"')
    lines = []
    for line in source.splitlines(keepends=True):
        key = line.split(" = ")[0]
        if key in fields:
            if explicit:
                lines.append(f'{key} = {json.dumps(expected[key])}\n')
        else:
            lines.append(line)
    alternate = tmp_path / "alternate.toml"
    alternate.write_text("".join(lines))
    monkeypatch.setenv("YUJ_CONFIG", str(alternate))
    monkeypatch.setenv("YUJ_CONFIG_LOCAL", str(tmp_path / "absent.toml"))
    # The suite imported config earlier. Resolve the normal startup paths
    # with this environment, as a fresh loader process would.
    monkeypatch.setattr(config, "_DEFAULT_CONFIG", default_config_path())
    monkeypatch.setattr(config, "_LOCAL_CONFIG", local_config_path())
    loaded = config.load_config()
    assert {key: getattr(loaded, key) for key in fields} == expected
    task = tmp_path / "task"
    task.mkdir()
    cfg = make_config(done_guard_enabled=False, post_mutation_verification_gate_after=3,
                      **{key: getattr(loaded, key) for key in fields})
    result, _, requests, _ = run_completion(task, cfg, implicit=True,
                                            checks=["-c 'import core; assert core.VALUE == 3'"])
    assert not result.done
    assert any(expected["done_reject_no_formal_verification"] in str(message.get("content"))
               for message in requests[-1])
