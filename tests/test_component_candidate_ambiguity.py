"""Candidate location cannot supply test identity or source relevance."""
from pathlib import Path
from dataclasses import replace
import json
import tomllib

import pytest

from _config_helpers import make_config
from scripts.llm_solver.harness._guardrails.state import GuardrailState
from scripts.llm_solver.harness._guardrails.test_inspection import observe_test_file_read
from scripts.llm_solver.harness._guardrails.verification import (
    _select_unique_candidate, resolve_component_verification_target,
)
from scripts.llm_solver.harness.tools import dispatch


@pytest.mark.parametrize("source,candidates", [
    ("scope/core.py", ["scope/checks/test_core.py", "scope/latest/test_core.py"]),
    ("scope/core.py", ["scope/checks/test_core.py", "scope/tests/test_core.py"]),
    ("scope/component/core.py", ["scope/component/test_core.py", "scope/checks/test_core.py"]),
])
def test_names_and_proximity_do_not_resolve_distinct_candidates(source, candidates):
    assert _select_unique_candidate(Path(source), [Path(p) for p in candidates]) is None


def test_production_resolver_defers_pytest_membership_to_native_collection(tmp_path):
    source = tmp_path / "component" / "core.py"
    source.parent.mkdir()
    source.write_text("value = 1\n")
    near = source.parent / "test_core.py"
    near.write_text("def test_value(): pass\n")
    other = tmp_path / "checks" / "test_core.py"
    other.parent.mkdir()
    other.write_text("def test_value(): pass\n")
    state = GuardrailState(post_mutation_source_paths=("component/core.py",))
    pending = resolve_component_verification_target(state, tmp_path, runner="pytest")
    assert pending.path == "" and pending.source_path == "component/core.py"
    other.unlink()
    candidate = resolve_component_verification_target(state, tmp_path, runner="pytest")
    assert candidate is not None
    assert candidate.path == ""  # Uniqueness is checked by the runner, not this scan.


def test_duplicate_entries_do_not_create_a_second_candidate():
    candidate = Path("scope/checks/test_core.py")
    assert _select_unique_candidate(Path("scope/core.py"), [candidate, candidate]) == candidate


@pytest.mark.parametrize("path", ["checks/checks_core.py", "tests/readme.txt"])
def test_read_receipts_record_excerpts_without_declaring_test_identity(tmp_path, path):
    target = tmp_path / path
    target.parent.mkdir()
    target.write_text("fixture content\n")
    cfg = make_config(test_read_warn_after=1, sandbox_bash=False)
    state = GuardrailState()
    facts = {}
    result = dispatch("read", {"path": path}, cwd=str(tmp_path), cfg=cfg,
                      execution_metadata=facts)
    observe_test_file_read(state, cfg, tc_name="read", tc_args={"path": path},
                           result=result, gate_blocked=False, execution_metadata=facts)
    assert state.inspected_files[str(target)]["intervals"] == [(1, 1)]
    assert state.test_file_reads == set()


@pytest.mark.parametrize("path", [
    "checks/checks_core.py", "tests/test_core.py", "tests/readme.txt",
])
def test_contract_allows_focused_inspection_without_test_name_inference(tmp_path, path):
    from scripts.llm_solver.harness._guardrails.checks_pre import contract_gate
    from scripts.llm_solver.harness._guardrails.checks_post import observe_contract_state
    from scripts.llm_solver.harness._guardrails.state import Action

    target = tmp_path / path
    target.parent.mkdir()
    target.write_text("fixture contents\n")
    cfg = make_config(sandbox_bash=False, contract_commit_block_after=1)
    state = GuardrailState(commit_pending=True, commit_source_path="source.py")
    decision = contract_gate(state, cfg, tc_name="read", tc_args={"path": path},
                             focus_key="file:" + path, focus_display=path)
    assert decision.action == Action.PASS
    facts = {}
    result = dispatch("read", {"path": path}, cwd=str(tmp_path), cfg=cfg,
                      execution_metadata=facts)
    observe_contract_state(state, cfg, tc_name="read", tc_args={"path": path},
                           gate_blocked=False, result=result, execution_metadata=facts,
                           focus_key="file:" + path, focus_display=path)
    assert state.commit_pending
    assert state.commit_source_path == path
    assert not state.test_file_reads  # No collection membership was observed.
    broad = contract_gate(state, cfg, tc_name="bash", tc_args={"cmd": "ls -R ."})
    assert broad.action == Action.BLOCK


@pytest.mark.parametrize("explicit", [False, True])
def test_public_loader_contract_guidance(tmp_path, monkeypatch, explicit):
    from scripts.llm_solver import config
    from scripts.llm_solver._shared.paths import default_config_path, local_config_path
    from scripts.llm_solver.harness._guardrails.checks_pre import contract_gate
    from scripts.llm_solver.harness._guardrails.state import Action

    fields = ("contract_commit_warn", "contract_commit_block")
    root = Path(__file__).resolve().parents[1]
    source = (root / "config.toml").read_text()
    defaults = tomllib.loads(source)["prompts"]
    expected = {key: f"Caller text for {key}: {{source}}" if explicit else defaults[key]
                for key in fields}
    source = source.replace('patterns_file = "security/patterns.toml"',
                            f'patterns_file = "{root / "security/patterns.toml"}"')
    lines = []
    for line in source.splitlines(keepends=True):
        key = line.split(" = ")[0]
        if key not in fields:
            lines.append(line)
        elif explicit:
            lines.append(f'{key} = {json.dumps(expected[key])}\n')
    alternate = tmp_path / "base.toml"
    alternate.write_text("".join(lines))
    monkeypatch.setenv("YUJ_CONFIG", str(alternate))
    monkeypatch.setenv("YUJ_CONFIG_LOCAL", str(tmp_path / "absent.toml"))
    monkeypatch.setattr(config, "_DEFAULT_CONFIG", default_config_path())
    monkeypatch.setattr(config, "_LOCAL_CONFIG", local_config_path())
    cfg = replace(config.load_config(), contract_commit_warn_after=1,
                  contract_commit_block_after=2)
    assert {key: getattr(cfg, key) for key in fields} == expected
    assert {key: getattr(make_config(), key) for key in fields} == {
        key: defaults[key] for key in fields}
    state = GuardrailState(commit_pending=True, commit_source_path="checks/custom.py")
    focused = contract_gate(state, cfg, tc_name="read",
                            tc_args={"path": "checks/custom.py"})
    assert focused.action == Action.PASS
    warning = contract_gate(state, cfg, tc_name="bash", tc_args={"cmd": "ls -R ."})
    assert warning.action == Action.WARN
    assert warning.text == expected[fields[0]].format(source="checks/custom.py")
    blocked = contract_gate(state, cfg, tc_name="bash", tc_args={"cmd": "ls -R ."})
    assert blocked.action == Action.BLOCK
    assert blocked.text == expected[fields[1]].format(source="checks/custom.py")
