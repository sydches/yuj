"""Acceptance coverage for issue #14's sequential ``task`` subagents."""
from __future__ import annotations

import hashlib
import io
import json
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "tests"))

from _config_helpers import make_config
from llm_solver.config import load_config
from llm_solver.harness._loop.model_role_runtime import build_model_role_runtime
from llm_solver.harness._loop.model_roles import RoleTokenLedger
from llm_solver.harness._loop.profile_resolution import apply_profile_to_schemas
from llm_solver.harness.loop import Session, TaskSpec, solve_task
from llm_solver.harness.schemas import get_tool_schemas
from llm_solver.harness.subagents import (
    AgentConfigError,
    SubagentModelBinding,
    SubagentRuntime,
    load_agent_spec,
    prepare_readonly_bash,
)
from llm_solver.harness.subagent_workspace import (
    APPLICATION_FILE,
    CHANGESET_FILE,
    MAX_CHANGESET_INPUT_BYTES,
    PATCH_FILE,
)
from llm_solver.harness.tool_policy import permission_match_argument
from llm_solver.harness.tool_specs import PARALLEL_READ_SAFE_TOOL_NAMES
from llm_solver.harness.tools import dispatch
from llm_solver.server.replay_client import ReplayClient
from llm_solver.server.types import ToolCall, TurnResult, Usage


class ScriptedClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.tool_surfaces: list[tuple[str, ...]] = []
        self.cfg = None

    def chat(self, messages, tools, turn=0):
        self.tool_surfaces.append(
            tuple(item["function"]["name"] for item in tools)
        )
        return self.responses.pop(0)

    def build_assistant_message(self, content, tool_calls):
        message = {"role": "assistant", "content": content}
        if tool_calls:
            message["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments),
                    },
                }
                for call in tool_calls
            ]
        return message

    def query_server_context(self):
        return None

    def set_transcript(self, path, append=False):
        self.transcript_path = path

    def close_transcript(self):
        return None


def _stop(text: str, prompt: int, completion: int) -> TurnResult:
    return TurnResult(
        content=text,
        tool_calls=[],
        finish_reason="stop",
        usage=Usage(prompt, completion),
    )


def _calls(*calls: ToolCall, prompt: int = 3, completion: int = 2) -> TurnResult:
    return TurnResult(
        content="investigating",
        tool_calls=list(calls),
        finish_reason="tool_calls",
        usage=Usage(prompt, completion),
    )


def _write_agent(
    root: Path,
    *,
    name: str = "probe",
    tools: tuple[str, ...] = ("read", "bash", "done"),
    read_only: bool = True,
    max_turns: int = 5,
    workspace: str | None = None,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "prompts").mkdir(exist_ok=True)
    (root / "prompts" / f"{name}.md").write_text("Return one exact finding.\n")
    rendered_tools = ", ".join(json.dumps(tool) for tool in tools)
    (root / f"{name}.toml").write_text(
        "[agent]\n"
        'model_profile = "_base"\n'
        f"tools = [{rendered_tools}]\n"
        f'system_prompt_file = "prompts/{name}.md"\n'
        f"max_turns = {max_turns}\n"
        f"read_only = {'true' if read_only else 'false'}\n"
        + (f'workspace = "{workspace}"\n' if workspace is not None else "")
    )
    return root


def _git(repo: Path, *args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if check and result.returncode != 0:
        raise AssertionError(result.stderr)
    return result.stdout.strip()


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "project"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "config", "user.email", "test@example.invalid")
    (repo / "app.py").write_text("value = 1\n")
    _git(repo, "add", "app.py")
    _git(repo, "commit", "-qm", "base")
    return repo


def _task_cfg(**overrides):
    values = {
        "tools_task_enabled": True,
        "tools_subagent_depth": 1,
        "tools_subagent_max_turns": 8,
        "tools_unified_envelope_enabled": False,
        "sandbox_bash": False,
        "max_turns": 4,
        "max_sessions": 1,
    }
    values.update(overrides)
    return make_config(**values)


def _parent_session(
    tmp_path: Path,
    *,
    child_client: ScriptedClient,
    agents_dir: Path,
    trace_file=None,
    client=None,
    run_root: Path | None = None,
    cfg=None,
):
    cfg = cfg or _task_cfg()
    parent_client = client or ScriptedClient([])

    def factory(parent, spec, task_id):
        return SubagentModelBinding(
            client=child_client,
            config=parent.cfg,
            dedicated=True,
        )

    runtime = SubagentRuntime(
        run_root or (tmp_path / "run"),
        agents_dir=agents_dir,
        client_factory=factory,
    )
    trace = trace_file or io.StringIO()
    session = Session(
        cfg,
        parent_client,
        "system",
        "parent task",
        str(tmp_path),
        trace_file=trace,
        trace_path=(run_root or (tmp_path / "run")) / ".trace.jsonl",
        artifact_dir=run_root or (tmp_path / "run"),
        subagent_runtime=runtime,
    )
    return cfg, session, runtime, trace


def test_public_config_defaults_overlay_validation_and_tool_gate(tmp_path):
    defaults = load_config()
    assert defaults.tools_task_enabled is False
    assert defaults.tools_subagent_depth == 1
    assert defaults.tools_subagent_max_turns == 20

    overlay = tmp_path / "subagents.toml"
    overlay.write_text(
        "[tools]\ntask_enabled = true\nsubagent_depth = 2\n"
        "subagent_max_turns = 7\n"
    )
    configured = load_config(user_config=overlay)
    assert configured.tools_task_enabled is True
    assert configured.tools_subagent_depth == 2
    assert configured.tools_subagent_max_turns == 7

    for body, match in (
        ('task_enabled = "yes"', "task_enabled"),
        ("subagent_depth = -1", "subagent_depth"),
        ("subagent_depth = true", "subagent_depth"),
        ("subagent_max_turns = 0", "subagent_max_turns"),
    ):
        overlay.write_text(f"[tools]\n{body}\n")
        with pytest.raises(ValueError, match=match):
            load_config(user_config=overlay)

    def names(cfg):
        schemas = apply_profile_to_schemas(
            get_tool_schemas(), cfg, object()
        )
        return {item["function"]["name"] for item in schemas}

    gated = {"task", "subagent_changes", "apply_subagent"}
    assert not gated & names(make_config(tools_task_enabled=False))
    assert gated <= names(make_config(tools_task_enabled=True))
    assert not gated & PARALLEL_READ_SAFE_TOOL_NAMES
    assert permission_match_argument(
        "task", {"agent": "research", "prompt": "private request"}
    ) == ("agent", "research")
    assert permission_match_argument(
        "apply_subagent", {"task_id": "task-000001"}
    ) == ("task_id", "task-000001")


def test_agent_descriptor_is_read_only_by_default_and_fail_closed(tmp_path):
    agents = _write_agent(tmp_path / "agents")
    descriptor = agents / "probe.toml"
    descriptor.write_text(descriptor.read_text().replace("read_only = true\n", ""))
    assert load_agent_spec("probe", agents).read_only is True

    for name, mutation_tool in (
        ("unsafe-write", "write"),
        ("unsafe-udiff", "udiff"),
        ("unsafe-cell", "exec_cell"),
    ):
        _write_agent(agents, name=name, tools=("read", mutation_tool))
        with pytest.raises(
            AgentConfigError, match="read-only agent cannot allow"
        ):
            load_agent_spec(name, agents)
    with pytest.raises(AgentConfigError, match="agent must start"):
        load_agent_spec("../probe", agents)
    _write_agent(
        agents,
        name="isolated-reader",
        workspace="isolated",
    )
    with pytest.raises(AgentConfigError, match="must use workspace='shared'"):
        load_agent_spec("isolated-reader", agents)
    _write_agent(
        agents,
        name="bad-workspace",
        read_only=False,
        workspace="remote",
    )
    with pytest.raises(AgentConfigError, match="workspace must be"):
        load_agent_spec("bad-workspace", agents)
    assert prepare_readonly_bash("pwd") == ("command -p pwd", "")
    assert prepare_readonly_bash("touch changed.txt")[0] is None
    assert prepare_readonly_bash("file -C")[0] is None


def test_subagent_allowlist_survives_deferred_tool_loading(tmp_path):
    cfg = _task_cfg(
        tools_lazy_loading_enabled=True,
        tools_active_default=("read", "done"),
    )
    session = Session(
        cfg,
        ScriptedClient([]),
        "system",
        "child task",
        str(tmp_path),
        tool_allowlist=frozenset({"read", "load_tools", "done"}),
        subagent_level=1,
    )

    assert set(session._tool_surface.registered_names) == {
        "read", "load_tools", "done",
    }
    assert "write" not in session.active_tool_names
    result = dispatch(
        "load_tools",
        {"names": ["write"]},
        cwd=str(tmp_path),
        cfg=cfg,
        tool_registry=session._tool_registry,
    )
    assert "unavailable in the current config/profile: write" in result
    assert "write" not in session.active_tool_names


def test_child_trace_parent_summary_and_token_accounting(tmp_path):
    agents = _write_agent(tmp_path / "agents", max_turns=50)
    child = ScriptedClient([_stop("exact child result", 11, 4)])
    cfg, parent, runtime, trace = _parent_session(
        tmp_path, child_client=child, agents_dir=agents
    )

    result = dispatch(
        "task",
        {"agent": "probe", "prompt": "Find the exact owner."},
        cwd=str(tmp_path),
        cfg=cfg,
        tool_registry=parent._tool_registry,
    )
    assert result == "exact child result"
    assert set(child.tool_surfaces[0]) == {"read", "bash", "done"}

    child_trace = tmp_path / "run" / "subagents" / "task-000001" / ".trace.jsonl"
    child_events = [json.loads(line) for line in child_trace.read_text().splitlines()]
    assert child_events[0]["event"] == "subagent_start"
    terminal = child_events[-1]
    assert terminal["event"] == "subagent_result"
    assert terminal["result"] == result
    assert terminal["prompt_tokens"] == 11
    assert terminal["completion_tokens"] == 4
    assert child_events[0]["max_turns"] == 8

    parent_events = [json.loads(line) for line in trace.getvalue().splitlines()]
    summary = parent_events[-1]
    assert summary == {
        "event": "subagent",
        "trace_schema_version": 2,
        "session_number": 0,
        "turn_number": 0,
        "id": "task-000001",
        "agent": "probe",
        "turns": 1,
        "tokens": 15,
        "result_chars": len(result),
    }
    assert "result" not in summary
    assert parent._subagent_prompt_tokens == 11
    assert parent._subagent_completion_tokens == 4
    assert runtime.metrics_payload() == {
        "calls": 1,
        "prompt_tokens": 11,
        "completion_tokens": 4,
        "total_tokens": 15,
    }


def test_subagent_does_not_inherit_primary_advisor(tmp_path):
    agents = _write_agent(tmp_path / "agents")
    child = ScriptedClient([_stop("exact child result", 11, 4)])
    cfg, parent, _runtime, _trace = _parent_session(
        tmp_path,
        child_client=child,
        agents_dir=agents,
        cfg=_task_cfg(advisor_enabled=True),
    )

    result = dispatch(
        "task",
        {"agent": "probe", "prompt": "Find the exact owner."},
        cwd=str(tmp_path),
        cfg=cfg,
        tool_registry=parent._tool_registry,
    )

    assert result == "exact child result"
    assert child.cfg.advisor_enabled is False
    assert not (
        tmp_path / "run" / "subagents" / "task-000001" / "advisor.jsonl"
    ).exists()


def test_isolated_writer_returns_reviewable_patch_then_applies_explicitly(tmp_path):
    repo = _repo(tmp_path)
    agents = _write_agent(
        tmp_path / "agents",
        tools=("read", "write", "done"),
        read_only=False,
        workspace="isolated",
    )
    child = ScriptedClient([
        _calls(ToolCall("write-1", "write", {
            "path": "app.py", "content": "value = 2\n",
        })),
        _stop("Changed app.py in the isolated workspace.", 11, 4),
    ])
    run_root = tmp_path / "run"
    cfg, parent, _runtime, _trace = _parent_session(
        repo,
        child_client=child,
        agents_dir=agents,
        run_root=run_root,
    )

    result = dispatch(
        "task",
        {"agent": "probe", "prompt": "Change the value."},
        cwd=str(repo),
        cfg=cfg,
        tool_registry=parent._tool_registry,
    )

    assert (repo / "app.py").read_text() == "value = 1\n"
    assert "Isolated change set task-000001: ready." in result
    assert 'subagent_changes(task_id="task-000001")' in result
    child_dir = run_root / "subagents" / "task-000001"
    manifest = json.loads((child_dir / CHANGESET_FILE).read_text())
    assert manifest["status"] == "ready"
    file_mode = (repo / "app.py").stat().st_mode & 0o777
    assert manifest["operations"] == [{
        "after_bytes": 10,
        "after_mode": file_mode,
        "after_sha256": hashlib.sha256(b"value = 2\n").hexdigest(),
        "before_bytes": 10,
        "before_mode": file_mode,
        "before_sha256": hashlib.sha256(b"value = 1\n").hexdigest(),
        "kind": "update",
        "path": "app.py",
    }]
    assert (child_dir / PATCH_FILE).is_file()
    assert not any((repo / ".yuj_worktrees").glob("subagent-*"))

    review = dispatch(
        "subagent_changes",
        {"task_id": "task-000001"},
        cwd=str(repo),
        cfg=cfg,
        tool_registry=parent._tool_registry,
    )
    assert "-value = 1" in review
    assert "+value = 2" in review
    applied = dispatch(
        "apply_subagent",
        {"task_id": "task-000001"},
        cwd=str(repo),
        cfg=cfg,
        tool_registry=parent._tool_registry,
    )
    assert applied.startswith("OK: applied isolated change set task-000001")
    assert (repo / "app.py").read_text() == "value = 2\n"
    assert (child_dir / APPLICATION_FILE).is_file()
    assert "already applied" in dispatch(
        "apply_subagent",
        {"task_id": "task-000001"},
        cwd=str(repo),
        cfg=cfg,
        tool_registry=parent._tool_registry,
    )


def test_isolated_change_application_rejects_stale_parent_state(tmp_path):
    repo = _repo(tmp_path)
    agents = _write_agent(
        tmp_path / "agents",
        tools=("write", "done"),
        read_only=False,
        workspace="isolated",
    )
    child = ScriptedClient([
        _calls(ToolCall("write-1", "write", {
            "path": "app.py", "content": "child value\n",
        })),
        _stop("Prepared the change.", 8, 3),
    ])
    cfg, parent, _runtime, _trace = _parent_session(
        repo,
        child_client=child,
        agents_dir=agents,
        run_root=tmp_path / "run",
    )
    assert "ready" in dispatch(
        "task",
        {"agent": "probe", "prompt": "Change app.py."},
        cwd=str(repo),
        cfg=cfg,
        tool_registry=parent._tool_registry,
    )

    (repo / "app.py").write_text("operator value\n")
    result = dispatch(
        "apply_subagent",
        {"task_id": "task-000001"},
        cwd=str(repo),
        cfg=cfg,
        tool_registry=parent._tool_registry,
    )
    assert "stale_subagent_changes" in result
    assert (repo / "app.py").read_text() == "operator value\n"
    assert not (
        tmp_path / "run" / "subagents" / "task-000001" / APPLICATION_FILE
    ).exists()


def test_isolated_application_obeys_parent_permission_policy(tmp_path):
    repo = _repo(tmp_path)
    agents = _write_agent(
        tmp_path / "agents",
        tools=("write", "done"),
        read_only=False,
        workspace="isolated",
    )
    child = ScriptedClient([
        _calls(ToolCall("write-1", "write", {
            "path": "app.py", "content": "child value\n",
        })),
        _stop("Prepared the change.", 8, 3),
    ])
    caller = ScriptedClient([
        _calls(ToolCall("task-1", "task", {
            "agent": "probe", "prompt": "Change app.py.",
        })),
        _calls(ToolCall("apply-1", "apply_subagent", {
            "task_id": "task-000001",
        })),
        _stop("The parent rejected the application.", 8, 2),
    ])
    run_root = tmp_path / "run"
    run_root.mkdir()
    trace_path = run_root / "parent.trace.jsonl"
    with trace_path.open("w+") as trace:
        cfg, parent, _runtime, _trace = _parent_session(
            repo,
            child_client=child,
            agents_dir=agents,
            client=caller,
            run_root=run_root,
            trace_file=trace,
            cfg=_task_cfg(
                max_turns=3,
                permissions_rules={"apply_subagent": {"*": "deny"}},
                error_nudge_threshold=99,
                error_abort_threshold=99,
                error_same_class_threshold=99,
            ),
        )
        assert parent.run().finish_reason == "stop"
    assert (repo / "app.py").read_text() == "value = 1\n"
    assert not (
        run_root / "subagents" / "task-000001" / APPLICATION_FILE
    ).exists()
    permissions = [
        event for event in map(json.loads, trace_path.read_text().splitlines())
        if event.get("event") == "permission"
        and event.get("tool") == "apply_subagent"
    ]
    assert permissions[-1]["decision"] == "deny"


def test_isolated_process_tools_fail_closed_without_a_sandbox(tmp_path):
    repo = _repo(tmp_path)
    agents = _write_agent(
        tmp_path / "agents",
        tools=("bash", "write", "done"),
        read_only=False,
        workspace="isolated",
    )
    child = ScriptedClient([_stop("must not run", 1, 1)])
    cfg, parent, _runtime, _trace = _parent_session(
        repo,
        child_client=child,
        agents_dir=agents,
        run_root=tmp_path / "run",
    )

    result = dispatch(
        "task",
        {"agent": "probe", "prompt": "Use bash."},
        cwd=str(repo),
        cfg=cfg,
        tool_registry=parent._tool_registry,
    )
    assert "need an active sandbox" in result
    assert child.responses
    assert (repo / "app.py").read_text() == "value = 1\n"


def test_isolated_setup_failure_never_starts_the_child_model(tmp_path):
    workspace = tmp_path / "not-a-repository"
    workspace.mkdir()
    (workspace / "app.py").write_text("value = 1\n")
    agents = _write_agent(
        tmp_path / "agents",
        tools=("write", "done"),
        read_only=False,
        workspace="isolated",
    )
    child = ScriptedClient([_stop("must not run", 1, 1)])
    cfg, parent, _runtime, _trace = _parent_session(
        workspace,
        child_client=child,
        agents_dir=agents,
        run_root=tmp_path / "run",
    )

    result = dispatch(
        "task",
        {"agent": "probe", "prompt": "Change app.py."},
        cwd=str(workspace),
        cfg=cfg,
        tool_registry=parent._tool_registry,
    )
    assert result.startswith("ERROR: subagent task-000001 failed")
    assert "not a git repository" in result.lower()
    assert child.responses
    assert (workspace / "app.py").read_text() == "value = 1\n"


def test_interrupted_isolated_child_cleans_up_without_parent_mutation(tmp_path):
    class InterruptingClient(ScriptedClient):
        def chat(self, messages, tools, turn=0):
            raise KeyboardInterrupt()

    repo = _repo(tmp_path)
    agents = _write_agent(
        tmp_path / "agents",
        tools=("write", "done"),
        read_only=False,
        workspace="isolated",
    )
    child = InterruptingClient([])
    run_root = tmp_path / "run"
    cfg, parent, _runtime, _trace = _parent_session(
        repo,
        child_client=child,
        agents_dir=agents,
        run_root=run_root,
    )

    with pytest.raises(KeyboardInterrupt):
        dispatch(
            "task",
            {"agent": "probe", "prompt": "Change app.py."},
            cwd=str(repo),
            cfg=cfg,
            tool_registry=parent._tool_registry,
        )
    assert (repo / "app.py").read_text() == "value = 1\n"
    assert not any((repo / ".yuj_worktrees").glob("subagent-*"))
    events = [
        json.loads(line)
        for line in (
            run_root / "subagents" / "task-000001" / ".trace.jsonl"
        ).read_text().splitlines()
    ]
    assert events[-1]["event"] == "subagent_result"
    assert events[-1]["finish_reason"] == "error"


def test_isolated_change_set_refuses_an_over_limit_payload(tmp_path):
    repo = _repo(tmp_path)
    agents = _write_agent(
        tmp_path / "agents",
        tools=("write", "done"),
        read_only=False,
        workspace="isolated",
    )
    oversized = "x" * (MAX_CHANGESET_INPUT_BYTES + 1)
    child = ScriptedClient([
        _calls(ToolCall("write-1", "write", {
            "path": "app.py", "content": oversized,
        })),
        _stop("Prepared a large change.", 8, 3),
    ])
    run_root = tmp_path / "run"
    cfg, parent, _runtime, _trace = _parent_session(
        repo,
        child_client=child,
        agents_dir=agents,
        run_root=run_root,
    )

    result = dispatch(
        "task",
        {"agent": "probe", "prompt": "Replace app.py."},
        cwd=str(repo),
        cfg=cfg,
        tool_registry=parent._tool_registry,
    )
    assert "Isolated change set task-000001: too_large." in result
    assert (repo / "app.py").read_text() == "value = 1\n"
    manifest = json.loads((
        run_root / "subagents" / "task-000001" / CHANGESET_FILE
    ).read_text())
    assert manifest["status"] == "too_large"
    assert manifest["patch_file"] is None
    assert "not ready" in dispatch(
        "apply_subagent",
        {"task_id": "task-000001"},
        cwd=str(repo),
        cfg=cfg,
        tool_registry=parent._tool_registry,
    )


def test_production_model_router_builds_a_fresh_profiled_child(tmp_path):
    cfg = _task_cfg()
    parent_client = ScriptedClient([])
    parent_client.cfg = cfg
    created = []

    def client_factory(child_cfg, profile):
        child_client = ScriptedClient([_stop("profiled result", 9, 3)])
        child_client.cfg = child_cfg
        created.append((child_client, child_cfg, profile))
        return child_client

    build_model_role_runtime(
        cfg=cfg,
        main_client=parent_client,
        profiles_dir=PROJECT_ROOT / "profiles",
        client_factory=client_factory,
    )
    run_root = tmp_path / "run"
    runtime = SubagentRuntime(run_root)
    session = Session(
        cfg,
        parent_client,
        "system",
        "parent task",
        str(tmp_path),
        subagent_runtime=runtime,
        trace_path=run_root / ".trace.jsonl",
        artifact_dir=run_root,
    )

    assert dispatch(
        "task",
        {"agent": "research", "prompt": "Find the exact owner."},
        cwd=str(tmp_path),
        cfg=cfg,
        tool_registry=session._tool_registry,
    ) == "profiled result"
    assert len(created) == 1
    child_client, child_cfg, profile = created[0]
    assert child_client is not parent_client
    assert child_cfg.profile_name == "_base"
    assert profile.name == "_base"


def test_depth_cap_and_read_only_runtime_block_all_mutation_paths(tmp_path):
    agents = _write_agent(tmp_path / "agents")
    child = ScriptedClient([
        _calls(
            ToolCall("edit-1", "edit", {
                "path": "edited.txt", "old_str": "a", "new_str": "b",
            }),
            ToolCall("write-1", "write", {
                "path": "written.txt", "content": "nope",
            }),
            ToolCall("bash-1", "bash", {"cmd": "touch shell-write.txt"}),
        ),
        _stop("mutation attempts were blocked", 5, 3),
    ])
    cfg, parent, runtime, _trace = _parent_session(
        tmp_path, child_client=child, agents_dir=agents
    )
    result = dispatch(
        "task",
        {"agent": "probe", "prompt": "Try the requested checks."},
        cwd=str(tmp_path),
        cfg=cfg,
        tool_registry=parent._tool_registry,
    )
    assert result == "mutation attempts were blocked"
    assert not (tmp_path / "edited.txt").exists()
    assert not (tmp_path / "written.txt").exists()
    assert not (tmp_path / "shell-write.txt").exists()

    events = [
        json.loads(line)
        for line in (
            tmp_path / "run" / "subagents" / "task-000001" / ".trace.jsonl"
        ).read_text().splitlines()
    ]
    tool_results = {
        event["tool_name"]: event["result_summary"]
        for event in events if event.get("event") == "tool_call"
    }
    assert "unknown tool 'edit'" in tool_results["edit"]
    assert "unknown tool 'write'" in tool_results["write"]
    assert "read-only subagent blocked bash" in tool_results["bash"]

    capped = Session(
        cfg,
        ScriptedClient([]),
        "system",
        "child",
        str(tmp_path),
        subagent_level=1,
        subagent_runtime=runtime,
    )
    assert "task" not in capped.active_tool_names
    assert "depth cap 1 reached" in dispatch(
        "task",
        {"agent": "probe", "prompt": "nest"},
        cwd=str(tmp_path),
        cfg=cfg,
        tool_registry=capped._tool_registry,
    )


def test_parent_replay_reads_child_trace_without_starting_child(tmp_path):
    agents = _write_agent(tmp_path / "agents")
    source_root = tmp_path / "source"
    source_root.mkdir()
    child = ScriptedClient([_stop("recorded child result", 13, 5)])
    cfg, source_parent, _runtime, source_trace_buffer = _parent_session(
        tmp_path,
        child_client=child,
        agents_dir=agents,
        run_root=source_root,
    )
    assert dispatch(
        "task",
        {"agent": "probe", "prompt": "record this"},
        cwd=str(tmp_path),
        cfg=cfg,
        tool_registry=source_parent._tool_registry,
    ) == "recorded child result"
    source_trace = source_root / ".trace.jsonl"
    source_trace.write_text(source_trace_buffer.getvalue())

    transcript = source_root / "parent.log"
    transcript.write_text(
        "=== turn 001 input ===\n{}\n"
        "=== turn 001 output ===\n"
        + json.dumps({
            "choices": [{
                "message": {
                    "content": "delegate",
                    "tool_calls": [{
                        "id": "parent-task-1",
                        "type": "function",
                        "function": {
                            "name": "task",
                            "arguments": json.dumps({
                                "agent": "probe", "prompt": "record this",
                            }),
                        },
                    }],
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {"prompt_tokens": 7, "completion_tokens": 2},
        })
        + "\n=== turn 002 input ===\n{}\n"
        "=== turn 002 output ===\n"
        + json.dumps({
            "choices": [{
                "message": {"content": "parent complete", "tool_calls": []},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 8, "completion_tokens": 2},
        })
        + "\n"
    )
    replay_client = ReplayClient(
        transcript,
        source_trace_path=source_trace,
        strict_fidelity=True,
    )
    child_started = False

    def forbidden_factory(parent, spec, task_id):
        nonlocal child_started
        child_started = True
        raise AssertionError("replay started a child model")

    replay_root = tmp_path / "replay"
    replay_root.mkdir()
    runtime = SubagentRuntime(
        replay_root,
        agents_dir=agents,
        client_factory=forbidden_factory,
    )
    output_trace = replay_root / ".trace.jsonl"
    with open(output_trace, "w") as trace_file:
        session = Session(
            cfg,
            replay_client,
            "system",
            "parent task",
            str(tmp_path),
            trace_file=trace_file,
            trace_path=output_trace,
            artifact_dir=replay_root,
            subagent_runtime=runtime,
        )
        role_ledger = RoleTokenLedger()
        session._role_token_ledger = role_ledger
        replay_result = session.run()
    assert replay_result.done is True
    assert child_started is False
    assert session._subagent_prompt_tokens == 13
    assert session._subagent_completion_tokens == 5
    assert (
        replay_root / "subagents" / "task-000001" / ".trace.jsonl"
    ).read_bytes() == (
        source_root / "subagents" / "task-000001" / ".trace.jsonl"
    ).read_bytes()
    tool_message = next(
        message for message in session.context.get_messages()
        if message.get("role") == "tool"
    )
    assert tool_message["content"] == "recorded child result"
    assert role_ledger.snapshot()["subagent.probe"] == {
        "requests": 1,
        "prompt_tokens": 13,
        "completion_tokens": 5,
        "cached_tokens": 0,
        "total_tokens": 18,
    }


def test_replay_copies_isolated_patch_and_applies_it_to_fresh_workspace(tmp_path):
    source_repo = _repo(tmp_path)
    agents = _write_agent(
        tmp_path / "agents",
        tools=("write", "done"),
        read_only=False,
        workspace="isolated",
    )
    source_root = tmp_path / "source-run"
    child = ScriptedClient([
        _calls(ToolCall("write-1", "write", {
            "path": "app.py", "content": "value = 2\n",
        })),
        _stop("Prepared app.py.", 9, 3),
    ])
    cfg, source_parent, _runtime, source_trace_buffer = _parent_session(
        source_repo,
        child_client=child,
        agents_dir=agents,
        run_root=source_root,
    )
    recorded_result = dispatch(
        "task",
        {"agent": "probe", "prompt": "Change app.py."},
        cwd=str(source_repo),
        cfg=cfg,
        tool_registry=source_parent._tool_registry,
    )
    source_trace = source_root / ".trace.jsonl"
    source_trace.write_text(source_trace_buffer.getvalue())
    transcript = source_root / "parent.log"
    transcript.write_text(
        "=== turn 001 input ===\n{}\n"
        "=== turn 001 output ===\n"
        '{"choices":[{"message":{"content":"done","tool_calls":[]},'
        '"finish_reason":"stop"}],"usage":{"prompt_tokens":1,'
        '"completion_tokens":1}}\n'
    )

    replay_repo = tmp_path / "replay-project"
    _git(tmp_path, "clone", "-q", str(source_repo), str(replay_repo))
    replay = ReplayClient(
        transcript,
        source_trace_path=source_trace,
        strict_fidelity=True,
    )
    replay_root = tmp_path / "replay-run"
    runtime = SubagentRuntime(replay_root, agents_dir=agents)
    replay_parent = Session(
        cfg,
        replay,
        "system",
        "parent task",
        str(replay_repo),
        subagent_runtime=runtime,
        trace_path=replay_root / ".trace.jsonl",
        artifact_dir=replay_root,
    )

    replayed_result = dispatch(
        "task",
        {"agent": "probe", "prompt": "Change app.py."},
        cwd=str(replay_repo),
        cfg=cfg,
        tool_registry=replay_parent._tool_registry,
    )
    assert replayed_result == recorded_result
    replay_child = replay_root / "subagents" / "task-000001"
    assert (replay_child / CHANGESET_FILE).is_file()
    assert (replay_child / PATCH_FILE).is_file()
    assert not (replay_child / APPLICATION_FILE).exists()
    applied = dispatch(
        "apply_subagent",
        {"task_id": "task-000001"},
        cwd=str(replay_repo),
        cfg=cfg,
        tool_registry=replay_parent._tool_registry,
    )
    assert applied.startswith("OK: applied isolated change set")
    assert (replay_repo / "app.py").read_text() == "value = 2\n"
    assert (source_repo / "app.py").read_text() == "value = 1\n"


def test_child_tokens_flow_into_metrics_json(tmp_path):
    parent = ScriptedClient([
        _calls(ToolCall("task-1", "task", {
            "agent": "research", "prompt": "Locate the owner.",
        }), prompt=5, completion=2),
        _stop("parent complete", 6, 3),
    ])
    child = ScriptedClient([_stop("agents/README.md owns the format", 11, 4)])

    def factory(parent_session, spec, task_id):
        return SubagentModelBinding(
            client=child,
            config=parent_session.cfg,
            dedicated=True,
        )

    parent._subagent_client_factory = factory
    cfg = _task_cfg()
    task = tmp_path / "task"
    task.mkdir()
    artifacts = tmp_path / "artifacts"
    assert solve_task(
        task,
        cfg,
        parent,
        task_spec=TaskSpec(prompt_text="Use the research agent, then finish."),
        artifacts_dir=artifacts,
    ) is True

    metrics = json.loads((artifacts / "metrics.json").read_text())["metrics"]
    assert metrics["total_prompt_tokens"] == 22
    assert metrics["total_completion_tokens"] == 9
    assert metrics["total_tokens"] == 31
    assert metrics["subagents"] == {
        "calls": 1,
        "prompt_tokens": 11,
        "completion_tokens": 4,
        "total_tokens": 15,
    }
    assert metrics["tokens_by_role"]["subagent.research"]["total_tokens"] == 15

    state = json.loads((artifacts / ".solver" / "state.json").read_text())
    assert len(state["trace"]) == 1
    assert state["trace"][0]["action"].startswith("task(")
    assert "agents/README.md owns the format" in state["trace"][0]["result"]

    parent_events = [
        json.loads(line)
        for line in (artifacts / ".trace.jsonl").read_text().splitlines()
    ]
    assert sum(event.get("event") == "subagent" for event in parent_events) == 1
