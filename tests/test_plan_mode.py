"""End-to-end acceptance coverage for the engine-enforced plan phase."""
from __future__ import annotations

import json
import subprocess
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from scripts.llm_assist.__main__ import (
    _render_provider_overlay,
    _transport_overrides_from_args,
    main as assist_main,
)
from scripts.llm_solver.__main__ import main as measurement_main
from scripts.llm_solver._shared.telemetry_paths import trace_path
from scripts.llm_solver.config import load_config
from scripts.llm_solver.harness._guardrails._git_dirty import (
    cwd_has_uncommitted_changes,
)
from scripts.llm_solver.harness._loop.profile_resolution import (
    apply_profile_to_schemas,
    build_plan_mode_schemas,
)
from scripts.llm_solver.harness._loop.trace_schema import (
    TRACE_EVENT_REQUIRED_FIELDS,
)
from scripts.llm_solver.harness.context import FullTranscript
from scripts.llm_solver.harness.guardrails import Action, GuardrailState, done_guard
from scripts.llm_solver.harness.loop import Session, solve_task
from scripts.llm_solver.harness.plan_mode import (
    PLAN_FILE,
    PlanModeController,
    bash_is_read_only,
    filter_plan_mode_schemas,
    is_exact_plan_path,
)
from scripts.llm_solver.harness.schemas import get_tool_schemas
from scripts.llm_solver.harness.state_writer import project
from scripts.llm_solver.server.types import ToolCall, TurnResult, Usage

from _config_helpers import make_config


def _turn(*, calls=(), content=None, reason="tool_calls") -> TurnResult:
    return TurnResult(
        content=content,
        tool_calls=list(calls),
        finish_reason=reason,
        usage=Usage(prompt_tokens=10, completion_tokens=2),
    )


def _client(*turns: TurnResult) -> MagicMock:
    client = MagicMock()
    client.chat.side_effect = list(turns)
    client.build_assistant_message.side_effect = [
        {"role": "assistant", "content": turn.content}
        for turn in turns
    ]
    return client


def _plan_cfg(**overrides):
    values = dict(
        plan_mode="required",
        plan_mode_max_turns=15,
        max_turns=8,
        max_sessions=1,
        duplicate_abort=20,
        error_nudge_threshold=99,
        error_abort_threshold=99,
        error_same_class_threshold=99,
        rumination_nudge_threshold=999,
        allow_implicit_done=False,
        tools_schema_validation="reject",
        tools_apply_patch_enabled=True,
        tools_list_definitions_enabled=True,
    )
    values.update(overrides)
    return make_config(**values)


def test_plan_config_defaults_overlay_validation_and_cli_overlay(tmp_path: Path):
    defaults = load_config()
    assert defaults.plan_mode == "off"
    assert defaults.plan_mode_max_turns == 15

    overlay = tmp_path / "plan.toml"
    overlay.write_text('[loop]\nplan_mode = "required"\nplan_mode_max_turns = 7\n')
    configured = load_config(overlay)
    assert configured.plan_mode == "required"
    assert configured.plan_mode_enabled is True
    assert configured.plan_mode_max_turns == 7

    for name, body in (
        ("bad-mode", '[loop]\nplan_mode = "sometimes"\n'),
        ("bad-turns", "[loop]\nplan_mode_max_turns = 0\n"),
    ):
        invalid = tmp_path / f"{name}.toml"
        invalid.write_text(body)
        with pytest.raises(ValueError, match="plan_mode"):
            load_config(invalid)

    args = SimpleNamespace(
        provider=None,
        base_url=None,
        api_key_env=None,
        thinking=None,
        plan_mode="required",
    )
    assert _transport_overrides_from_args(args) == {"plan_mode": "required"}
    assert _render_provider_overlay({"plan_mode": "required"}) == (
        '[loop]\nplan_mode = "required"\n'
    )


def test_measurement_cli_maps_plan_mode_override(tmp_path: Path):
    run_dir = tmp_path / "run"
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    cfg = _plan_cfg(runtime_mode="measurement")

    with (
        patch("scripts.llm_solver.__main__.load_config", return_value=cfg) as load,
        patch(
            "scripts.llm_solver.__main__.load_profile",
            return_value=SimpleNamespace(name="test", inherits="_base"),
        ),
        patch("scripts.llm_solver.__main__._build_run_metadata", return_value={}),
        patch("scripts.llm_solver.__main__._write_session_json"),
    ):
        assert measurement_main([
            str(run_dir),
            "--task", str(task_dir),
            "--dry-run",
            "--plan-mode", "required",
        ]) == 0

    assert load.call_args.kwargs["overrides"]["plan_mode"] == "required"


@pytest.mark.parametrize(
    ("command", "handler"),
    [("code", "cmd_run"), ("run", "cmd_run"), ("smoke", "cmd_smoke")],
)
def test_installed_cli_exposes_plan_mode(command: str, handler: str):
    with patch(
        f"scripts.llm_assist.__main__.{handler}", return_value=0,
    ) as invoked:
        assert assist_main([command, "--plan-mode", "required"]) == 0

    assert invoked.call_args.args[0].plan_mode == "required"


def test_exit_tool_is_profile_gated_and_cap_immune():
    schemas = get_tool_schemas("minimal")
    profile = SimpleNamespace(max_tools=1, simplify_schemas=False)
    client = SimpleNamespace(profile=profile)

    off_names = [
        schema["function"]["name"]
        for schema in apply_profile_to_schemas(schemas, make_config(), client)
    ]
    assert off_names == ["done"]

    required_names = [
        schema["function"]["name"]
        for schema in build_plan_mode_schemas(_plan_cfg(), client)
    ]
    assert required_names == ["write", "exit_plan_mode"]
    assert filter_plan_mode_schemas(
        build_plan_mode_schemas(_plan_cfg(), client), active=True,
    ) == build_plan_mode_schemas(_plan_cfg(), client)


@pytest.mark.parametrize(
    "command",
    [
        "git status --short",
        "git -C . diff --stat && rg plan scripts | head -20",
        "grep 'value[0-9]' config.toml",
        "jq '.items[]' data.json",
        "sed -n '1,40p' config.toml",
        "find scripts -maxdepth 2 -type f",
        "ls -la | wc -l",
    ],
)
def test_plan_bash_classifier_accepts_only_known_read_surfaces(command):
    assert bash_is_read_only({"cmd": command}) is True


@pytest.mark.parametrize(
    "arguments",
    [
        {"cmd": "touch changed.txt"},
        {"cmd": "sed -i 's/a/b/' file.py"},
        {"cmd": "printf x > changed.txt"},
        {"cmd": "find . -delete"},
        {"cmd": "sed -n 'w changed.txt' file.py"},
        {"cmd": "sort input.txt -o changed.txt"},
        {"cmd": "git diff --output=changed.patch"},
        {"cmd": "git branch -d old-branch"},
        {"cmd": "git -c diff.external='touch changed.txt' diff"},
        {"cmd": "GIT_EXTERNAL_DIFF='touch changed.txt' git diff"},
        {"cmd": "find . $ACTION"},
        {"cmd": "find . {-print,-delete}"},
        {"cmd": "git diff *"},
        {"cmd": "git diff --ext-diff"},
        {"cmd": "/tmp/cat README.md"},
        {"cmd": "sed -n -e '1p' -e 'w changed.txt' file.py"},
        {"cmd": "sed -n -i '1p' file.py"},
        {"cmd": "printf -v SHELL_STATE x"},
        {"cmd": "hostname changed-host"},
        {"cmd": "python -c \"open('changed.txt', 'w').write('x')\""},
        {"cmd": "git checkout -- file.py"},
        {"cmd": "ls", "background": True},
    ],
)
def test_plan_bash_classifier_fails_closed_on_mutating_or_async_calls(arguments):
    assert bash_is_read_only(arguments) is False


def test_plan_path_is_exact_and_rejects_symlink_escape(tmp_path: Path):
    assert is_exact_plan_path(tmp_path, PLAN_FILE)
    assert is_exact_plan_path(tmp_path, f"./{PLAN_FILE}")
    assert not is_exact_plan_path(tmp_path, ".solver/other.md")

    outside = tmp_path / "outside"
    outside.mkdir()
    linked_root = tmp_path / "linked"
    linked_root.mkdir()
    (linked_root / ".solver").symlink_to(outside, target_is_directory=True)
    assert not is_exact_plan_path(linked_root, PLAN_FILE)


def test_plan_turn_cap_keeps_only_plan_write_and_exit_available(tmp_path: Path):
    events = [
        {"event": "plan_mode_enter", "session_number": 1, "turn": 0},
        {"event": "turn", "session_number": 1, "turn_number": 0},
    ]
    controller = PlanModeController(
        cwd=str(tmp_path),
        cfg=_plan_cfg(plan_mode_max_turns=1),
        events=events,
        event_sink=lambda _event: None,
    )

    assert not controller.check("read", {"path": "README.md"}, turn=0).allowed
    assert controller.check(
        "write", {"path": PLAN_FILE, "content": "plan"}, turn=0
    ).allowed
    assert controller.check("exit_plan_mode", {}, turn=0).allowed


def test_mutating_tools_are_unified_rejections_before_dispatch(tmp_path: Path):
    calls = [
        ToolCall("edit", "edit", {
            "path": PLAN_FILE, "old_str": "old", "new_str": "new",
        }),
        ToolCall("patch", "apply_patch", {
            "patch": "*** Begin Patch\n*** Add File: x.py\n+x = 1\n*** End Patch",
        }),
        ToolCall("write", "write", {"path": "outside.txt", "content": "x"}),
        ToolCall("bash", "bash", {"cmd": "touch shell-write.txt"}),
    ]
    client = _client(
        _turn(calls=calls),
        _turn(content="stopping", reason="stop"),
    )
    trace = StringIO()
    captured: dict[str, str] = {}

    with patch("scripts.llm_solver.harness.loop.dispatch") as dispatch_spy, (
        patch.object(Session, "_get_server_ctx", return_value=8192)
    ):
        session = Session(
            _plan_cfg(max_turns=2),
            client,
            "system",
            "task",
            str(tmp_path),
            context_manager=FullTranscript(),
            trace_file=trace,
            session_number=1,
        )
        original = session.context.add_tool_result

        def capture(call_id, result, **kwargs):
            captured[call_id] = result
            return original(call_id, result, **kwargs)

        session.context.add_tool_result = capture
        result = session.run()

    assert result.done is False
    assert result.finish_reason == "no_tool_call"
    dispatch_spy.assert_not_called()
    assert set(captured) == {"edit", "patch", "write", "bash"}
    assert all('status="error"' in value for value in captured.values())
    assert all('error_kind="plan_mode"' in value for value in captured.values())
    assert not (tmp_path / "outside.txt").exists()
    assert not (tmp_path / "shell-write.txt").exists()
    assert not (tmp_path / "x.py").exists()

    events = [json.loads(line) for line in trace.getvalue().splitlines()]
    rejected = [event for event in events if event.get("gate_reason") == "plan_mode"]
    assert len(rejected) == 4
    assert all(event["gate_blocked"] is True for event in rejected)


def test_malformed_plan_write_stays_non_mutating_in_trace(tmp_path: Path):
    client = _client(
        _turn(calls=[ToolCall("bad-plan", "write", {"path": PLAN_FILE})]),
        _turn(content="stopping", reason="stop"),
    )
    trace = StringIO()
    session = Session(
        _plan_cfg(max_turns=2),
        client,
        "system",
        "task",
        str(tmp_path),
        context_manager=FullTranscript(),
        trace_file=trace,
        session_number=1,
    )

    with patch.object(Session, "_get_server_ctx", return_value=8192):
        result = session.run()

    assert result.done is False
    event = next(
        item for item in map(json.loads, trace.getvalue().splitlines())
        if item.get("tool_call_id") == "bad-plan"
    )
    assert event["gate_reason"] == "schema_reject"
    assert event["plan_artifact"] is True
    assert event["write_like"] is False
    assert event["source_write_like"] is False


@pytest.mark.parametrize("plan_contents", [None, " \n"])
def test_exit_without_nonempty_plan_is_a_unified_error(
    tmp_path: Path, plan_contents: str | None,
):
    if plan_contents is not None:
        plan_path = tmp_path / PLAN_FILE
        plan_path.parent.mkdir(parents=True)
        plan_path.write_text(plan_contents)
    client = _client(
        _turn(calls=[ToolCall("exit", "exit_plan_mode", {})]),
        _turn(content="stopping", reason="stop"),
    )
    captured: list[str] = []
    session = Session(
        _plan_cfg(max_turns=2),
        client,
        "system",
        "task",
        str(tmp_path),
        context_manager=FullTranscript(),
    )
    original = session.context.add_tool_result
    session.context.add_tool_result = lambda call_id, value, **kwargs: (
        captured.append(value), original(call_id, value, **kwargs)
    )[1]

    with patch.object(Session, "_get_server_ctx", return_value=8192):
        result = session.run()

    assert result.done is False
    assert session._plan_mode.active is True
    assert len(captured) == 1
    assert 'status="error"' in captured[0]
    assert 'error_kind="plan_mode"' in captured[0]
    assert PLAN_FILE in captured[0]


def test_plan_write_does_not_flip_in_memory_mutation_state(tmp_path: Path):
    client = _client(
        _turn(calls=[ToolCall(
            "plan", "write", {"path": PLAN_FILE, "content": "Implement safely.\n"},
        )]),
        _turn(content="stopping", reason="stop"),
    )
    session = Session(
        _plan_cfg(max_turns=2),
        client,
        "system",
        "task",
        str(tmp_path),
        context_manager=FullTranscript(),
    )

    with patch.object(Session, "_get_server_ctx", return_value=8192):
        result = session.run()

    assert result.done is False
    assert (tmp_path / PLAN_FILE).read_text() == "Implement safely.\n"
    assert session._guards.has_mutated is False


def test_exit_does_not_unlock_a_mutating_sibling_call(tmp_path: Path):
    client = _client(
        _turn(calls=[
            ToolCall(
                "plan", "write",
                {"path": PLAN_FILE, "content": "Implement safely.\n"},
            ),
            ToolCall("exit", "exit_plan_mode", {}),
            ToolCall(
                "source", "write",
                {"path": "too-early.py", "content": "unsafe = True\n"},
            ),
        ]),
        _turn(content="stopping", reason="stop"),
    )
    captured: dict[str, str] = {}
    session = Session(
        _plan_cfg(max_turns=2),
        client,
        "system",
        "task",
        str(tmp_path),
        context_manager=FullTranscript(),
    )
    original = session.context.add_tool_result

    def capture(call_id, result, **kwargs):
        captured[call_id] = result
        return original(call_id, result, **kwargs)

    session.context.add_tool_result = capture
    with patch.object(Session, "_get_server_ctx", return_value=8192):
        result = session.run()

    assert result.done is False
    assert session._plan_mode.active is False
    assert not (tmp_path / "too-early.py").exists()
    assert 'error_kind="plan_mode"' in captured["source"]


def test_runtime_writes_plan_exits_phase_then_unlocks_mutation(tmp_path: Path):
    plan = "1. Inspect the parser.\n2. Update it and run the focused test.\n"
    client = _client(
        _turn(calls=[ToolCall(
            "plan", "write", {"path": PLAN_FILE, "content": plan},
        )]),
        _turn(calls=[ToolCall("exit", "exit_plan_mode", {})]),
        _turn(calls=[ToolCall(
            "source",
            "apply_patch",
            {
                "patch": (
                    "*** Begin Patch\n"
                    "*** Add File: answer.py\n"
                    "+answer = 42\n"
                    "*** End Patch"
                )
            },
        )]),
        _turn(calls=[ToolCall("done", "done", {"message": "complete"})]),
    )

    with patch("scripts.llm_solver.harness.loop._auto_commit"), patch.object(
        Session, "_get_server_ctx", return_value=8192,
    ):
        success = solve_task(
            tmp_path,
            _plan_cfg(
                max_turns=4,
                done_guard_enabled=False,
                require_intent=True,
                intent_grace_turns=0,
                guardrails_arm_after_turn=0,
            ),
            client,
            context_class=FullTranscript,
            initial_prompt="Implement the requested change.",
        )

    assert success is True
    assert (tmp_path / PLAN_FILE).read_text() == plan
    assert (tmp_path / "answer.py").read_text() == "answer = 42\n"

    first_surface = {
        schema["function"]["name"] for schema in client.chat.call_args_list[0].args[1]
    }
    implementation_surface = {
        schema["function"]["name"] for schema in client.chat.call_args_list[2].args[1]
    }
    assert first_surface == {
        "bash",
        "read",
        "write",
        "glob",
        "grep",
        "list_definitions",
        "exit_plan_mode",
    }
    assert {"apply_patch", "done"} <= implementation_surface
    assert "edit" not in implementation_surface

    events = [
        json.loads(line)
        for line in trace_path(tmp_path).read_text().splitlines()
        if line.strip()
    ]
    enters = [event for event in events if event["event"] == "plan_mode_enter"]
    exits = [event for event in events if event["event"] == "plan_mode_exit"]
    assert len(enters) == 1
    assert len(exits) == 1
    assert exits[0]["turn"] == 1
    assert exits[0]["plan_chars"] == len(plan)
    assert TRACE_EVENT_REQUIRED_FIELDS["plan_mode_exit"] == frozenset({
        "session_number", "turn", "plan_chars",
    })
    assert not [
        event for event in events
        if event["event"] == "stale_guard_observe"
        and event.get("path") == PLAN_FILE
    ]

    state = json.loads((tmp_path / ".solver" / "state.json").read_text())
    assert state["state"]["phase"] == "implementation"
    plan_write = next(
        item for item in state["trace"]
        if item["action"].startswith("write(") and item["plan_artifact"]
    )
    assert plan_write["write_like"] is False
    assert plan_write["source_write_like"] is False


def test_plan_phase_temporarily_supersedes_code_mode_surface(tmp_path: Path):
    plan_path = tmp_path / PLAN_FILE
    plan_path.parent.mkdir(parents=True)
    plan_path.write_text("Inspect first, then implement through code mode.\n")
    client = _client(
        _turn(calls=[ToolCall("exit", "exit_plan_mode", {})]),
        _turn(content="stopping", reason="stop"),
    )
    session = Session(
        _plan_cfg(max_turns=2, tools_exec_cell_enabled=True),
        client,
        "system",
        "task",
        str(tmp_path),
        context_manager=FullTranscript(),
    )

    with patch.object(Session, "_get_server_ctx", return_value=8192):
        result = session.run()

    assert result.done is False
    plan_surface = {
        schema["function"]["name"]
        for schema in client.chat.call_args_list[0].args[1]
    }
    code_surface = {
        schema["function"]["name"]
        for schema in client.chat.call_args_list[1].args[1]
    }
    assert plan_surface == {
        "bash",
        "read",
        "write",
        "glob",
        "grep",
        "list_definitions",
        "exit_plan_mode",
    }
    assert code_surface == {
        "list_functions",
        "get_function_details",
        "exec_cell",
        "done",
    }


def test_runtime_state_remains_in_plan_phase_without_exit(tmp_path: Path):
    client = _client(_turn(content="stopping", reason="stop"))

    with patch("scripts.llm_solver.harness.loop._auto_commit"), patch.object(
        Session, "_get_server_ctx", return_value=8192,
    ):
        success = solve_task(
            tmp_path,
            _plan_cfg(max_turns=1),
            client,
            context_class=FullTranscript,
            initial_prompt="Inspect and plan the requested change.",
        )

    assert success is False
    events = [
        json.loads(line)
        for line in trace_path(tmp_path).read_text().splitlines()
        if line.strip()
    ]
    assert len([event for event in events if event["event"] == "plan_mode_enter"]) == 1
    assert not [event for event in events if event["event"] == "plan_mode_exit"]
    state = json.loads((tmp_path / ".solver" / "state.json").read_text())
    assert state["state"]["phase"] == "plan"


def _init_git_repo(path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    (path / "seed.txt").write_text("seed\n")
    subprocess.run(["git", "add", "seed.txt"], cwd=path, check=True)
    subprocess.run(
        [
            "git", "-c", "user.name=Plan Test",
            "-c", "user.email=plan@example.invalid",
            "commit", "-q", "-m", "seed",
        ],
        cwd=path,
        check=True,
    )


def test_done_guard_does_not_count_plan_file_as_mutation(tmp_path: Path):
    _init_git_repo(tmp_path)
    plan_path = tmp_path / PLAN_FILE
    plan_path.parent.mkdir(parents=True)
    plan_path.write_text("Inspect, edit, verify.\n")

    assert cwd_has_uncommitted_changes(str(tmp_path)) is False
    state = GuardrailState()
    state.verified_since_mutation = True
    cfg = make_config(
        done_guard_enabled=True,
        done_require_mutation=True,
        done_require_verify=False,
    )
    decision = done_guard(state, cfg, tc_name="done", cwd=str(tmp_path))
    assert decision.action == Action.BLOCK
    assert state.has_mutated is False

    subprocess.run(["git", "add", PLAN_FILE], cwd=tmp_path, check=True)
    subprocess.run(
        [
            "git", "-c", "user.name=Plan Test",
            "-c", "user.email=plan@example.invalid",
            "commit", "-q", "-m", "track plan",
        ],
        cwd=tmp_path,
        check=True,
    )
    plan_path.write_text("Revise, implement, verify.\n")
    assert cwd_has_uncommitted_changes(str(tmp_path)) is False

    (tmp_path / "implementation.py").write_text("implemented = True\n")
    assert cwd_has_uncommitted_changes(str(tmp_path)) is True
    changed_state = GuardrailState()
    changed_state.verified_since_mutation = True
    assert done_guard(
        changed_state, cfg, tc_name="done", cwd=str(tmp_path)
    ).action == Action.PASS


def test_state_projection_keeps_plan_artifact_non_mutating():
    events = [
        {"event": "plan_mode_enter", "session_number": 1, "turn": 0},
        {
            "event": "tool_call",
            "session_number": 1,
            "turn_number": 0,
            "tool_name": "write",
            "args_summary": "path='.solver/plan.md'",
            "result_summary": "OK",
            "plan_artifact": True,
            "write_like": False,
            "source_write_like": False,
        },
        {
            "event": "plan_mode_exit",
            "session_number": 1,
            "turn": 1,
            "plan_chars": 20,
        },
    ]

    plan_state = project(
        events[:2],
        max_result_chars=20000,
        imperative_projection=True,
    )
    assert plan_state["state"]["phase"] == "plan"
    assert plan_state["process"]["last_mutation_step"] is None

    state = project(
        events,
        max_result_chars=20000,
        imperative_projection=True,
    )
    assert state["state"]["phase"] == "implementation"
    assert state["trace"][0]["plan_artifact"] is True
    assert state["process"]["last_mutation_step"] is None
    assert state["process"]["phase"] == "pre_mutation_discovery"
