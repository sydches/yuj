"""Acceptance coverage for Agent Skills discovery and runtime wiring."""
from __future__ import annotations

import json
import shlex
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from scripts.llm_solver._shared.telemetry_paths import trace_path
from scripts.llm_solver.config import load_config
from scripts.llm_solver.harness._tools.read import read
from scripts.llm_solver.harness._tools.write import write
from scripts.llm_solver.harness._loop.profile_resolution import (
    apply_profile_to_schemas,
)
from scripts.llm_solver.harness.loop import solve_task
from scripts.llm_solver.harness.schemas import get_tool_schemas
from scripts.llm_solver.harness.sandbox import (
    _build_bwrap_argv,
    bwrap_preflight,
)
from scripts.llm_solver.harness.sandbox.ignore_policy import load_ignore_policy
from scripts.llm_solver.harness.sandbox.container_backend import (
    _build_container_argv,
)
from scripts.llm_solver.harness.skills import (
    SkillError,
    discover_skills,
    load_skill,
)
from scripts.llm_solver.harness.tools import bash, dispatch
from scripts.llm_solver.server.types import ToolCall, TurnResult, Usage

from _config_helpers import make_config


FIXTURES = Path(__file__).parent / "fixtures" / "skills"
IMAGE = "example.invalid/yuj-task@sha256:" + ("a" * 64)


def _skill(directory: Path, name: str, description: str, body: str, **fields) -> Path:
    root = directory / name
    root.mkdir(parents=True)
    lines = ["---", f"name: {name}", f"description: {description}"]
    for key, value in fields.items():
        lines.append(f"{key.replace('_', '-')}: {str(value).lower()}")
    lines.extend(("---", "", body, ""))
    path = root / "SKILL.md"
    path.write_text("\n".join(lines))
    return path


def test_canonical_skill_knobs_load_with_reference_discovery_defaults() -> None:
    cfg = load_config()

    assert cfg.skills_enabled is False
    assert cfg.skills_dirs == (
        "~/.pi/agent/skills",
        "~/.agents/skills",
        ".pi/skills",
        ".agents/skills",
    )
    assert cfg.skill_paths == ()
    assert cfg.skills_readable_dirs == ()


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ('skills_enabled = "yes"', "skills_enabled must be a boolean"),
        ('skills_dirs = ".agents/skills"', "skills_dirs must be an array"),
        ('skills_dirs = [""]', "skills_dirs entries must be non-empty"),
        ('skill_paths = [1]', "skill_paths must be an array"),
        ('skill_paths = [""]', "skill_paths entries must be non-empty"),
    ],
)
def test_skill_config_rejects_invalid_values(
    tmp_path: Path,
    body: str,
    message: str,
) -> None:
    overlay = tmp_path / "invalid.toml"
    overlay.write_text(f"[prompts]\n{body}\n")

    with pytest.raises(ValueError, match=message):
        load_config(user_config=overlay)


def test_skills_prioritize_read_under_profile_tool_cap() -> None:
    profile = SimpleNamespace(max_tools=2, simplify_schemas=False)
    client = SimpleNamespace(profile=profile)
    schemas = get_tool_schemas("minimal")

    enabled = apply_profile_to_schemas(
        schemas,
        make_config(
            skills_enabled=True,
            skills_readable_dirs=("/opt/yuj-test-skill",),
        ),
        client,
    )
    disabled = apply_profile_to_schemas(
        schemas,
        make_config(skills_enabled=False),
        client,
    )
    empty = apply_profile_to_schemas(
        schemas,
        make_config(skills_enabled=True, skills_readable_dirs=()),
        client,
    )

    assert [schema["function"]["name"] for schema in enabled] == [
        "done",
        "read",
    ]
    assert [schema["function"]["name"] for schema in disabled] == [
        "done",
        "bash",
    ]
    assert [schema["function"]["name"] for schema in empty] == [
        "done",
        "bash",
    ]


def test_skills_reject_profile_cap_that_cannot_retain_read() -> None:
    profile = SimpleNamespace(max_tools=1, simplify_schemas=False)
    client = SimpleNamespace(profile=profile)

    with pytest.raises(ValueError, match="skills_enabled requires the read tool"):
        apply_profile_to_schemas(
            get_tool_schemas("minimal"),
            make_config(
                skills_enabled=True,
                skills_readable_dirs=("/opt/yuj-test-skill",),
            ),
            client,
        )


def test_frontmatter_fixture_validates_agent_skills_fields() -> None:
    skill = load_skill(FIXTURES / "valid" / "code-review" / "SKILL.md")

    assert skill.name == "code-review"
    assert skill.description.startswith("Review a change")
    assert skill.license == "MIT"
    assert skill.compatibility == "Requires git."
    assert skill.metadata == {"author": "yuj-tests", "version": "1"}
    assert skill.allowed_tools == "Read Grep"
    assert skill.disable_model_invocation is False


def test_startup_reads_frontmatter_without_decoding_skill_body(
    tmp_path: Path,
) -> None:
    root = tmp_path / "binary-body"
    root.mkdir()
    path = root / "SKILL.md"
    path.write_bytes(
        b"---\nname: binary-body\ndescription: Metadata only.\n---\n\xff"
    )

    skill = load_skill(path)

    assert skill.name == "binary-body"


@pytest.mark.parametrize(
    "fixture",
    ["bad-name", "missing-description"],
)
def test_invalid_frontmatter_fixtures_stop_startup(fixture: str) -> None:
    path = FIXTURES / "invalid" / fixture / "SKILL.md"

    with pytest.raises(SkillError):
        discover_skills(
            FIXTURES,
            enabled=True,
            skills_dirs=(),
            skill_paths=(str(path),),
        )


def test_invalid_frontmatter_stops_solve_before_model_call(
    tmp_path: Path,
) -> None:
    work = tmp_path / "task"
    work.mkdir()
    (work / ".git").mkdir()
    (work / "prompt.txt").write_text("finish")
    client = MagicMock()
    cfg = make_config(
        skills_enabled=True,
        skills_dirs=(),
        skill_paths=(
            str(FIXTURES / "invalid" / "bad-name" / "SKILL.md"),
        ),
    )

    with pytest.raises(SkillError):
        solve_task(work, cfg, client)

    client.chat.assert_not_called()


def test_name_collision_keeps_first_configured_skill_fixture(caplog) -> None:
    first = FIXTURES / "collisions" / "first"
    second = FIXTURES / "collisions" / "second"

    catalog = discover_skills(
        FIXTURES,
        enabled=True,
        skills_dirs=(str(first), str(second)),
        skill_paths=(),
    )

    assert [skill.name for skill in catalog.skills] == ["calendar"]
    assert catalog.skills[0].description == "First calendar skill."
    assert catalog.skills[0].path == (
        first / "calendar" / "SKILL.md"
    ).resolve()
    assert "keeping first" in caplog.text


def test_exact_skill_path_wins_before_directory_discovery(
    tmp_path: Path,
) -> None:
    exact = _skill(
        tmp_path / "exact",
        "calendar",
        "Exact calendar skill.",
        "EXACT",
    )
    discovered = _skill(
        tmp_path / "collection",
        "calendar",
        "Discovered calendar skill.",
        "DISCOVERED",
    )

    catalog = discover_skills(
        tmp_path,
        enabled=True,
        skills_dirs=(str(discovered.parent.parent),),
        skill_paths=(str(exact),),
    )

    assert catalog.skills[0].path == exact.resolve()


def test_relative_collection_searches_from_cwd_to_project_root(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    nested = root / "packages" / "app"
    nested.mkdir(parents=True)
    (root / ".git").mkdir()
    _skill(
        root / ".agents" / "skills",
        "calendar",
        "Root calendar skill.",
        "ROOT",
    )
    local = _skill(
        nested / ".agents" / "skills",
        "calendar",
        "Local calendar skill.",
        "LOCAL",
    )

    catalog = discover_skills(
        nested,
        enabled=True,
        skills_dirs=(".agents/skills",),
        skill_paths=(),
    )

    assert [skill.path for skill in catalog.skills] == [local.resolve()]


def test_collection_scan_ignores_root_skill_file(tmp_path: Path) -> None:
    collection = tmp_path / "collection"
    collection.mkdir()
    (collection / "SKILL.md").write_text(
        "---\nname: collection\ndescription: Explicit only.\n---\n"
    )
    child = _skill(
        collection,
        "child-skill",
        "Discovered child skill.",
        "CHILD",
    )

    catalog = discover_skills(
        tmp_path,
        enabled=True,
        skills_dirs=(str(collection),),
        skill_paths=(),
    )

    assert [skill.path for skill in catalog.skills] == [child.resolve()]


def test_discovered_masked_skill_is_skipped_but_masked_exact_path_fails() -> None:
    skill = FIXTURES / "valid" / "code-review" / "SKILL.md"

    catalog = discover_skills(
        FIXTURES,
        enabled=True,
        skills_dirs=(str(FIXTURES / "valid"),),
        skill_paths=(),
        unreadable_paths=(str(skill),),
    )
    assert catalog.skills == ()

    with pytest.raises(SkillError, match="hidden by sandbox policy"):
        discover_skills(
            FIXTURES,
            enabled=True,
            skills_dirs=(),
            skill_paths=(str(skill),),
            unreadable_paths=(str(skill),),
        )


def test_hidden_fixture_is_loaded_but_omitted_from_prompt_catalog() -> None:
    catalog = discover_skills(
        FIXTURES,
        enabled=True,
        skills_dirs=(
            str(FIXTURES / "valid"),
            str(FIXTURES / "hidden"),
        ),
        skill_paths=(),
    )

    block = catalog.format_prompt_block()
    assert block.startswith("<skills>\n")
    assert "code-review: Review a change" in block
    assert str((FIXTURES / "valid" / "code-review" / "SKILL.md").resolve()) in block
    assert "Read the changed source" not in block
    assert "manual-only" not in block
    assert [row["name"] for row in catalog.trace_records()] == [
        "code-review",
        "manual-only",
    ]
    assert catalog.trace_records()[1]["disable_model_invocation"] is True


def test_external_skill_is_readable_but_file_tools_cannot_write_it(
    tmp_path: Path,
) -> None:
    task = tmp_path / "task"
    task.mkdir()
    skill_path = _skill(
        tmp_path / "external-skills",
        "outside-skill",
        "Read-only external instructions.",
        "EXTERNAL SKILL BODY",
    )
    cfg = make_config(skills_readable_dirs=(str(skill_path.parent),))

    result = read(str(skill_path), cwd=str(task), cfg=cfg)
    assert "EXTERNAL SKILL BODY" in result

    attempted = write(
        str(skill_path),
        "MUTATED",
        cwd=str(task),
        cfg=cfg,
    )
    assert attempted == f"ERROR: skill path is read-only: {skill_path}"
    assert "EXTERNAL SKILL BODY" in skill_path.read_text()
    assert not (task / str(skill_path).lstrip("/")).exists()


def test_external_skill_read_respects_configured_unreadable_mask(
    tmp_path: Path,
) -> None:
    task = tmp_path / "task"
    task.mkdir()
    skill_path = _skill(
        tmp_path / "external-skills",
        "masked-resource",
        "External instructions with a masked resource.",
        "PUBLIC BODY",
    )
    private = skill_path.parent / "private.txt"
    private.write_text("PRIVATE RESOURCE")
    cfg = make_config(
        skills_readable_dirs=(str(skill_path.parent),),
        unreadable_paths=(str(private),),
    )

    assert read(str(private), cwd=str(task), cfg=cfg) == (
        f"ERROR: file not found: {private}"
    )
    assert "No such file or directory" in bash(
        f"cat {shlex.quote(str(private))}",
        cwd=str(task),
        timeout=10,
        sandbox=True,
        readable_paths=(str(skill_path.parent),),
        unreadable_paths=(str(private),),
    )


def test_external_skill_resources_remain_listable_with_project_ignore_policy(
    tmp_path: Path,
) -> None:
    task = tmp_path / "task"
    task.mkdir()
    (task / ".yujignore").write_text("ignored.txt\n")
    skill_path = _skill(
        tmp_path / "external-skills",
        "listed-resources",
        "External instructions with resources.",
        "PUBLIC BODY",
    )
    (skill_path.parent / "reference.md").write_text("REFERENCE")
    cfg = make_config(
        sandbox_bash=True,
        skills_readable_dirs=(str(skill_path.parent),),
    )

    result = dispatch(
        "bash",
        {"cmd": f"ls {shlex.quote(str(skill_path.parent))}"},
        cwd=str(task),
        cfg=cfg,
        ignore_policy=load_ignore_policy(task),
    )

    assert "SKILL.md\nreference.md" in result
    assert ".yujignore" not in result


def test_external_skill_mount_is_read_only_in_bwrap(tmp_path: Path) -> None:
    preflight_ok, reason = bwrap_preflight("/usr/bin/bwrap")
    if not preflight_ok:
        pytest.skip(reason or "bwrap preflight failed")
    task = tmp_path / "task"
    task.mkdir()
    skill_path = _skill(
        tmp_path / "external-skills",
        "sandbox-skill",
        "Sandbox fixture.",
        "SANDBOX SKILL BODY",
    )
    skill_dir = str(skill_path.parent)

    readable = bash(
        f"sed -n '1,20p' {shlex.quote(str(skill_path))}",
        cwd=str(task),
        timeout=10,
        sandbox=True,
        sandbox_required=True,
        readable_paths=(skill_dir,),
    )
    assert "SANDBOX SKILL BODY" in readable

    blocked = bash(
        f"printf MUTATED > {shlex.quote(str(skill_path))}",
        cwd=str(task),
        timeout=10,
        sandbox=True,
        sandbox_required=True,
        readable_paths=(skill_dir,),
    )
    assert "read-only" in blocked.lower()
    assert "SANDBOX SKILL BODY" in skill_path.read_text()


def test_bwrap_argv_declares_external_skill_directory_read_only(
    tmp_path: Path,
) -> None:
    task = tmp_path / "task"
    task.mkdir()
    skill_dir = tmp_path / "external" / "bwrap-skill"
    skill_dir.mkdir(parents=True)

    argv = _build_bwrap_argv(
        "true",
        str(task),
        readable_paths=(str(skill_dir),),
    )

    triples = tuple(zip(argv, argv[1:], argv[2:]))
    assert ("--ro-bind", str(skill_dir), str(skill_dir)) in triples
    assert ("--bind", str(skill_dir), str(skill_dir)) not in triples


def test_container_argv_mounts_external_skill_directory_read_only(
    tmp_path: Path,
) -> None:
    task = tmp_path / "task"
    task.mkdir()
    skill_dir = tmp_path / "external" / "container-skill"
    skill_dir.mkdir(parents=True)

    argv = _build_container_argv(
        "true",
        task,
        image=IMAGE,
        runtime_bin="docker",
        uid=1,
        gid=1,
        readable_paths=(str(skill_dir),),
    )

    assert (
        f"type=bind,source={skill_dir},target={skill_dir},readonly,"
        "bind-propagation=rprivate"
    ) in argv


def test_solve_task_catalog_and_session_start_trace_loaded_skills(
    tmp_path: Path,
) -> None:
    work = tmp_path / "task"
    work.mkdir()
    (work / ".git").mkdir()
    (work / "prompt.txt").write_text("finish")
    skill_root = tmp_path / "external-skills"
    visible = _skill(
        skill_root,
        "visible-skill",
        "Visible startup metadata.",
        "VISIBLE BODY MUST BE ON DEMAND",
    )
    hidden = _skill(
        skill_root,
        "hidden-skill",
        "Hidden startup metadata.",
        "HIDDEN BODY MUST BE ON DEMAND",
        disable_model_invocation=True,
    )
    savings_dir = tmp_path / "savings"
    cfg = make_config(
        max_sessions=1,
        skills_enabled=True,
        skills_dirs=(),
        skill_paths=(str(visible), str(hidden)),
    )
    client = MagicMock()
    client.cfg = cfg
    client.chat.side_effect = [
        TurnResult(
            content="Load the matched skill.",
            tool_calls=[
                ToolCall(
                    id="read-skill",
                    name="read",
                    arguments={"path": str(visible)},
                )
            ],
            finish_reason="tool_calls",
            usage=Usage(prompt_tokens=10, completion_tokens=2),
        ),
        TurnResult(
            content="done",
            tool_calls=[],
            finish_reason="stop",
            usage=Usage(prompt_tokens=20, completion_tokens=2),
        ),
    ]

    def assistant_message(content, tool_calls):
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

    client.build_assistant_message.side_effect = assistant_message

    with patch("scripts.llm_solver.harness.loop._auto_commit"):
        assert solve_task(work, cfg, client, savings_dir=savings_dir) is True

    outgoing = client.chat.call_args_list[0].args[0]
    system_prompt = outgoing[0]["content"]
    assert "<skills>" in system_prompt
    assert f"visible-skill: Visible startup metadata. ({visible})" in system_prompt
    assert "VISIBLE BODY MUST BE ON DEMAND" not in system_prompt
    assert "hidden-skill" not in system_prompt
    second_request = client.chat.call_args_list[1].args[0]
    assert "VISIBLE BODY MUST BE ON DEMAND" in json.dumps(second_request)

    events = [
        json.loads(line)
        for line in trace_path(work).read_text().splitlines()
        if line.strip()
    ]
    start = next(row for row in events if row["event"] == "session_start")
    assert start["loaded_skills"] == [
        {
            "name": "visible-skill",
            "path": str(visible),
            "disable_model_invocation": False,
        },
        {
            "name": "hidden-skill",
            "path": str(hidden),
            "disable_model_invocation": True,
        },
    ]

    ledger = [
        json.loads(line)
        for line in (savings_dir / f"{work.name}.jsonl").read_text().splitlines()
    ]
    skill_cost = next(row for row in ledger if row["bucket"] == "skills_catalog")
    assert skill_cost["ctx"]["skills"] == ["visible-skill"]

    metrics = json.loads((work / "metrics.json").read_text())
    assert metrics["provenance"]["config"]["skills_readable_dirs"] == [
        str(visible.parent),
        str(hidden.parent),
    ]

    state = work / ".solver" / "state.json"
    assert state.is_file()
    assert "loaded_skills" not in state.read_text()
