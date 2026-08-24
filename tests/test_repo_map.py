"""Focused acceptance coverage for ranked repository-map construction."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import sys
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from _config_helpers import make_config
from llm_solver.config import load_config
from llm_solver.harness.context import FullTranscript
from llm_solver.harness.context_contract import build_context_contract
from llm_solver.harness.loop import Session, solve_task
from llm_solver.harness.repo_map import build_repo_map
from llm_solver.server.types import ToolCall, TurnResult, Usage


class _CharacterTokenizer:
    """Exact deterministic tokenizer fixture: one content character/token."""

    def __init__(self) -> None:
        self.calls = 0

    def count(self, messages, tools=None):
        self.calls += 1
        assert tools is None
        return sum(len(str(message.get("content") or "")) for message in messages)


def _write_rank_fixture(root: Path) -> None:
    source = root / "src"
    source.mkdir()
    (source / "focus.py").write_text(
        "from important import important\n\n"
        "def run():\n"
        "    return important()\n",
        encoding="utf-8",
    )
    (source / "important.py").write_text(
        "def important(value: int = 1) -> int:\n"
        "    return value\n",
        encoding="utf-8",
    )
    (source / "unused.py").write_text(
        "def unused(value: int = 1) -> int:\n"
        "    return value\n",
        encoding="utf-8",
    )
    (source / "noise.py").write_text(
        "from distractor import distractor\n\n"
        "def churn():\n"
        "    return distractor() + distractor() + distractor()\n",
        encoding="utf-8",
    )
    (source / "distractor.py").write_text(
        "def distractor(value: int = 1) -> int:\n"
        "    return value\n",
        encoding="utf-8",
    )


def test_map_is_local_tokenizer_bounded_and_deterministic_for_fixed_tree(tmp_path):
    _write_rank_fixture(tmp_path)
    tokenizer = _CharacterTokenizer()
    kwargs = dict(
        task_message="Fix src/focus.py",
        token_budget=245,
        refresh="always",
        cache_dir=tmp_path / "cache",
        tokenizer=tokenizer,
    )

    first = build_repo_map(tmp_path, **kwargs)
    second = build_repo_map(tmp_path, **kwargs)

    assert first.content == second.content
    assert first.sha256 == second.sha256
    assert first.tokens == len("\n\n" + first.content)
    assert 0 < first.tokens <= kwargs["token_budget"]
    assert first.symbols > 0
    assert tokenizer.calls > 2


def test_task_named_file_references_rank_above_unreferenced_symbols(tmp_path):
    _write_rank_fixture(tmp_path)

    focus_result = build_repo_map(
        tmp_path,
        task_message="Repair the caller in src/focus.py.",
        token_budget=2_000,
        refresh="always",
        tokenizer=_CharacterTokenizer(),
    )
    noise_result = build_repo_map(
        tmp_path,
        task_message="Repair the caller in src/noise.py.",
        token_budget=2_000,
        refresh="always",
        tokenizer=_CharacterTokenizer(),
    )

    assert "important" in focus_result.content
    assert "distractor" in focus_result.content
    assert "unused" in focus_result.content
    assert focus_result.content.index("important") < focus_result.content.index(
        "distractor"
    )
    assert focus_result.content.index("important") < focus_result.content.index(
        "unused"
    )
    # Naming the other caller reverses the two referenced-symbol ranks. This
    # proves task-path personalization rather than global popularity alone.
    assert noise_result.content.index("distractor") < noise_result.content.index(
        "important"
    )


def test_files_refresh_uses_content_hash_and_cache_stays_path_private(tmp_path):
    source = tmp_path / "module.py"
    source.write_text("def alpha():\n    return 1\n", encoding="utf-8")
    cache_dir = tmp_path / "cache"
    common = dict(
        task_message="Fix module.py",
        token_budget=1_000,
        refresh="files",
        cache_dir=cache_dir,
        tokenizer=_CharacterTokenizer(),
    )

    first = build_repo_map(tmp_path, **common)
    second = build_repo_map(tmp_path, **common)
    original_stat = source.stat()
    source.write_text("def bravo():\n    return 1\n", encoding="utf-8")
    os.utime(
        source,
        ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
    )
    changed = build_repo_map(tmp_path, **common)

    assert first.cache_hit is False
    assert second.cache_hit is True
    assert changed.cache_hit is False
    assert "bravo" in changed.content and "alpha" not in changed.content
    cache_text = (cache_dir / "symbols.v1.json").read_text(encoding="utf-8")
    assert str(tmp_path) not in cache_text


def test_refresh_policies_auto_always_and_manual(tmp_path):
    source = tmp_path / "module.py"
    source.write_text("def alpha():\n    return 1\n", encoding="utf-8")
    cache_dir = tmp_path / "cache"
    common = dict(
        task_message="Fix module.py",
        token_budget=1_000,
        cache_dir=cache_dir,
        tokenizer=_CharacterTokenizer(),
    )

    assert build_repo_map(tmp_path, refresh="auto", **common).cache_hit is False
    assert build_repo_map(tmp_path, refresh="auto", **common).cache_hit is True
    source.write_text("def bravo():\n    return 1\n", encoding="utf-8")
    stat = source.stat()
    os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    changed = build_repo_map(tmp_path, refresh="auto", **common)
    assert changed.cache_hit is False
    assert "bravo" in changed.content and "alpha" not in changed.content

    assert build_repo_map(tmp_path, refresh="always", **common).cache_hit is False
    source.write_text("def charm():\n    return 1\n", encoding="utf-8")
    manual = build_repo_map(tmp_path, refresh="manual", **common)
    assert manual.cache_hit is True
    assert "bravo" in manual.content and "charm" not in manual.content


def test_map_honors_unreadable_files(tmp_path):
    visible = tmp_path / "visible.py"
    secret = tmp_path / "secret.py"
    visible.write_text("def visible():\n    return 1\n", encoding="utf-8")
    secret.write_text("def hidden_secret():\n    return 2\n", encoding="utf-8")

    result = build_repo_map(
        tmp_path,
        task_message="Fix visible.py",
        token_budget=1_000,
        refresh="always",
        unreadable_paths=(str(secret),),
        tokenizer=_CharacterTokenizer(),
    )

    assert "visible" in result.content
    assert "hidden_secret" not in result.content


def test_repo_map_config_defaults_overlay_validation_and_context_contract(tmp_path):
    defaults = load_config()
    assert defaults.repo_map_tokens == 0
    assert defaults.repo_map_refresh == "auto"

    overlay = tmp_path / "map.toml"
    overlay.write_text(
        "[context]\nrepo_map_tokens = 321\nrepo_map_refresh = 'files'\n",
        encoding="utf-8",
    )
    configured = load_config(user_config=overlay)
    assert configured.repo_map_tokens == 321
    assert configured.repo_map_refresh == "files"
    contract = build_context_contract(FullTranscript, configured)
    assert contract["repo_map"] == {
        "enabled": True,
        "token_budget": 321,
        "refresh": "files",
        "placement": "task_message_suffix",
        "stable_for_session": True,
        "source": "live_workspace_structural_index",
    }

    for invalid in ("-1", "true", "1.5"):
        overlay.write_text(
            f"[context]\nrepo_map_tokens = {invalid}\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="context.repo_map_tokens"):
            load_config(user_config=overlay)
    overlay.write_text(
        "[context]\nrepo_map_refresh = 'sometimes'\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="context.repo_map_refresh"):
        load_config(user_config=overlay)


def _turn_with_read() -> TurnResult:
    return TurnResult(
        content="Inspect the named entry point.",
        tool_calls=[
            ToolCall(
                id="call-read",
                name="read",
                arguments={"path": "src/entry.py"},
            )
        ],
        finish_reason="tool_calls",
        usage=Usage(prompt_tokens=20, completion_tokens=4),
    )


def _done_turn() -> TurnResult:
    return TurnResult(
        content="Done.",
        tool_calls=[],
        finish_reason="stop",
        usage=Usage(prompt_tokens=30, completion_tokens=3),
    )


def test_runtime_injects_stable_profile_facing_prefix_and_traces_only_metadata(
    tmp_path,
):
    work = tmp_path / "work"
    artifacts = tmp_path / "artifacts"
    (work / "src").mkdir(parents=True)
    (work / "src" / "entry.py").write_text(
        "def entry():\n    return helper()\n",
        encoding="utf-8",
    )
    (work / "src" / "helper.py").write_text(
        "def helper():\n    return 1\n",
        encoding="utf-8",
    )
    (work / "src" / "hidden.py").write_text(
        "def hidden_answer():\n    return 42\n",
        encoding="utf-8",
    )
    (work / ".yujignore").write_text("src/hidden.py\n", encoding="utf-8")
    client = MagicMock()
    seen_messages: list[list[dict]] = []
    responses = iter((_turn_with_read(), _done_turn()))

    def chat(messages, _tools, *, turn):
        assert turn in {0, 1}
        seen_messages.append(copy.deepcopy(messages))
        return next(responses)

    client.chat.side_effect = chat
    client.build_assistant_message.side_effect = [
        {
            "role": "assistant",
            "content": "Inspect the named entry point.",
            "tool_calls": [{
                "id": "call-read",
                "type": "function",
                "function": {
                    "name": "read",
                    "arguments": json.dumps({"path": "src/entry.py"}),
                },
            }],
        },
        {"role": "assistant", "content": "Done."},
    ]
    cfg = make_config(
        max_sessions=1,
        max_turns=3,
        duplicate_abort=10,
        repo_map_tokens=400,
        repo_map_refresh="files",
        state_writer_enabled=True,
    )

    with (
        patch("llm_solver.harness.loop._auto_commit"),
        patch.object(Session, "_get_server_ctx", return_value=0),
    ):
        assert solve_task(
            work,
            cfg,
            client,
            context_class=FullTranscript,
            initial_prompt="Fix src/entry.py and verify it.",
            artifacts_dir=artifacts,
        )

    assert client.chat.call_count == 2
    first_messages, second_messages = seen_messages
    assert [message["role"] for message in first_messages] == ["system", "user"]
    task = first_messages[1]["content"]
    assert task.index("Fix src/entry.py") < task.index("<repo-map>")
    assert task.endswith("</repo-map>")
    assert "hidden_answer" not in task
    # The canonical system+task messages are byte-identical on the next turn;
    # profile denormalization receives the map before the changing tail.
    assert second_messages[:2] == first_messages

    events = [
        json.loads(line)
        for line in (artifacts / ".trace.jsonl").read_text().splitlines()
    ]
    start = next(event for event in events if event["event"] == "session_start")
    assert 0 < start["repo_map_tokens"] <= cfg.repo_map_tokens
    assert start["repo_map_refresh"] == "files"
    assert start["repo_map_symbols"] > 0
    assert len(start["repo_map_sha256"]) == 64
    assert start["context_contract"]["repo_map"]["stable_for_session"] is True
    trace_text = (artifacts / ".trace.jsonl").read_text()
    assert "<repo-map>" not in trace_text

    state_text = (artifacts / ".solver" / "state.json").read_text()
    assert "repo_map_tokens" not in state_text
    assert "<repo-map>" not in state_text
    assert (artifacts / ".repo_map_cache" / "symbols.v1.json").is_file()


def test_measurement_cache_is_run_private_and_outside_task_root(tmp_path):
    run_dir = tmp_path / "run"
    work = run_dir / "repos" / "task"
    work.mkdir(parents=True)
    (work / "entry.py").write_text(
        "def entry():\n    return 1\n",
        encoding="utf-8",
    )
    client = MagicMock()
    client.chat.return_value = _done_turn()
    client.build_assistant_message.return_value = {
        "role": "assistant",
        "content": "Done.",
    }
    cfg = make_config(
        max_sessions=1,
        max_turns=1,
        repo_map_tokens=200,
        repo_map_refresh="auto",
    )

    with (
        patch("llm_solver.harness.loop._auto_commit"),
        patch.object(Session, "_get_server_ctx", return_value=0),
    ):
        assert solve_task(
            work,
            cfg,
            client,
            context_class=FullTranscript,
            initial_prompt="Fix entry.py.",
            run_metadata={"run_dir": str(run_dir)},
        )

    cache_files = list(
        (run_dir / ".repo_map_cache").glob("*/symbols.v1.json")
    )
    assert len(cache_files) == 1
    assert work not in cache_files[0].parents
    assert not (work / ".repo_map_cache").exists()
