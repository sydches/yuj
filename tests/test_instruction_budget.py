"""Instruction-size policy may reject a chain, but must not hand off a prefix."""
import pytest

from scripts.llm_solver.harness.project_instructions import (
    discover_project_instructions, resolve_project_instruction_imports,
)


@pytest.mark.parametrize("limit", [None, 0])
def test_default_and_unlimited_discovery_preserve_long_required_suffix(tmp_path, limit):
    (tmp_path / ".git").mkdir()
    source = "x" * 40000 + "\nREQUIRED FINAL INSTRUCTION"
    (tmp_path / "AGENTS.md").write_text(source)
    project = discover_project_instructions(tmp_path, **({} if limit is None else {"max_bytes": limit}))
    result = resolve_project_instruction_imports(project, enabled=False, max_depth=1)
    assert result.documents[0].content == source
    assert result.resolved_bytes == len(source.encode())
    assert not result.truncated


@pytest.mark.parametrize("defer", [False, True])
def test_explicit_ceiling_refuses_incomplete_instruction_chain(tmp_path, defer):
    (tmp_path / ".git").mkdir()
    child = tmp_path / "child"
    child.mkdir()
    (tmp_path / "AGENTS.md").write_text("ROOT")
    (child / "AGENTS.md").write_text("ééé")
    with pytest.raises(ValueError, match="instruction.*byte.*ceiling"):
        project = discover_project_instructions(child, max_bytes=9, defer_byte_cap=defer)
        resolve_project_instruction_imports(project, enabled=False, max_depth=1)


def test_import_expansion_is_checked_before_returning_any_prefix(tmp_path):
    (tmp_path / "AGENTS.md").write_text("@shared.md\n")
    (tmp_path / "shared.md").write_text("é" * 20)
    project = discover_project_instructions(tmp_path, max_bytes=30, defer_byte_cap=True)
    with pytest.raises(ValueError, match="instruction.*byte.*ceiling"):
        resolve_project_instruction_imports(project, enabled=True, max_depth=2)


def test_previously_truncated_input_is_not_accepted_as_a_complete_chain(tmp_path):
    from dataclasses import replace
    (tmp_path / "AGENTS.md").write_text("RULE")
    project = discover_project_instructions(tmp_path)
    with pytest.raises(ValueError, match="previously truncated"):
        resolve_project_instruction_imports(replace(project, truncated=True), enabled=False, max_depth=1)


def test_solver_rejects_byte_overflow_before_model_call(tmp_path):
    from unittest.mock import MagicMock, patch
    from _config_helpers import make_config
    from scripts.llm_solver.harness.loop import solve_task
    (tmp_path / "AGENTS.md").write_text("REQUIRED RULE")
    (tmp_path / "prompt.txt").write_text("fixture")
    cfg = make_config(max_sessions=1, project_docs_enabled=True,
                      project_doc_global_dir="", project_doc_max_bytes=3)
    client = MagicMock()
    with patch("scripts.llm_solver.harness.loop._auto_commit"):
        with pytest.raises(ValueError, match="instruction.*byte.*ceiling"):
            solve_task(tmp_path, cfg, client)
    client.chat.assert_not_called()


@pytest.mark.parametrize("capacity,accepted", [(4096, False), (65536, True)])
def test_complete_instruction_request_is_admitted_by_backend_count(tmp_path, monkeypatch, capacity, accepted):
    import json
    import httpx
    from _config_helpers import make_config
    from tests.test_backend_token_counting import attach_transport, completion
    from scripts.llm_solver.harness._loop._driver_setup import load_system_prompt_and_provenance
    from scripts.llm_solver.harness.loop import Session
    from scripts.llm_solver.server.client import LlamaClient

    source = "x" * 40000 + "\nREQUIRED FINAL INSTRUCTION"
    (tmp_path / ".git").mkdir()
    (tmp_path / "AGENTS.md").write_text(source)
    cfg = make_config(tokenizer_id="auto", context_size=capacity, max_tokens=1000,
                      max_turns=1, project_docs_enabled=True, project_doc_global_dir="",
                      model="instruction-reader")
    client = LlamaClient(cfg)
    counted, generated = [], []

    def handler(request):
        body = json.loads(request.content)
        assert body["model"] == "instruction-reader"
        assert any(source in (m.get("content") or "") for m in body["messages"])
        size = len(json.dumps([body["messages"], body.get("tools")])) // 4
        if request.url.path.endswith("/input_tokens"):
            counted.append(body)
            return httpx.Response(200, json={"input_tokens": size})
        assert request.url.path.endswith("/chat/completions")
        generated.append(body)
        return completion(size)

    attach_transport(client, handler)
    monkeypatch.setattr(client, "query_server_context", lambda: capacity)
    try:
        system, *_ = load_system_prompt_and_provenance(cfg, client, tmp_path, None, None, None, None)
        session = Session(cfg, client, system, "Read the project instructions.", str(tmp_path))
        result = session.run()
        assert counted and counted[0]["tools"] == session.model_tool_schemas
        assert bool(generated) is accepted
        assert (result.finish_reason == "context_full") is (not accepted)
        assert any(source in (m.get("content") or "") for m in session.context.get_messages())
        if accepted:
            assert generated[0]["messages"] == counted[-1]["messages"]
    finally:
        client.client.close()
