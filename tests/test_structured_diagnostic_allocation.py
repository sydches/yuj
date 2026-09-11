"""Long diagnostics reach shared request admission and keep their saved source."""
import json
import re
from unittest.mock import MagicMock

import pytest

from tests._config_helpers import make_config
from tests.test_structured_output_retention import projection
from scripts.llm_solver.harness._loop.state_projection import project_and_sink
from scripts.llm_solver.harness.loop import Session
from scripts.llm_solver.harness.tools import dispatch
from scripts.llm_solver.server.types import TurnResult, Usage


class FixtureCounter:
    """Synthetic backend-count contract, not a real tokenizer measurement."""
    def __init__(self):
        self.last = {}
        self.tools_seen = []

    def count(self, messages, tools=None):
        self.tools_seen.append(tools)
        value = len(json.dumps([messages, tools])) // 4
        self.last = {"count_basis": "backend_input_tokens",
                     "count_precision": "backend_reported", "prompt_tokens": value}
        return value


@pytest.mark.parametrize("context_size,clipped", [(8192, True), (65536, False)])
def test_request_space_selects_display_and_full_diagnostic_stays_readable(
    projection, monkeypatch, context_size, clipped,
):
    owner, _, _ = projection
    raw = "FAILED case_one " + "x" * 32000 + " REQUIRED_DIAGNOSTIC " + "y" * 32000 + "\n1 failed\n"
    projected = project_and_sink(owner, "bash", "check", raw, 3)
    assert "REQUIRED_DIAGNOSTIC" in projected
    pointer = re.search(r'full_path="([^"]+)"', projected)[1]
    from scripts.llm_solver.harness.savings import _NullLedger
    ledger = _NullLedger()
    monkeypatch.setattr("scripts.llm_solver.harness.savings.get_ledger", lambda: ledger)
    cfg = make_config(context_size=context_size, max_turns=1, sandbox_bash=False,
                      max_output_chars=100000, context_fill_ratio=0.95)
    client = MagicMock()
    client.query_server_context.return_value = context_size
    client.chat.return_value = TurnResult(content="reported", tool_calls=[],
                                         finish_reason="stop", usage=Usage(10, 5))
    counter = FixtureCounter()
    session = Session(cfg, client, "system", "inspect the supplied check result",
                      str(owner.cwd), local_tokenizer=counter)
    session.context.add_assistant({"role": "assistant", "content": "",
        "tool_calls": [{"id": "c1", "type": "function", "function": {
            "name": "bash", "arguments": '{"cmd":"check"}'}}]})
    session.context.add_tool_result("c1", projected, tool_name="bash")
    result = session.run()
    assert result.finish_reason == "stop"
    client.chat.assert_called_once()
    outgoing = client.chat.call_args.args[0]
    body = next(m["content"] for m in outgoing if m.get("role") == "tool")
    assert ("HARNESS re-clip" in body) is clipped
    assert ("REQUIRED_DIAGNOSTIC" in body) is not clipped
    assert pointer in body
    tools = next(item for item in counter.tools_seen if item)
    assert counter.count(outgoing, tools=tools) <= int(context_size * cfg.context_fill_ratio)
    readback = dispatch("read", {"path": pointer}, cwd=str(owner.cwd), cfg=cfg)
    assert "REQUIRED_DIAGNOSTIC" in readback
    assert (owner.cwd / pointer).read_text() == raw
