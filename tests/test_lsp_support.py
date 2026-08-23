"""Deterministic fake-server tests for the LSP support leaf."""
from __future__ import annotations

import subprocess
import sys
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from _config_helpers import make_config
from scripts.llm_solver.config import load_config
from scripts.llm_solver.harness.loop import Session
from scripts.llm_solver.harness.lsp_support import (
    DiagnosticsReport,
    LspDiagnostic,
    LspManager,
    LspServerSpec,
    LspSupportError,
    append_diagnostics_to_tool_result,
    build_lsp_sandbox_argv,
    parse_server_specs,
)
from scripts.llm_solver.server.types import ToolCall, TurnResult, Usage


_FAKE_SERVER = r'''
import json
import sys

def read_message():
    headers = {}
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            return None
        if line in (b"\r\n", b"\n"):
            break
        name, value = line.decode("ascii").split(":", 1)
        headers[name.lower()] = value.strip()
    return json.loads(sys.stdin.buffer.read(int(headers["content-length"])))

def send(value):
    payload = json.dumps(value, separators=(",", ":")).encode()
    sys.stdout.buffer.write(b"Content-Length: " + str(len(payload)).encode() + b"\r\n\r\n" + payload)
    sys.stdout.buffer.flush()

def diagnostics(params):
    document = params["textDocument"]
    text = document.get("text", "")
    if not text:
        text = params.get("contentChanges", [{}])[0].get("text", "")
    if "no_publish" in text:
        return
    items = [] if "fixed" in text else [
        {"range":{"start":{"line":1,"character":2},"end":{"line":1,"character":3}},
         "severity":1,"message":"broken <syntax>","source":"fake","code":"E1"},
        {"range":{"start":{"line":2,"character":0},"end":{"line":2,"character":1}},
         "severity":2,"message":"style warning","source":"fake","code":"W1"},
    ]
    send({"jsonrpc":"2.0","method":"textDocument/publishDiagnostics",
          "params":{"uri":document["uri"],"diagnostics":items}})

while True:
    message = read_message()
    if message is None:
        break
    method = message.get("method")
    if method == "initialize":
        send({"jsonrpc":"2.0","id":message["id"],"result":{"capabilities":{}}})
    elif method in ("textDocument/didOpen", "textDocument/didChange"):
        diagnostics(message["params"])
    elif method == "textDocument/definition":
        send({"jsonrpc":"2.0","id":message["id"],"result":{"uri":message["params"]["textDocument"]["uri"],"range":{"start":{"line":4,"character":1},"end":{"line":4,"character":2}}}})
    elif method == "textDocument/references":
        send({"jsonrpc":"2.0","id":message["id"],"result":[]})
    elif method == "textDocument/documentSymbol":
        send({"jsonrpc":"2.0","id":message["id"],"result":[{"name":"thing","kind":12}]})
    elif method == "shutdown":
        send({"jsonrpc":"2.0","id":message["id"],"result":None})
    elif method == "exit":
        break
'''


class CountingPopen:
    def __init__(self):
        self.processes = []

    def __call__(self, argv, **kwargs):
        process = subprocess.Popen(argv, **kwargs)
        self.processes.append(process)
        return process


def fake_spec(tmp_path: Path) -> LspServerSpec:
    server = tmp_path / "fake_lsp.py"
    server.write_text(_FAKE_SERVER)
    return LspServerSpec(
        name="python",
        command=(sys.executable, "-u", str(server)),
        extensions=(".py",),
        root_markers=("pyproject.toml",),
    )


def make_manager(tmp_path: Path, **overrides):
    events = overrides.pop("events", [])
    warnings = overrides.pop("warnings", [])
    popen = overrides.pop("popen", CountingPopen())
    spec = overrides.pop("spec", fake_spec(tmp_path))
    manager = LspManager(
        cwd=tmp_path,
        servers=(spec,),
        argv_builder=lambda server, root: server.command,
        diagnostics_timeout_s=overrides.pop("diagnostics_timeout_s", 1),
        min_severity=overrides.pop("min_severity", "error"),
        event_sink=events.append,
        warning_sink=warnings.append,
        popen_factory=popen,
        **overrides,
    )
    return manager, popen, events, warnings


def test_lazy_server_collects_diagnostics_and_emits_trace(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\n")
    target = tmp_path / "pkg" / "app.py"
    target.parent.mkdir()
    target.write_text("bad\ncode\n")
    manager, popen, events, warnings = make_manager(tmp_path)

    assert popen.processes == []
    report = manager.after_edit("pkg/app.py")

    assert report is not None
    assert report.errors == 1
    assert report.warnings == 1
    assert [item.message for item in report.visible] == ["broken <syntax>"]
    assert len(popen.processes) == 1
    assert manager.active_roots() == (tmp_path,)
    assert warnings == []
    assert events[-1]["event"] == "lsp_diagnostics"
    assert {key: events[-1][key] for key in ("file", "errors", "warnings")} == {
        "file": "pkg/app.py", "errors": 1, "warnings": 1,
    }
    assert isinstance(events[-1]["ms"], int)
    envelope = append_diagnostics_to_tool_result(
        "OK", report, max_output_chars=2000
    )
    assert "broken &lt;syntax&gt;" in envelope
    assert "style warning" not in envelope
    manager.close()
    assert popen.processes[0].poll() is not None


def test_second_edit_reuses_server_and_clears_diagnostics(tmp_path):
    target = tmp_path / "app.py"
    target.write_text("bad\n")
    manager, popen, _events, _warnings = make_manager(tmp_path)
    assert manager.after_edit("app.py").errors == 1

    target.write_text("fixed\n")
    report = manager.after_edit("app.py")

    assert report.errors == report.warnings == 0
    assert len(popen.processes) == 1
    manager.close()


def test_warning_threshold_includes_errors_and_warnings(tmp_path):
    (tmp_path / "app.py").write_text("bad\n")
    manager, _popen, _events, _warnings = make_manager(
        tmp_path, min_severity="warning"
    )

    report = manager.after_edit("app.py")

    assert [item.severity for item in report.visible] == [1, 2]
    manager.close()


def test_diagnostics_timeout_is_a_traced_noop(tmp_path):
    (tmp_path / "app.py").write_text("no_publish\n")
    manager, _popen, events, warnings = make_manager(
        tmp_path, diagnostics_timeout_s=0.01
    )

    report = manager.after_edit("app.py")

    assert report.status == "timeout"
    assert report.diagnostics == ()
    assert events[-1]["status"] == "timeout"
    assert warnings == []
    manager.close()


def test_missing_binary_is_noop_with_one_warning(tmp_path):
    (tmp_path / "app.py").write_text("bad\n")
    missing = LspServerSpec("python", ("definitely-no-lsp-binary",), (".py",))
    manager, popen, events, warnings = make_manager(tmp_path, spec=missing)

    first = manager.after_edit("app.py")
    second = manager.after_edit("app.py")

    assert first.status == second.status == "unavailable"
    assert first.diagnostics == second.diagnostics == ()
    assert len(warnings) == 1
    assert "continuing without it" in warnings[0]
    assert len(events) == 2
    assert popen.processes == []


def test_nonmatching_extension_is_noop_and_does_not_start_server(tmp_path):
    (tmp_path / "README.md").write_text("text\n")
    manager, popen, events, _warnings = make_manager(tmp_path)

    assert manager.after_edit("README.md") is None
    assert popen.processes == []
    assert events == []


def test_diagnostics_append_inside_envelope_and_honor_budget(tmp_path):
    report = DiagnosticsReport(
        file="app.py", server="python",
        diagnostics=(LspDiagnostic(1, "x" * 1000, 2, 3),),
        threshold=1, ms=1,
    )
    original = (
        '<tool_result tool_name="edit" status="ok" v="1">\n'
        + "original\n" * 100
        + "</tool_result>"
    )

    result = append_diagnostics_to_tool_result(
        original, report, max_output_chars=600
    )

    assert len(result) <= 600
    assert result.endswith("</tool_result>")
    assert result.index("<lsp_diagnostics") < result.rindex("</tool_result>")
    assert "[... tool result clipped for LSP diagnostics ...]" in result


def test_raw_edit_result_is_wrapped_before_diagnostics(tmp_path):
    report = DiagnosticsReport(
        "app.py", "python", (LspDiagnostic(1, "broken", 1, 1),), 1, 0
    )

    result = append_diagnostics_to_tool_result(
        "OK", report, max_output_chars=1000
    )

    assert result.startswith('<tool_result tool_name="edit" status="ok"')
    assert "<lsp_diagnostics" in result
    assert result.endswith("</tool_result>")


def test_error_default_omits_warning_text_from_model_result(tmp_path):
    report = DiagnosticsReport(
        "app.py", "python",
        (
            LspDiagnostic(1, "error text", 1, 1),
            LspDiagnostic(2, "warning text", 2, 1),
        ),
        1,
        0,
    )

    result = append_diagnostics_to_tool_result(
        "OK", report, max_output_chars=1000
    )

    assert "error text" in result
    assert "warning text" not in result
    assert 'errors="1" warnings="1"' in result


def test_navigation_queries_use_same_lazy_server(tmp_path):
    (tmp_path / "app.py").write_text("bad\n")
    manager, popen, _events, _warnings = make_manager(tmp_path, tool_enabled=True)

    definition = manager.query("definition", path="app.py", line=1, character=2)
    references = manager.query("references", path="app.py")
    symbols = manager.query("symbols", path="app.py")

    assert '"line":4' in definition.result
    assert references.result == "[]"
    assert symbols.result == '[{"kind":12,"name":"thing"}]'
    assert len(popen.processes) == 1
    manager.close()


def test_navigation_tool_gate_and_query_validation(tmp_path):
    (tmp_path / "app.py").write_text("bad\n")
    manager, _popen, _events, _warnings = make_manager(tmp_path)

    with pytest.raises(LspSupportError, match="disabled"):
        manager.query("definition", path="app.py")


def test_parse_server_specs_normalizes_and_rejects_duplicate_extensions():
    specs = parse_server_specs({
        "python": {"command": "pyright --stdio", "extensions": ["py"],
                   "root_markers": "pyproject.toml"},
    })
    assert specs[0].command == ("pyright", "--stdio")
    assert specs[0].extensions == (".py",)
    assert specs[0].root_markers == ("pyproject.toml",)

    with pytest.raises(ValueError, match="belongs to both"):
        parse_server_specs({
            "one": {"command": ["one"], "extensions": [".py"]},
            "two": {"command": ["two"], "extensions": ["py"]},
        })


def test_lsp_sandbox_argv_uses_no_network_bwrap(tmp_path, monkeypatch):
    monkeypatch.delenv("YUJ_CONTAINER", raising=False)

    argv = build_lsp_sandbox_argv(
        ["fake-lsp", "--stdio"], cwd=str(tmp_path),
        bwrap_bin="/usr/bin/bwrap", sandbox_required=True,
    )

    assert argv[0] == "/usr/bin/bwrap"
    assert "--unshare-net" in argv
    assert "--die-with-parent" in argv
    assert argv[-2:] == ["fake-lsp", "--stdio"]


def _turn(*, tool_calls=(), content="", reason="tool_calls") -> TurnResult:
    return TurnResult(
        content=content,
        tool_calls=list(tool_calls),
        finish_reason=reason,
        usage=Usage(prompt_tokens=10, completion_tokens=3),
    )


def _client(*turns: TurnResult):
    client = MagicMock()
    client.chat.side_effect = turns
    client.build_assistant_message.side_effect = lambda content, tool_calls: {
        "role": "assistant", "content": content,
    }
    return client


def test_session_appends_diagnostics_inside_edit_envelope_and_traces(tmp_path):
    target = tmp_path / "app.py"
    target.write_text("old\n")
    manager, _popen, _events, _warnings = make_manager(tmp_path)
    trace = StringIO()
    client = _client(
        _turn(tool_calls=[ToolCall(
            id="edit-1", name="edit",
            arguments={"path": "app.py", "old_str": "old", "new_str": "bad"},
        )]),
        _turn(content="done", reason="stop"),
    )
    cfg = make_config(
        max_turns=2,
        lsp_enabled=True,
        tools_unified_envelope_enabled=True,
    )
    session = Session(
        cfg, client, "system", "task", str(tmp_path),
        trace_file=trace, lsp_manager=manager,
    )
    captured: list[str] = []
    original_add = session.context.add_tool_result

    def capture(tool_call_id, result, **kwargs):
        captured.append(result)
        return original_add(tool_call_id, result, **kwargs)

    session.context.add_tool_result = capture
    manager.event_sink = lambda payload: session._emit(
        str(payload.get("event", "lsp_diagnostics")),
        session_number=session._session_number,
        turn_number=session._current_turn,
        **{key: value for key, value in payload.items() if key != "event"},
    )

    with patch.object(session, "_get_server_ctx", return_value=cfg.context_size):
        result = session.run()

    assert result.finish_reason == "stop"
    assert target.read_text() == "bad\n"
    edit_result = captured[0]
    assert edit_result.startswith('<tool_result tool_name="edit" status="ok"')
    assert edit_result.endswith("</tool_result>")
    assert edit_result.index("<lsp_diagnostics") < edit_result.rindex("</tool_result>")
    assert "broken &lt;syntax&gt;" in edit_result
    assert len(edit_result) <= cfg.max_output_chars
    diagnostics = [event for event in session._trace_events
                   if event.get("event") == "lsp_diagnostics"]
    assert len(diagnostics) == 1
    assert diagnostics[0]["turn_number"] == 0
    assert diagnostics[0]["file"] == "app.py"
    assert diagnostics[0]["errors"] == diagnostics[0]["warnings"] == 1
    assert diagnostics[0]["trace_schema_version"] == 2


def test_session_lsp_navigation_tool_uses_manager(tmp_path):
    (tmp_path / "app.py").write_text("bad\n")
    manager, popen, _events, _warnings = make_manager(tmp_path, tool_enabled=True)
    client = _client(
        _turn(tool_calls=[ToolCall(
            id="lsp-1", name="lsp",
            arguments={
                "kind": "definition", "path": "app.py",
                "line": 1, "character": 2,
            },
        )]),
        _turn(content="done", reason="stop"),
    )
    cfg = make_config(max_turns=2, lsp_tool_enabled=True)
    session = Session(cfg, client, "system", "task", str(tmp_path), lsp_manager=manager)
    captured: list[str] = []
    original_add = session.context.add_tool_result
    session.context.add_tool_result = lambda tool_call_id, result, **kwargs: (
        captured.append(result), original_add(tool_call_id, result, **kwargs)
    )[1]

    with patch.object(session, "_get_server_ctx", return_value=cfg.context_size):
        result = session.run()

    assert result.finish_reason == "stop"
    assert 'LSP definition app.py status=ok' in captured[0]
    assert '"line":4' in captured[0]
    assert len(popen.processes) == 1
    assert popen.processes[0].poll() is not None


def test_lsp_config_defaults_overlay_and_validation(tmp_path):
    defaults = load_config()
    assert defaults.lsp_enabled is False
    assert defaults.lsp_servers == {}
    assert defaults.lsp_diagnostics_timeout_s == 2.0
    assert defaults.lsp_min_severity == "error"
    assert defaults.lsp_tool_enabled is False

    overlay = tmp_path / "lsp.toml"
    overlay.write_text(
        "[lsp]\n"
        "enabled = true\n"
        "diagnostics_timeout_s = 0.25\n"
        'min_severity = "warning"\n'
        "tool_enabled = true\n"
        "[lsp.servers.python]\n"
        'command = ["fake-lsp", "--stdio"]\n'
        'extensions = [".py"]\n'
        'root_markers = ["pyproject.toml"]\n'
    )
    configured = load_config(user_config=overlay)
    assert configured.lsp_enabled is True
    assert configured.lsp_tool_enabled is True
    assert configured.lsp_diagnostics_timeout_s == 0.25
    assert configured.lsp_min_severity == "warning"
    assert configured.lsp_servers["python"]["command"] == ["fake-lsp", "--stdio"]

    overlay.write_text("[lsp]\ndiagnostics_timeout_s = -1\n")
    with pytest.raises(ValueError, match="lsp.diagnostics_timeout_s"):
        load_config(user_config=overlay)

    overlay.write_text('[lsp]\nmin_severity = "notice"\n')
    with pytest.raises(ValueError, match="invalid LSP severity"):
        load_config(user_config=overlay)
