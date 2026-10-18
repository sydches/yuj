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


def test_local_lsp_definition_does_not_disclose_outside_task_location(tmp_path):
    import json
    task = tmp_path / 'task'
    task.mkdir()
    (task / 'app.py').write_text('symbol\n')
    outside = tmp_path / 'unrelated.py'
    outside.write_text('PRIVATE_OUTSIDE_BYTES\n')
    spec = fake_spec(task)
    server = Path(spec.command[-1])
    manager, _popen, _events, _warnings = make_manager(task, spec=spec, tool_enabled=True)
    needle = '"result":{"uri":message["params"]["textDocument"]["uri"]'
    server.write_text(_FAKE_SERVER.replace(needle, '"result":{"uri":' + json.dumps(outside.as_uri())))
    try:
        response = manager.query('definition', path='app.py', line=0, character=0)
        assert 'PRIVATE_OUTSIDE_BYTES' not in response.result
        assert outside.as_uri() not in response.result
        assert response.status == 'outside_locations_removed'
    finally:
        manager.close()


def test_lsp_filters_location_shapes_preserving_valid_and_virtual_results(tmp_path):
    import json
    task = tmp_path / 'task'
    task.mkdir()
    local = task / 'local # file.py'
    local.write_text('local')
    outside = tmp_path / 'outside.py'
    outside.write_text('outside')
    link = task / 'escape.py'
    link.symlink_to(outside)
    manager, *_ = make_manager(task)
    span = {'start': {'line': 1, 'character': 2}, 'end': {'line': 3, 'character': 4}}
    valid = [{'uri': local.as_uri(), 'range': span},
             {'targetUri': 'git:/history/file.py', 'targetRange': span},
             {'name': 'local', 'kind': 12, 'location': {'uri': local.as_uri(), 'range': span}}]
    invalid = [{'uri': outside.as_uri(), 'range': span},
               {'targetUri': link.as_uri(), 'targetRange': span},
               {'name': 'outside', 'location': {'uri': outside.as_uri(), 'range': span}},
               {'uri': 'file://other-host/task/file.py', 'range': span}]
    result, omitted = manager._visible_locations(valid + invalid, None)
    assert omitted and result == valid
    assert 'outside' not in json.dumps(result)
    assert manager._visible_locations(valid, None) == (valid, False)


def test_unsandboxed_lsp_task_transport_does_not_grant_external_locations(tmp_path):
    task = tmp_path / 'task'
    task.mkdir()
    outside = tmp_path / 'outside.py'
    outside.write_text('outside')
    manager = LspManager.sandboxed(cwd=task, servers=(), bwrap_bin='unused', sandbox=False)
    result, omitted = manager._visible_locations({'uri': outside.as_uri()}, None)
    assert omitted and result is None


@pytest.mark.parametrize('launcher', ['lsp', 'background'])
def test_selected_process_backend_never_falls_back_to_host(monkeypatch, tmp_path, launcher):
    from scripts.llm_solver.harness.lsp_support import build_lsp_sandbox_argv
    from scripts.llm_solver.harness.process_manager import build_background_sandbox_argv
    from scripts.llm_solver.harness.sandbox.container_backend import ContainerRuntimeUnavailable
    monkeypatch.delenv('YUJ_CONTAINER', raising=False)
    monkeypatch.setattr('scripts.llm_solver.harness.sandbox.container_backend.shutil.which', lambda _name: None)
    build = build_lsp_sandbox_argv if launcher == 'lsp' else build_background_sandbox_argv
    command = ['command-that-must-not-run'] if launcher == 'lsp' else 'command-that-must-not-run'
    with pytest.raises(ContainerRuntimeUnavailable):
        build(command, cwd=str(tmp_path), bwrap_bin='absent', sandbox=True,
              sandbox_required=False, sandbox_backend='container', container_image='prepared:latest')
    argv = build(command, cwd=str(tmp_path), bwrap_bin='absent', sandbox=False,
                 sandbox_required=True, sandbox_backend='container')
    assert 'command-that-must-not-run' in argv


def test_local_lsp_locations_respect_ignored_symlink_targets(tmp_path):
    from scripts.llm_solver.harness.sandbox.ignore_policy import load_ignore_policy
    (tmp_path / '.yujignore').write_text('hidden.py\n')
    hidden = tmp_path / 'hidden.py'
    hidden.write_text('hidden')
    link = tmp_path / 'alias.py'
    link.symlink_to(hidden)
    manager, *_ = make_manager(tmp_path)
    assert manager._visible_locations({'uri': link.as_uri()}, load_ignore_policy(tmp_path)) == (None, True)


@pytest.mark.parametrize('mismatch', [False, True])
def test_guarded_lsp_startup_preserves_rpc_and_refuses_changed_credentials(tmp_path, native_process_identity, mismatch):
    from dataclasses import replace
    from scripts.llm_solver.harness.process_identity import guarded_process_argv
    identity = native_process_identity
    if mismatch:
        identity = replace(identity, uids=(identity.uids[0] + 1,) * 4)
    spec = fake_spec(tmp_path)
    started = tmp_path / 'server-started'
    target = tmp_path / 'app.py'
    target.write_text('bad\ncode\n')
    manager, _popen, _events, warnings = make_manager(tmp_path, spec=spec)
    Path(spec.command[-1]).write_text(
        f'from pathlib import Path\nPath({str(started)!r}).touch()\n' + _FAKE_SERVER)
    manager.argv_builder = lambda server, root: guarded_process_argv([], server.command, identity)
    try:
        report = manager.after_edit('app.py')
        if mismatch:
            assert report.status == 'unavailable' and not started.exists()
            assert any('verified process credentials' in warning for warning in warnings)
        else:
            assert report is not None and report.errors == 1 and started.exists()
            assert not warnings
    finally:
        manager.close()


def test_rejected_lsp_verification_precedes_rpc_write(tmp_path, native_process_identity, monkeypatch):
    from dataclasses import replace
    from scripts.llm_solver.harness import lsp_support
    from scripts.llm_solver.harness.process_identity import guarded_process_argv

    identity = replace(native_process_identity, uids=(native_process_identity.uids[0] + 1,) * 4)
    spec = fake_spec(tmp_path)
    started = tmp_path / 'server-started'
    Path(spec.command[-1]).write_text(
        f'from pathlib import Path\nPath({str(started)!r}).touch()\n' + _FAKE_SERVER)
    (tmp_path / 'app.py').write_text('bad\ncode\n')
    popen = CountingPopen()

    def rejected_before_client(argv, **kwargs):
        process = popen(argv, **kwargs)
        assert process.wait(timeout=2) == 77
        return process

    sends = []
    original_send = lsp_support._RpcClient._send

    def record_send(client, message):
        sends.append(message)
        return original_send(client, message)

    monkeypatch.setattr(lsp_support._RpcClient, '_send', record_send)
    manager, _, _, warnings = make_manager(tmp_path, spec=spec, popen=rejected_before_client)
    manager.argv_builder = lambda server, root: guarded_process_argv([], server.command, identity)
    try:
        assert manager.after_edit('app.py').status == 'unavailable'
        assert len(warnings) == 1 and 'verified process credentials' in warnings[0]
        assert not sends and not started.exists()
        assert manager.after_edit('app.py').status == 'unavailable'
        assert len(warnings) == 1 and len(popen.processes) == 1
    finally:
        manager.close()


def test_lsp_waits_for_delayed_verification_before_first_request(tmp_path, native_process_identity, monkeypatch):
    import queue
    import threading
    from scripts.llm_solver.harness import lsp_support
    from scripts.llm_solver.harness.process_identity import ProcessVerification, guarded_process_argv

    spec = fake_spec(tmp_path)
    argv = guarded_process_argv([], spec.command, native_process_identity)
    verification = ProcessVerification(argv)
    entered, release, admitted, sent = (threading.Event() for _ in range(4))
    original_read = verification.read

    def delayed_read(stream):
        entered.set()
        assert release.wait(2)
        original_read(stream)
        admitted.set()

    monkeypatch.setattr(verification, 'read', delayed_read)
    original_send = lsp_support._RpcClient._send

    def verified_send(client, message):
        sent.set()
        assert admitted.is_set()
        return original_send(client, message)

    monkeypatch.setattr(lsp_support._RpcClient, '_send', verified_send)
    process = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.DEVNULL, start_new_session=True)
    client = lsp_support._RpcClient(process, request_timeout_s=1, process_verification=verification)
    results = queue.Queue()

    def initialize():
        try:
            results.put(client.request('initialize', {}))
        except BaseException as error:
            results.put(error)

    request = threading.Thread(target=initialize)
    try:
        assert entered.wait(1)
        request.start()
        assert not sent.wait(0.05)
        release.set()
        request.join(2)
        assert not request.is_alive()
        assert results.get_nowait() == {'capabilities': {}}
        client.notify('initialized', {})
        assert client.request('textDocument/references', {}) == []
    finally:
        release.set()
        client.close()
        if request.ident is not None:
            request.join(2)


@pytest.mark.parametrize('operation', ['request', 'notify'])
def test_lsp_missing_verification_times_out_without_rpc_or_shutdown_wait(monkeypatch, operation):
    import time
    from scripts.llm_solver.harness import lsp_support
    from scripts.llm_solver.harness.process_identity import GuardedProcessArgv, ProcessVerification

    argv = GuardedProcessArgv([sys.executable, '-c', 'import time; time.sleep(30)'])
    process = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.DEVNULL, start_new_session=True)
    client = lsp_support._RpcClient(process, request_timeout_s=0.05,
                                   process_verification=ProcessVerification(argv))
    sends = MagicMock(side_effect=AssertionError('RPC sent before verification'))
    monkeypatch.setattr(client, '_send', sends)
    start = time.monotonic()
    try:
        with pytest.raises(TimeoutError, match='LSP response timed out'):
            getattr(client, operation)('initialize', {})
    finally:
        client.close()
    assert time.monotonic() - start < 0.75
    assert process.poll() is not None
    sends.assert_not_called()


def test_lsp_startup_and_response_share_deadline_and_queued_ready_can_expire(monkeypatch):
    from io import BytesIO
    from types import SimpleNamespace
    from scripts.llm_solver.harness import lsp_support

    monkeypatch.setattr(lsp_support.threading, 'Thread', MagicMock())
    process = SimpleNamespace(stdin=BytesIO(), stdout=BytesIO())
    client = lsp_support._RpcClient(process, request_timeout_s=1,
                                   process_verification=SimpleNamespace(verified=False))
    client._messages.put(client._STARTUP_READY)
    monotonic = MagicMock(side_effect=[10.0, 10.5, 10.8, 10.9])
    monkeypatch.setattr(lsp_support.time, 'monotonic', monotonic)
    receive = MagicMock(wraps=client._receive)
    monkeypatch.setattr(client, '_receive', receive)
    with pytest.raises(TimeoutError, match='LSP response timed out'):
        client.request('initialize', {})
    assert [call.args[0] for call in receive.call_args_list] == pytest.approx([0.5, 0.1])
    assert b'initialize' in process.stdin.getvalue()

    process.stdin = BytesIO()
    client = lsp_support._RpcClient(process, request_timeout_s=1,
                                   process_verification=SimpleNamespace(verified=False))
    client._messages.put(client._STARTUP_READY)
    monotonic.side_effect = [20.0, 20.5, 21.1]
    with pytest.raises(TimeoutError, match='LSP response timed out'):
        client.request('initialize', {})
    assert not process.stdin.getvalue()


def test_lsp_drain_retains_startup_readiness_and_failure(monkeypatch):
    from io import BytesIO
    from types import SimpleNamespace
    from scripts.llm_solver.harness import lsp_support
    from scripts.llm_solver.harness.process_identity import ProcessIdentityError

    monkeypatch.setattr(lsp_support.threading, 'Thread', MagicMock())
    process = SimpleNamespace(stdin=BytesIO(), stdout=BytesIO())
    client = lsp_support._RpcClient(process, request_timeout_s=1,
                                   process_verification=SimpleNamespace(verified=False))
    client._messages.put(client._STARTUP_READY)
    client.drain()
    client.notify('initialized', {})
    assert b'initialized' in process.stdin.getvalue()

    process.stdin = BytesIO()
    client = lsp_support._RpcClient(process, request_timeout_s=1,
                                   process_verification=SimpleNamespace(verified=False))
    error = ProcessIdentityError('task command did not start with verified process credentials')
    client._messages.put(lsp_support._ReaderFailure(error))
    for operation in (client.drain, lambda: client.request('initialize', {}),
                      lambda: client.notify('initialized', {})):
        with pytest.raises(LspSupportError, match='verified process credentials') as caught:
            operation()
        assert caught.value.__cause__ is error
    assert not process.stdin.getvalue()


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


def test_container_lsp_keeps_stdin_and_the_credential_frame_contract(
    tmp_path, monkeypatch, native_process_identity,
):
    import os
    from scripts.llm_solver.harness import task_environment
    from scripts.llm_solver.harness.process_identity import GuardedProcessArgv
    from scripts.llm_solver.harness._tools._run_in_sandbox import _execute
    selected = task_environment.TaskEnvironment(
        str(tmp_path), '/selected/task', ('/selected/task',),
        container='named-task', container_id='c' * 64, process_identity=native_process_identity,
    )
    monkeypatch.setenv('YUJ_CONTAINER', 'named-task')
    monkeypatch.setattr(task_environment, 'discover_task_environment', lambda cwd: selected)
    argv = build_lsp_sandbox_argv(
        ['cat'], cwd=str(tmp_path), bwrap_bin='unused',
        effective_env={'PATH': os.environ['PATH']},
    )
    assert isinstance(argv, GuardedProcessArgv)
    assert argv[:6] == ['docker', 'exec', '-i', '--workdir', '/selected/task', 'c' * 64]
    request = b'Content-Length: 2\r\n\r\n{}'
    result = _execute(GuardedProcessArgv(argv[6:]), timeout=3, input_bytes=request, binary=True)
    assert result.returncode == 0 and result.stdout == request


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
