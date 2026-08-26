"""Lazy, sandboxable Language Server Protocol support.

This leaf owns stdio JSON-RPC, server selection/root discovery, document sync,
diagnostic filtering/rendering, navigation queries, and server teardown.  The
tool layer decides when to call it and supplies trace/system-warning sinks.
"""
from __future__ import annotations

import html
import json
import logging
import os
import queue
import shlex
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

from ._tools._common import _resolve


log = logging.getLogger(__name__)
EventSink = Callable[[dict[str, object]], None]
WarningSink = Callable[[str], None]
ArgvBuilder = Callable[["LspServerSpec", Path], Sequence[str]]
_SEVERITY = {"error": 1, "warning": 2, "information": 3, "info": 3, "hint": 4}
_SEVERITY_NAME = {1: "error", 2: "warning", 3: "information", 4: "hint"}


class LspSupportError(RuntimeError):
    """Configuration, protocol, or model-actionable LSP failure."""


class _ReaderFailure:
    def __init__(self, error: BaseException) -> None:
        self.error = error


@dataclass(frozen=True)
class LspServerSpec:
    name: str
    command: tuple[str, ...]
    extensions: tuple[str, ...]
    root_markers: tuple[str, ...] = ()
    initialization: Mapping[str, object] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, name: str, value: Mapping[str, object]) -> "LspServerSpec":
        raw_command = value.get("command", ())
        if isinstance(raw_command, str):
            command = tuple(shlex.split(raw_command))
        elif isinstance(raw_command, Sequence):
            command = tuple(str(part) for part in raw_command)
        else:
            command = ()
        raw_extensions = value.get("extensions", ())
        if isinstance(raw_extensions, str):
            raw_extensions = (raw_extensions,)
        extensions = tuple(
            extension if str(extension).startswith(".") else f".{extension}"
            for extension in raw_extensions
        )
        raw_markers = value.get("root_markers", ())
        if isinstance(raw_markers, str):
            raw_markers = (raw_markers,)
        if not command or not extensions:
            raise ValueError(f"lsp server {name!r} needs command and extensions")
        initialization = value.get("initialization", {})
        if not isinstance(initialization, Mapping):
            raise ValueError(f"lsp server {name!r} initialization must be a table")
        return cls(
            name=name,
            command=command,
            extensions=tuple(str(item).lower() for item in extensions),
            root_markers=tuple(str(item) for item in raw_markers),
            initialization=dict(initialization),
        )


def parse_server_specs(values: Mapping[str, Mapping[str, object]]) -> tuple[LspServerSpec, ...]:
    specs = tuple(LspServerSpec.from_mapping(name, value) for name, value in values.items())
    owners: dict[str, str] = {}
    for spec in specs:
        for extension in spec.extensions:
            if extension in owners:
                raise ValueError(
                    f"lsp extension {extension!r} belongs to both "
                    f"{owners[extension]!r} and {spec.name!r}"
                )
            owners[extension] = spec.name
    return specs


@dataclass(frozen=True)
class LspDiagnostic:
    severity: int
    message: str
    line: int
    character: int
    source: str = ""
    code: str = ""

    @classmethod
    def from_payload(cls, value: Mapping[str, object]) -> "LspDiagnostic":
        start = value.get("range", {})
        start = start.get("start", {}) if isinstance(start, Mapping) else {}
        return cls(
            severity=int(value.get("severity") or 2),
            message=str(value.get("message", "")),
            line=int(start.get("line", 0)) + 1,
            character=int(start.get("character", 0)) + 1,
            source=str(value.get("source", "")),
            code=str(value.get("code", "")),
        )


@dataclass(frozen=True)
class DiagnosticsReport:
    file: str
    server: str
    diagnostics: tuple[LspDiagnostic, ...]
    threshold: int
    ms: int
    status: str = "ok"

    @property
    def errors(self) -> int:
        return sum(item.severity == 1 for item in self.diagnostics)

    @property
    def warnings(self) -> int:
        return sum(item.severity == 2 for item in self.diagnostics)

    @property
    def visible(self) -> tuple[LspDiagnostic, ...]:
        return tuple(item for item in self.diagnostics if item.severity <= self.threshold)


@dataclass(frozen=True)
class LspQueryResult:
    kind: str
    file: str
    result: str
    status: str = "ok"


def build_lsp_sandbox_argv(
    command: Sequence[str], *, cwd: str, bwrap_bin: str,
    unreadable_paths: tuple[str, ...] = (), sandbox_required: bool = True,
    readable_paths: tuple[str, ...] = (),
    sandbox: bool = True, sandbox_backend: str = "bwrap",
    container_runtime: str = "docker", container_runtime_bin: str = "",
    container_image: str = "",
    container_flags: tuple[str, ...] = (),
    effective_env: Mapping[str, str] | None = None,
    allow_login_shell: bool = False,
) -> list[str]:
    """Run a stdio server under the same no-network policy as model bash."""
    from .sandbox import AMBIENT_CONTAINER, _build_bwrap_argv, container_mode

    command = tuple(command)
    command_text = shlex.join(command)
    from .sandbox.env_policy import build_clean_exec_argv

    def explicit(argv: list[str]) -> list[str]:
        return (
            argv if effective_env is None
            else build_clean_exec_argv(argv, effective_env)
        )

    if not sandbox:
        return explicit(list(command))
    if sandbox_backend == "container":
        if container_mode() is not None:
            raise LspSupportError(
                "sandbox.backend='container' cannot be combined with "
                "legacy YUJ_CONTAINER"
            )
        from .sandbox.container_backend import ContainerBackend

        backend = ContainerBackend(
            runtime=container_runtime,
            image=container_image,
            flags=container_flags,
        )
        runtime_bin = (
            container_runtime_bin
            or backend.resolve_runtime(sandbox_required=True)
        )
        assert runtime_bin is not None
        return backend.build_argv(
            command_text,
            cwd,
            runtime_bin=runtime_bin,
            unreadable_paths=unreadable_paths,
            readable_paths=readable_paths,
            sandbox_required=True,
            effective_env=effective_env,
            allow_login_shell=allow_login_shell,
        )
    if sandbox_backend != "bwrap":
        raise LspSupportError(f"unknown sandbox backend {sandbox_backend!r}")
    if container_mode() == AMBIENT_CONTAINER:
        from ._tools._run_in_sandbox import _probe_ambient_unshare_net

        prefix = ["unshare", "-n"] if _probe_ambient_unshare_net() else []
        return [*prefix, *explicit(list(command))]
    return _build_bwrap_argv(
        command_text, cwd, bwrap_bin,
        unreadable_paths=unreadable_paths,
        readable_paths=readable_paths,
        sandbox_required=sandbox_required,
        effective_env=effective_env,
        allow_login_shell=allow_login_shell,
        tail=list(command),
    )


def _read_rpc_message(stream) -> dict[str, object] | None:
    headers: dict[str, str] = {}
    while True:
        line = stream.readline()
        if not line:
            return None
        if line in {b"\r\n", b"\n"}:
            break
        name, separator, value = line.decode("ascii").partition(":")
        if not separator:
            raise LspSupportError("malformed LSP header")
        headers[name.strip().lower()] = value.strip()
    try:
        length = int(headers["content-length"])
    except (KeyError, ValueError) as exc:
        raise LspSupportError("missing or invalid LSP Content-Length") from exc
    payload = stream.read(length)
    if len(payload) != length:
        raise LspSupportError("truncated LSP payload")
    value = json.loads(payload.decode("utf-8"))
    if not isinstance(value, dict):
        raise LspSupportError("LSP payload must be an object")
    return value


class _RpcClient:
    def __init__(self, process, *, request_timeout_s: float) -> None:
        if process.stdin is None or process.stdout is None:
            raise LspSupportError("LSP process needs stdin and stdout pipes")
        self.process = process
        self.stdin = process.stdin
        self.stdout = process.stdout
        self.request_timeout_s = request_timeout_s
        self._messages: queue.Queue[object] = queue.Queue()
        self._write_lock = threading.Lock()
        self._next_id = 1
        self._responses: dict[int, dict[str, object]] = {}
        self._diagnostics: dict[str, tuple[int, Mapping[str, object]]] = {}
        self._diagnostic_generation = 0
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

    def _read_loop(self) -> None:
        try:
            while True:
                message = _read_rpc_message(self.stdout)
                if message is None:
                    raise LspSupportError("LSP server closed stdout")
                self._messages.put(message)
        except BaseException as exc:
            self._messages.put(_ReaderFailure(exc))

    def _send(self, message: Mapping[str, object]) -> None:
        payload = json.dumps(message, separators=(",", ":")).encode("utf-8")
        frame = f"Content-Length: {len(payload)}\r\n\r\n".encode("ascii") + payload
        with self._write_lock:
            self.stdin.write(frame)
            self.stdin.flush()

    def notify(self, method: str, params: Mapping[str, object]) -> None:
        self._send({"jsonrpc": "2.0", "method": method, "params": params})

    def _route(self, message: Mapping[str, object]) -> None:
        method = message.get("method")
        if method == "textDocument/publishDiagnostics":
            params = message.get("params", {})
            if isinstance(params, Mapping):
                uri = str(params.get("uri", ""))
                self._diagnostic_generation += 1
                self._diagnostics[uri] = (self._diagnostic_generation, params)
            return
        message_id = message.get("id")
        if method and message_id is not None:
            result: object = None
            if method == "workspace/configuration":
                params = message.get("params", {})
                items = params.get("items", []) if isinstance(params, Mapping) else []
                result = [None for _item in items]
            self._send({"jsonrpc": "2.0", "id": message_id, "result": result})
        elif isinstance(message_id, int):
            self._responses[message_id] = dict(message)

    def _receive(self, timeout: float) -> None:
        try:
            message = self._messages.get(timeout=max(0.0, timeout))
        except queue.Empty as exc:
            raise TimeoutError("LSP response timed out") from exc
        if isinstance(message, _ReaderFailure):
            raise LspSupportError(str(message.error)) from message.error
        if isinstance(message, Mapping):
            self._route(message)

    def drain(self) -> None:
        """Route messages already queued before a new document generation."""
        while True:
            try:
                message = self._messages.get_nowait()
            except queue.Empty:
                return
            if isinstance(message, _ReaderFailure):
                raise LspSupportError(str(message.error)) from message.error
            if isinstance(message, Mapping):
                self._route(message)

    def request(self, method: str, params: Mapping[str, object], *, timeout_s=None):
        request_id = self._next_id
        self._next_id += 1
        self._send({
            "jsonrpc": "2.0", "id": request_id, "method": method, "params": params,
        })
        timeout = self.request_timeout_s if timeout_s is None else float(timeout_s)
        deadline = time.monotonic() + timeout
        while request_id not in self._responses:
            self._receive(deadline - time.monotonic())
        response = self._responses.pop(request_id)
        if "error" in response:
            raise LspSupportError(f"LSP {method} failed: {response['error']}")
        return response.get("result")

    def diagnostic_generation(self, uri: str) -> int:
        return self._diagnostics.get(uri, (0, {}))[0]

    def wait_for_diagnostics(
        self, uri: str, *, after: int, timeout_s: float,
    ) -> Mapping[str, object] | None:
        deadline = time.monotonic() + timeout_s
        while self.diagnostic_generation(uri) <= after:
            try:
                self._receive(deadline - time.monotonic())
            except TimeoutError:
                return None
        return self._diagnostics[uri][1]

    def close(self) -> None:
        if self.process.poll() is None:
            try:
                self.request("shutdown", {}, timeout_s=min(1.0, self.request_timeout_s))
                self.notify("exit", {})
            except Exception:
                pass
            try:
                try:
                    os.killpg(self.process.pid, signal.SIGTERM)
                except (AttributeError, OSError, ProcessLookupError):
                    self.process.terminate()
                self.process.wait(timeout=1)
            except Exception:
                try:
                    try:
                        os.killpg(self.process.pid, signal.SIGKILL)
                    except (AttributeError, OSError, ProcessLookupError):
                        self.process.kill()
                    self.process.wait(timeout=1)
                except Exception:
                    pass


@dataclass
class _ServerState:
    spec: LspServerSpec
    root: Path
    rpc: _RpcClient
    versions: dict[str, int] = field(default_factory=dict)


class LspManager:
    """Lazy LSP server pool scoped to one solver session."""

    def __init__(
        self, *, cwd: str | Path, servers: Iterable[LspServerSpec],
        argv_builder: ArgvBuilder, diagnostics_timeout_s: float = 2.0,
        min_severity: str | int = "error", enabled: bool = True,
        tool_enabled: bool = False, event_sink: EventSink | None = None,
        warning_sink: WarningSink | None = None,
        popen_factory: Callable[..., object] = subprocess.Popen,
    ) -> None:
        if diagnostics_timeout_s < 0:
            raise ValueError("lsp diagnostics timeout must be >= 0")
        if isinstance(min_severity, str):
            try:
                threshold = _SEVERITY[min_severity.lower()]
            except KeyError:
                raise ValueError(f"invalid LSP severity {min_severity!r}") from None
        else:
            threshold = int(min_severity)
            if threshold not in _SEVERITY_NAME:
                raise ValueError("LSP severity must be 1 through 4")
        self.cwd = Path(cwd).resolve()
        self.servers = tuple(servers)
        self.argv_builder = argv_builder
        self.diagnostics_timeout_s = float(diagnostics_timeout_s)
        self.threshold = threshold
        self.enabled = bool(enabled)
        self.tool_enabled = bool(tool_enabled)
        self.event_sink = event_sink
        self.warning_sink = warning_sink
        self.popen_factory = popen_factory
        self._states: dict[tuple[str, Path], _ServerState] = {}
        self._unavailable: set[tuple[str, Path]] = set()
        self._warned: set[str] = set()
        self._lock = threading.RLock()

    @classmethod
    def sandboxed(
        cls, *, cwd: str | Path, servers: Iterable[LspServerSpec],
        bwrap_bin: str, unreadable_paths: tuple[str, ...] = (),
        readable_paths: tuple[str, ...] = (),
        sandbox_required: bool = True, sandbox: bool = True,
        sandbox_backend: str = "bwrap", container_runtime: str = "docker",
        container_runtime_bin: str = "",
        container_image: str = "", container_flags: tuple[str, ...] = (),
        effective_env: Mapping[str, str] | None = None,
        allow_login_shell: bool = False,
        **kwargs,
    ) -> "LspManager":
        cwd_text = str(Path(cwd).resolve())

        def argv_builder(spec: LspServerSpec, _root: Path) -> list[str]:
            return build_lsp_sandbox_argv(
                spec.command, cwd=cwd_text, bwrap_bin=bwrap_bin,
                unreadable_paths=unreadable_paths,
                readable_paths=readable_paths,
                sandbox_required=sandbox_required,
                sandbox=sandbox,
                sandbox_backend=sandbox_backend,
                container_runtime=container_runtime,
                container_runtime_bin=container_runtime_bin,
                container_image=container_image,
                container_flags=container_flags,
                effective_env=effective_env,
                allow_login_shell=allow_login_shell,
            )

        return cls(cwd=cwd_text, servers=servers, argv_builder=argv_builder, **kwargs)

    def _emit(self, report: DiagnosticsReport) -> None:
        if self.event_sink is not None:
            self.event_sink({
                "event": "lsp_diagnostics", "file": report.file,
                "errors": report.errors, "warnings": report.warnings,
                "ms": report.ms, "server": report.server, "status": report.status,
            })

    def _warn_once(self, spec: LspServerSpec, detail: str) -> None:
        if spec.name in self._warned:
            return
        self._warned.add(spec.name)
        message = f"LSP server {spec.name!r} unavailable; continuing without it: {detail}"
        log.warning(message)
        if self.warning_sink is not None:
            self.warning_sink(message)

    def _path(self, path: str) -> tuple[Path, str]:
        target = _resolve(str(self.cwd), path)
        relative = target.relative_to(self.cwd).as_posix()
        return target, relative

    def _select(self, target: Path) -> LspServerSpec | None:
        suffix = target.suffix.lower()
        return next((spec for spec in self.servers if suffix in spec.extensions), None)

    def _root(self, target: Path, spec: LspServerSpec) -> Path:
        current = target.parent
        while True:
            if any((current / marker).exists() for marker in spec.root_markers):
                return current
            if current == self.cwd:
                return self.cwd
            if self.cwd not in current.parents:
                return self.cwd
            current = current.parent

    def _start(self, spec: LspServerSpec, root: Path) -> _ServerState | None:
        key = (spec.name, root)
        if key in self._unavailable:
            return None
        try:
            process = self.popen_factory(
                list(self.argv_builder(spec, root)), cwd=str(self.cwd),
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, start_new_session=True,
            )
            rpc = _RpcClient(process, request_timeout_s=max(1.0, self.diagnostics_timeout_s))
            rpc.request("initialize", {
                "processId": os.getpid(), "rootUri": root.as_uri(),
                "capabilities": {"textDocument": {"publishDiagnostics": {}}},
                "workspaceFolders": [{"uri": root.as_uri(), "name": root.name}],
                "initializationOptions": dict(spec.initialization),
            })
            rpc.notify("initialized", {})
        except Exception as exc:
            try:
                rpc.close()  # type: ignore[possibly-undefined]
            except Exception:
                try:
                    process.terminate()  # type: ignore[possibly-undefined]
                except Exception:
                    pass
            self._unavailable.add(key)
            self._warn_once(spec, str(exc))
            return None
        state = _ServerState(spec=spec, root=root, rpc=rpc)
        self._states[key] = state
        return state

    def _state(self, target: Path, spec: LspServerSpec) -> _ServerState | None:
        root = self._root(target, spec)
        return self._states.get((spec.name, root)) or self._start(spec, root)

    @staticmethod
    def _sync(state: _ServerState, target: Path, text: str) -> tuple[str, int]:
        uri = target.as_uri()
        version = state.versions.get(uri, 0) + 1
        state.versions[uri] = version
        if version == 1:
            state.rpc.notify("textDocument/didOpen", {"textDocument": {
                "uri": uri, "languageId": state.spec.name,
                "version": version, "text": text,
            }})
        else:
            state.rpc.notify("textDocument/didChange", {
                "textDocument": {"uri": uri, "version": version},
                "contentChanges": [{"text": text}],
            })
        return uri, version

    def after_edit(self, path: str) -> DiagnosticsReport | None:
        """Sync one edited file and collect its next diagnostics publication."""
        if not self.enabled:
            return None
        target, relative = self._path(path)
        spec = self._select(target)
        if spec is None:
            return None
        started = time.monotonic()
        with self._lock:
            state = self._state(target, spec)
            if state is None:
                report = DiagnosticsReport(
                    relative, spec.name, (), self.threshold,
                    int((time.monotonic() - started) * 1000), "unavailable",
                )
                self._emit(report)
                return report
            try:
                text = target.read_text(encoding="utf-8")
                uri = target.as_uri()
                state.rpc.drain()
                generation = state.rpc.diagnostic_generation(uri)
                self._sync(state, target, text)
                payload = state.rpc.wait_for_diagnostics(
                    uri, after=generation, timeout_s=self.diagnostics_timeout_s,
                )
                raw = payload.get("diagnostics", []) if payload else []
                diagnostics = tuple(
                    LspDiagnostic.from_payload(item)
                    for item in raw if isinstance(item, Mapping)
                )
                status = "ok" if payload is not None else "timeout"
            except Exception as exc:
                self._unavailable.add((spec.name, state.root))
                state.rpc.close()
                self._states.pop((spec.name, state.root), None)
                self._warn_once(spec, str(exc))
                diagnostics, status = (), "unavailable"
        report = DiagnosticsReport(
            relative, spec.name, diagnostics, self.threshold,
            int((time.monotonic() - started) * 1000), status,
        )
        self._emit(report)
        return report

    def query(
        self, kind: str, *, path: str, line: int = 0, character: int = 0,
    ) -> LspQueryResult:
        """Run ``definition``, ``references``, or document ``symbols``."""
        if not self.tool_enabled:
            raise LspSupportError("lsp tool is disabled")
        if kind not in {"definition", "references", "symbols"}:
            raise LspSupportError(f"unsupported lsp query {kind!r}")
        if line < 0 or character < 0:
            raise LspSupportError("line and character must be >= 0")
        target, relative = self._path(path)
        spec = self._select(target)
        if spec is None:
            return LspQueryResult(kind, relative, "", "unmatched")
        with self._lock:
            state = self._state(target, spec)
            if state is None:
                return LspQueryResult(kind, relative, "", "unavailable")
            uri = target.as_uri()
            if uri not in state.versions:
                self._sync(state, target, target.read_text(encoding="utf-8"))
            document = {"uri": uri}
            position = {"line": line, "character": character}
            if kind == "definition":
                method, params = "textDocument/definition", {
                    "textDocument": document, "position": position,
                }
            elif kind == "references":
                method, params = "textDocument/references", {
                    "textDocument": document, "position": position,
                    "context": {"includeDeclaration": True},
                }
            else:
                method, params = "textDocument/documentSymbol", {
                    "textDocument": document,
                }
            result = state.rpc.request(method, params)
        return LspQueryResult(
            kind, relative, json.dumps(result, sort_keys=True, separators=(",", ":")),
        )

    def active_roots(self) -> tuple[Path, ...]:
        return tuple(state.root for state in self._states.values())

    def close(self) -> None:
        with self._lock:
            for state in self._states.values():
                state.rpc.close()
            self._states.clear()

    def __enter__(self) -> "LspManager":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def _diagnostic_xml(report: DiagnosticsReport, max_chars: int) -> str:
    visible = report.visible
    if not visible:
        return ""
    opening = (
        f'<lsp_diagnostics file="{html.escape(report.file, quote=True)}" '
        f'server="{html.escape(report.server, quote=True)}" '
        f'errors="{report.errors}" warnings="{report.warnings}">'
    )
    closing = "</lsp_diagnostics>"
    if len(opening) + len(closing) + 1 > max_chars:
        return (
            f'<lsp_diagnostics errors="{report.errors}" warnings="{report.warnings}" '
            'truncated="true"></lsp_diagnostics>'
        )
    lines = [opening]
    for item in visible:
        attrs = (
            f' severity="{_SEVERITY_NAME.get(item.severity, "warning")}" '
            f'line="{item.line}" column="{item.character}"'
        )
        if item.source:
            attrs += f' source="{html.escape(item.source, quote=True)}"'
        if item.code:
            attrs += f' code="{html.escape(item.code, quote=True)}"'
        message = html.escape(item.message, quote=False)
        line = f"<diagnostic{attrs}>{message}</diagnostic>"
        if len("\n".join([*lines, line, closing])) > max_chars:
            marker = '<diagnostic truncated="true">additional diagnostics omitted</diagnostic>'
            if len("\n".join([*lines, marker, closing])) <= max_chars:
                lines.append(marker)
            break
        lines.append(line)
    lines.append(closing)
    return "\n".join(lines)


def _clip_middle(text: str, budget: int) -> str:
    if budget <= 0:
        return ""
    if len(text) <= budget:
        return text
    marker = "\n[... tool result clipped for LSP diagnostics ...]\n"
    if budget <= len(marker):
        return text[:budget]
    remaining = max(0, budget - len(marker))
    head = remaining // 2
    return text[:head] + marker + text[-(remaining - head):]


def append_diagnostics_to_tool_result(
    tool_result: str, report: DiagnosticsReport | None, *,
    max_output_chars: int, tool_name: str = "edit",
) -> str:
    """Append visible diagnostics inside a valid, bounded tool envelope."""
    if report is None or not report.visible:
        return tool_result
    if max_output_chars < 256:
        raise ValueError("max_output_chars must be >= 256 for LSP diagnostics")
    if not tool_result.startswith("<tool_result"):
        from .._shared.classification import derive_envelope_status
        from ._tools._common import _xml_attr

        status, error_kind = derive_envelope_status(tool_result)
        attrs = f' tool_name="{_xml_attr(tool_name)}" status="{status}"'
        if error_kind:
            attrs += f' error_kind="{_xml_attr(error_kind)}"'
        tool_result = f"<tool_result{attrs} v=\"1\">\n{tool_result}\n</tool_result>"
    closing = "</tool_result>"
    if not tool_result.rstrip().endswith(closing):
        raise LspSupportError("tool result envelope has no closing tag")
    close_at = tool_result.rfind(closing)
    base = tool_result[:close_at].rstrip()
    section = _diagnostic_xml(report, max_output_chars // 2)
    available = max_output_chars - len(section) - len(closing) - 2
    opening_end = base.find(">") + 1
    if opening_end <= 0 or opening_end > available:
        raise LspSupportError("tool result envelope opening exceeds output budget")
    base = base[:opening_end] + _clip_middle(
        base[opening_end:], available - opening_end
    )
    combined = f"{base}\n{section}\n{closing}"
    if len(combined) > max_output_chars:
        raise LspSupportError("could not fit diagnostics inside output budget")
    return combined


__all__ = [
    "DiagnosticsReport", "LspDiagnostic", "LspManager", "LspQueryResult",
    "LspServerSpec", "LspSupportError", "append_diagnostics_to_tool_result",
    "build_lsp_sandbox_argv", "parse_server_specs",
]
