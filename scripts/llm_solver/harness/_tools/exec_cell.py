"""Sandboxed Python eval cell and its on-demand function catalog."""
from __future__ import annotations

import codecs
from dataclasses import dataclass, replace
import json
import math
import os
from pathlib import Path
import selectors
import shlex
import signal
import subprocess
import time
from collections.abc import Callable, Mapping

from ...config import Config
from ..schemas import get_exec_cell_function_schemas
from ..sandbox import (
    AMBIENT_CONTAINER,
    _build_bwrap_argv,
    container_mode,
)
from ..sandbox.env_policy import build_subprocess_env
from ..sandbox.policy import sandbox_execution_kwargs
from ..tool_specs import EXEC_CELL_API_TOOL_NAMES


_PROTOCOL_PREFIX = "__YUJ_EXEC_CELL_REQUEST_V1__"
_MAX_SOURCE_CHARS = 65_536


# This program is fixed harness code.  The model-written source arrives over
# stdin after the sandbox boundary exists, so it is never interpolated into a
# host shell command.  stderr is the private request channel; model stderr and
# tracebacks are redirected to stdout and become ordinary cell output.
_CELL_RUNNER = r'''
import contextlib
import json
import os
import signal
import sys
import traceback

_PREFIX = "__YUJ_EXEC_CELL_REQUEST_V1__"


def _call(name, arguments):
    request = json.dumps(
        {"name": name, "arguments": arguments},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    sys.__stderr__.write(_PREFIX + request + "\n")
    sys.__stderr__.flush()
    response_line = sys.__stdin__.readline()
    if not response_line:
        raise RuntimeError("exec_cell dispatcher closed the response channel")
    response = json.loads(response_line)
    if not response.get("ok"):
        raise RuntimeError(str(response.get("error") or "inner call failed"))
    return str(response.get("result") or "")


def read(path, offset=0, limit=0):
    return _call("read", {"path": path, "offset": offset, "limit": limit})


def grep(pattern, path=".", glob="", page=1):
    return _call(
        "grep",
        {"pattern": pattern, "path": path, "glob": glob, "page": page},
    )


def glob(pattern, path=".", page=1):
    return _call("glob", {"pattern": pattern, "path": path, "page": page})


def list_definitions(
    path, symbol=None, kind=None, repo_wide=False, page=1,
):
    arguments = {"path": path, "repo_wide": repo_wide, "page": page}
    if symbol is not None:
        arguments["symbol"] = symbol
    if kind is not None:
        arguments["kind"] = kind
    return _call("list_definitions", arguments)


def bash(cmd):
    return _call("bash", {"cmd": cmd})


initial_line = sys.__stdin__.readline()
if not initial_line:
    raise SystemExit(2)
initial = json.loads(initial_line)
source = str(initial["source"])
timeout = int(initial["timeout"])


def _timeout(_signum, _frame):
    os._exit(124)


signal.signal(signal.SIGALRM, _timeout)
signal.alarm(max(1, timeout))
namespace = {
    "__name__": "__exec_cell__",
    "__builtins__": __builtins__,
    "read": read,
    "grep": grep,
    "glob": glob,
    "list_definitions": list_definitions,
    "bash": bash,
}
exit_status = 0
with contextlib.redirect_stderr(sys.stdout):
    try:
        exec(compile(source, "<exec_cell>", "exec"), namespace, namespace)
    except BaseException:
        traceback.print_exc()
        exit_status = 1
signal.alarm(0)
raise SystemExit(exit_status)
'''.lstrip()


InnerDispatch = Callable[[str, dict, Config], tuple[str, dict]]


@dataclass(frozen=True)
class ExecCellExecution:
    """A model-facing string plus raw trace metadata retained by dispatch."""

    output: str
    source: str
    combined_output_chars: int
    combined_output_bytes: int
    inner_calls: tuple[dict, ...]
    exit_status: int | None
    timed_out: bool

    def __str__(self) -> str:
        return self.output

    def trace_metadata(self) -> dict:
        return {
            "source": self.source,
            "combined_output_chars": self.combined_output_chars,
            "combined_output_bytes": self.combined_output_bytes,
            "inner_calls": [dict(item) for item in self.inner_calls],
        }


class _OutputCollector:
    """Count exact output size while retaining only a bounded head and tail."""

    def __init__(self, max_chars: int) -> None:
        self.max_chars = max(1_024, int(max_chars))
        self.head_limit = max(1, int(self.max_chars * 0.6))
        self.tail_limit = max(1, self.max_chars - self.head_limit)
        self.head = ""
        self.tail = ""
        self.total_bytes = 0
        self.total_chars = 0
        self._decoder = codecs.getincrementaldecoder("utf-8")("replace")

    def append(self, data: bytes) -> None:
        if not data:
            return
        self.total_bytes += len(data)
        self._append_text(self._decoder.decode(data, final=False))

    def finish(self) -> None:
        self._append_text(self._decoder.decode(b"", final=True))

    def _append_text(self, text: str) -> None:
        if not text:
            return
        self.total_chars += len(text)
        if len(self.head) < self.head_limit:
            take = min(self.head_limit - len(self.head), len(text))
            self.head += text[:take]
            text = text[take:]
        if text:
            self.tail = (self.tail + text)[-self.tail_limit:]

    def render(self) -> str:
        if self.total_chars <= self.max_chars:
            return self.head + self.tail
        omitted = max(0, self.total_chars - len(self.head) - len(self.tail))
        return (
            self.head
            + f"\n... [exec_cell output: {omitted} chars omitted] ...\n"
            + self.tail
        )


def list_functions_result() -> str:
    """Return a compact name-only discovery result."""
    return json.dumps(
        {"functions": list(EXEC_CELL_API_TOOL_NAMES)},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def get_function_details_result(names: object, *, mode: str) -> str:
    """Return exact injected-function schemas for selected names."""
    if (
        not isinstance(names, list)
        or not names
        or any(not isinstance(name, str) or not name for name in names)
    ):
        return "ERROR: names must be a non-empty array of function names"
    if len(set(names)) != len(names):
        return "ERROR: names must not contain duplicates"
    unknown = [name for name in names if name not in EXEC_CELL_API_TOOL_NAMES]
    if unknown:
        return "ERROR: unknown exec_cell function(s): " + ", ".join(unknown)
    schemas = {
        item["function"]["name"]: item["function"]
        for item in get_exec_cell_function_schemas(mode)
    }
    return json.dumps(
        {"functions": [schemas[name] for name in names]},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _cell_command() -> str:
    return "exec python3 -u -c " + shlex.quote(_CELL_RUNNER)


def _build_cell_process(
    *,
    cwd: str,
    cfg: Config,
    unreadable_paths: tuple[str, ...],
    readable_paths: tuple[str, ...],
    effective_env: Mapping[str, str],
    allow_login_shell: bool,
) -> tuple[list[str], str | None, dict[str, str] | None]:
    """Return argv, subprocess cwd, and host-side env for one cell."""
    execution = sandbox_execution_kwargs(cfg)
    if not execution["sandbox"]:
        return (
            ["python3", "-u", "-c", _CELL_RUNNER],
            cwd,
            build_subprocess_env(effective_env),
        )
    backend = str(execution["sandbox_backend"])
    legacy_mode = container_mode()
    if backend == "container":
        if legacy_mode is not None:
            raise RuntimeError(
                "sandbox.backend='container' cannot be combined with "
                "legacy YUJ_CONTAINER"
            )
        from ..sandbox.container_backend import ContainerBackend

        container = ContainerBackend(
            runtime=str(execution["container_runtime"]),
            image=str(execution["container_image"]),
            flags=tuple(execution["container_flags"]),
        )
        runtime_bin = (
            str(execution["container_runtime_bin"])
            or container.resolve_runtime(sandbox_required=True)
        )
        assert runtime_bin is not None
        return (
            container.build_argv(
                _cell_command(),
                cwd,
                runtime_bin=runtime_bin,
                effective_env=effective_env,
                unreadable_paths=unreadable_paths,
                readable_paths=readable_paths,
                sandbox_required=True,
                allow_login_shell=allow_login_shell,
            ),
            None,
            None,
        )
    if backend != "bwrap":
        raise RuntimeError(
            "resolved sandbox backend must be 'bwrap' or 'container'"
        )
    if legacy_mode == AMBIENT_CONTAINER:
        # The explicitly declared outer container is the filesystem boundary.
        # Mirror the normal bash path's best-effort empty network namespace.
        from ._run_in_sandbox import _probe_ambient_unshare_net

        prefix = ["unshare", "-n"] if _probe_ambient_unshare_net() else []
        return (
            [*prefix, "python3", "-u", "-c", _CELL_RUNNER],
            cwd,
            build_subprocess_env(effective_env),
        )
    if legacy_mode is None and not Path(cfg.bwrap_bin).is_file():
        raise RuntimeError(
            f"exec_cell requires bwrap at {cfg.bwrap_bin!r}; refusing "
            "unsandboxed execution"
        )
    argv = _build_bwrap_argv(
        _cell_command(),
        cwd,
        cfg.bwrap_bin,
        unreadable_paths=unreadable_paths,
        readable_paths=readable_paths,
        sandbox_required=True,
        effective_env=effective_env,
        allow_login_shell=allow_login_shell,
    )
    if legacy_mode is not None:
        # docker exec closes stdin unless -i is explicit.  The response pipe
        # is required for injected API calls.
        argv.insert(2, "-i")
    return argv, None, None


def _remaining_cfg(cfg: Config, deadline: float) -> Config:
    seconds = max(1, int(math.ceil(deadline - time.monotonic())))
    return replace(
        cfg,
        bash_timeout=min(int(cfg.bash_timeout), seconds),
        grep_timeout=min(int(cfg.grep_timeout), seconds),
    )


def _terminate_process(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (OSError, ProcessLookupError):
        try:
            proc.kill()
        except OSError:
            pass


def execute_cell(
    source: object,
    *,
    cwd: str,
    cfg: Config,
    inner_dispatch: InnerDispatch,
    unreadable_paths: tuple[str, ...],
    readable_paths: tuple[str, ...],
    effective_env: Mapping[str, str],
    allow_login_shell: bool,
) -> ExecCellExecution:
    """Run model-written Python in the sandbox and service injected calls."""
    if not bool(getattr(cfg, "tools_exec_cell_enabled", False)):
        return ExecCellExecution(
            "ERROR: exec_cell is disabled", str(source or ""), 0, 0, (),
            None, False,
        )
    if not isinstance(source, str) or not source:
        return ExecCellExecution(
            "ERROR: exec_cell source must be a non-empty string",
            str(source or ""), 0, 0, (), None, False,
        )
    if len(source) > _MAX_SOURCE_CHARS:
        return ExecCellExecution(
            f"ERROR: exec_cell source exceeds {_MAX_SOURCE_CHARS} characters",
            source, 0, 0, (), None, False,
        )
    timeout = int(getattr(cfg, "tools_exec_cell_timeout", 30))
    collector = _OutputCollector(int(getattr(cfg, "max_output_chars", 80_000)))
    inner_calls: list[dict] = []
    try:
        argv, process_cwd, process_env = _build_cell_process(
            cwd=cwd,
            cfg=cfg,
            unreadable_paths=unreadable_paths,
            readable_paths=readable_paths,
            effective_env=effective_env,
            allow_login_shell=allow_login_shell,
        )
        proc = subprocess.Popen(
            argv,
            cwd=process_cwd,
            env=process_env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
    except Exception as exc:
        return ExecCellExecution(
            f"ERROR: exec_cell sandbox failed: {exc}",
            source, 0, 0, (), None, False,
        )

    assert proc.stdin is not None
    assert proc.stdout is not None
    assert proc.stderr is not None
    deadline = time.monotonic() + timeout
    timed_out = False
    stderr_buffer = bytearray()
    selector = selectors.DefaultSelector()
    selector.register(proc.stdout, selectors.EVENT_READ, "stdout")
    selector.register(proc.stderr, selectors.EVENT_READ, "stderr")
    os.set_blocking(proc.stdout.fileno(), False)
    os.set_blocking(proc.stderr.fileno(), False)
    try:
        initial = json.dumps(
            {"source": source, "timeout": timeout},
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8") + b"\n"
        proc.stdin.write(initial)
        proc.stdin.flush()

        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                _terminate_process(proc)
                break
            events = selector.select(timeout=min(remaining, 0.1))
            if not events and proc.poll() is not None:
                # Pipes become readable-at-EOF on the next selector pass.
                continue
            for key, _mask in events:
                try:
                    chunk = os.read(key.fileobj.fileno(), 65_536)
                except BlockingIOError:
                    continue
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                if key.data == "stdout":
                    collector.append(chunk)
                    continue
                stderr_buffer.extend(chunk)
                while b"\n" in stderr_buffer:
                    raw_line, _, remainder = stderr_buffer.partition(b"\n")
                    stderr_buffer = bytearray(remainder)
                    line = raw_line.decode("utf-8", errors="replace")
                    if not line.startswith(_PROTOCOL_PREFIX):
                        collector.append(raw_line + b"\n")
                        continue
                    try:
                        request = json.loads(line[len(_PROTOCOL_PREFIX):])
                        name = request.get("name")
                        arguments = request.get("arguments")
                        if name not in EXEC_CELL_API_TOOL_NAMES:
                            raise ValueError(f"function {name!r} is not available")
                        if not isinstance(arguments, dict):
                            raise ValueError("function arguments must be an object")
                    except Exception as exc:
                        response = {"ok": False, "error": str(exc)}
                        arguments = None
                    if not isinstance(arguments, dict):
                        try:
                            proc.stdin.write(
                                json.dumps(
                                    response,
                                    ensure_ascii=False,
                                    separators=(",", ":"),
                                ).encode("utf-8") + b"\n"
                            )
                            proc.stdin.flush()
                        except (BrokenPipeError, OSError):
                            pass
                        continue

                    started = time.perf_counter()
                    dispatch_error = ""
                    try:
                        result, call_metadata = inner_dispatch(
                            str(name), arguments, _remaining_cfg(cfg, deadline)
                        )
                    except Exception as exc:
                        dispatch_error = str(exc)
                        result = f"ERROR: exec_cell inner dispatch failed: {exc}"
                        call_metadata = {
                            "executed": False,
                            "gate_blocked": False,
                        }
                    inner_calls.append({
                        "index": len(inner_calls) + 1,
                        "name": str(name),
                        "arguments": dict(arguments),
                        "result": str(result),
                        "duration_ms": round(
                            (time.perf_counter() - started) * 1000, 2
                        ),
                        "execution_metadata": dict(call_metadata),
                    })
                    response = (
                        {"ok": False, "error": dispatch_error}
                        if dispatch_error
                        else {"ok": True, "result": str(result)}
                    )
                    try:
                        proc.stdin.write(
                            json.dumps(
                                response,
                                ensure_ascii=False,
                                separators=(",", ":"),
                            ).encode("utf-8") + b"\n"
                        )
                        proc.stdin.flush()
                    except (BrokenPipeError, OSError):
                        pass
        try:
            exit_status = proc.wait(timeout=1)
        except subprocess.TimeoutExpired:
            timed_out = True
            _terminate_process(proc)
            exit_status = proc.wait(timeout=2)
    except (BrokenPipeError, OSError) as exc:
        _terminate_process(proc)
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass
        collector.append(f"exec_cell transport error: {exc}\n".encode())
        exit_status = None
    finally:
        selector.close()
        try:
            proc.stdin.close()
        except OSError:
            pass
        if stderr_buffer:
            collector.append(bytes(stderr_buffer))
        collector.finish()

    # The in-sandbox alarm exits with 124.  Treat it identically to the host
    # deadline so all backends have one timeout contract.
    timed_out = timed_out or exit_status == 124
    rendered = collector.render()
    if timed_out:
        prefix = f"ERROR: exec_cell timed out after {timeout} seconds"
        output = prefix + (("\n" + rendered) if rendered else "")
    elif exit_status != 0:
        prefix = f"ERROR: exec_cell failed with exit status {exit_status}"
        output = prefix + (("\n" + rendered) if rendered else "")
    else:
        output = rendered or "Cell completed with no output."
    return ExecCellExecution(
        output=output,
        source=source,
        combined_output_chars=collector.total_chars,
        combined_output_bytes=collector.total_bytes,
        inner_calls=tuple(inner_calls),
        exit_status=exit_status,
        timed_out=timed_out,
    )


__all__ = [
    "ExecCellExecution",
    "execute_cell",
    "get_function_details_result",
    "list_functions_result",
]
