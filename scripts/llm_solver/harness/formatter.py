"""Explicit, attributable formatter execution after file mutations."""
from __future__ import annotations

import hashlib
import os
import shlex
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from .._shared.formatter_spec import FormatterSpec, parse_formatter_spec
from ..config import Config
from .tool_policy import PermissionPolicy


MAX_ATTRIBUTION_PATHS = 200
MAX_ATTRIBUTION_BYTES = 64 * 1024 * 1024
_READ_CHUNK_BYTES = 128 * 1024


class FormatterAttributionError(RuntimeError):
    """A formatter cannot run without a bounded before/after comparison."""


@dataclass(frozen=True, slots=True)
class FileSignature:
    kind: str
    sha256: str
    size: int


@dataclass(frozen=True, slots=True)
class WorkspaceSnapshot:
    dirty_paths: frozenset[str]
    signatures: dict[str, FileSignature]


@dataclass(frozen=True, slots=True)
class FormatterResult:
    """Model-visible formatter report appended to a mutation result."""

    output: str = ""
    status: str = "not_selected"
    changed_paths: tuple[str, ...] = ()


def format_edited_file(
    path: str,
    *,
    cwd: str,
    cfg: Config | None,
) -> FormatterResult:
    """Run the first matching declared formatter and report exact effects."""
    if cfg is None or not bool(getattr(cfg, "formatter_enabled", False)):
        return FormatterResult()
    try:
        workspace, target, workspace_path = _resolve_target(cwd, path)
        selected = _select_formatter(target, workspace, cfg)
        if selected is None:
            return FormatterResult()
        spec, project_root = selected
        project_path = target.relative_to(project_root).as_posix()
        argv = tuple(
            part.replace("{path}", project_path) for part in spec.command
        )
        command = shlex.join(argv)
        permission = PermissionPolicy.from_rule_tables(
            getattr(cfg, "permissions_preset_rules", {}),
            getattr(cfg, "permissions_rules", {}),
        ).evaluate(
            tool_name="bash",
            arguments={"cmd": command},
            runtime_mode=getattr(cfg, "runtime_mode", "measurement"),
            ask_fallback="deny",
            approval_available=False,
        )
        if not permission.allowed:
            result = _permission_result(
                spec,
                workspace_path=workspace_path,
                project_root=_display_root(project_root, workspace),
                decision=permission.decision,
                rule=permission.rule,
            )
            _record_formatter_event(
                spec.name,
                path=workspace_path,
                project_root=_display_root(project_root, workspace),
                status=result.status,
                command_status="not_run",
                exit_code=None,
                timed_out=False,
                changed_paths=(),
                before_sha256="",
                after_sha256="",
                output="",
                output_truncated=False,
                permission_decision=permission.decision,
                permission_rule=permission.rule,
            )
            return result

        try:
            before = _workspace_snapshot(
                workspace,
                include_paths=(workspace_path,),
            )
        except FormatterAttributionError as exc:
            result = _attribution_failure(
                spec,
                workspace_path=workspace_path,
                project_root=_display_root(project_root, workspace),
                detail=str(exc),
                command_ran=False,
            )
            _record_formatter_event(
                spec.name,
                path=workspace_path,
                project_root=_display_root(project_root, workspace),
                status=result.status,
                command_status="not_run",
                exit_code=None,
                timed_out=False,
                changed_paths=(),
                before_sha256="",
                after_sha256="",
                output="",
                output_truncated=False,
                permission_decision=permission.decision,
                permission_rule=permission.rule,
            )
            return result

        process_output = _run_formatter_command(
            command,
            cwd=project_root,
            workspace=workspace,
            cfg=cfg,
        )
        exit_code = getattr(process_output, "exit_status", None)
        timed_out = bool(getattr(process_output, "timed_out", False))
        raw_output = str(process_output)
        filtered_output = _filter_output(raw_output, command=command, cfg=cfg)
        shown_output, output_truncated = _bound_output(
            filtered_output,
            int(getattr(cfg, "formatter_max_output_chars", 4000)),
        )
        command_status = (
            "timed_out"
            if timed_out
            else "passed"
            if exit_code == 0
            else "failed"
        )
        try:
            after = _workspace_snapshot(
                workspace,
                include_paths=tuple(before.signatures),
            )
        except FormatterAttributionError as exc:
            result = _attribution_failure(
                spec,
                workspace_path=workspace_path,
                project_root=_display_root(project_root, workspace),
                detail=str(exc),
                command_ran=True,
                command_status=command_status,
                process_output=shown_output,
                output_truncated=output_truncated,
                exit_code=exit_code,
                timed_out=timed_out,
            )
            _record_formatter_event(
                spec.name,
                path=workspace_path,
                project_root=_display_root(project_root, workspace),
                status=result.status,
                command_status=command_status,
                exit_code=exit_code,
                timed_out=timed_out,
                changed_paths=(),
                before_sha256=_signature_sha(before, workspace_path),
                after_sha256="",
                output=filtered_output,
                output_truncated=output_truncated,
                permission_decision=permission.decision,
                permission_rule=permission.rule,
            )
            return result

        changed_paths = _changed_paths(before, after)
        status = (
            "timed_out"
            if timed_out
            else "failed"
            if exit_code != 0
            else "changed"
            if changed_paths
            else "unchanged"
        )
        before_sha = _signature_sha(before, workspace_path)
        after_sha = _signature_sha(after, workspace_path)
        output = _render_result(
            spec,
            status=status,
            workspace_path=workspace_path,
            project_root=_display_root(project_root, workspace),
            command_status=command_status,
            exit_code=exit_code,
            timed_out=timed_out,
            changed_paths=changed_paths,
            before_sha256=before_sha,
            after_sha256=after_sha,
            process_output=shown_output,
            output_chars=len(filtered_output),
            output_sha256=hashlib.sha256(
                filtered_output.encode("utf-8", errors="replace")
            ).hexdigest(),
            output_truncated=output_truncated,
        )
        _record_formatter_event(
            spec.name,
            path=workspace_path,
            project_root=_display_root(project_root, workspace),
            status=status,
            command_status=command_status,
            exit_code=exit_code,
            timed_out=timed_out,
            changed_paths=changed_paths,
            before_sha256=before_sha,
            after_sha256=after_sha,
            output=filtered_output,
            output_truncated=output_truncated,
            permission_decision=permission.decision,
            permission_rule=permission.rule,
        )
        return FormatterResult(
            output=output,
            status=status,
            changed_paths=changed_paths,
        )
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        safe_path = _display_text(str(path))
        detail = type(exc).__name__
        output = (
            "\n\n<formatter_run status=\"internal_error\">\n"
            f"Formatter setup failed for {_xml_body(safe_path)} ({detail}). "
            "The model edit remains applied.\n"
            "</formatter_run>"
        )
        return FormatterResult(output=output, status="internal_error")


def _resolve_target(cwd: str, path: str) -> tuple[Path, Path, str]:
    workspace = Path(cwd).expanduser().resolve()
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = workspace / candidate
    target = candidate.resolve()
    try:
        relative = target.relative_to(workspace)
    except ValueError:
        raise ValueError("formatter target escapes the task workspace") from None
    current = workspace
    for part in relative.parts:
        current = current / part
        inspected = current.lstat()
        if stat.S_ISLNK(inspected.st_mode):
            raise ValueError("formatter target cannot cross a symbolic link")
    if not target.is_file():
        raise ValueError("formatter target is not a regular file")
    return workspace, target, relative.as_posix()


def _select_formatter(
    target: Path,
    workspace: Path,
    cfg: Config,
) -> tuple[FormatterSpec, Path] | None:
    extension = target.suffix.lower()
    for raw in tuple(getattr(cfg, "formatters", ()) or ()):
        spec = parse_formatter_spec(raw)
        if extension not in spec.extensions:
            continue
        root = _formatter_root(target, workspace, spec.root_markers)
        if root is not None:
            return spec, root
    return None


def _formatter_root(
    target: Path,
    workspace: Path,
    markers: tuple[str, ...],
) -> Path | None:
    if not markers:
        return workspace
    current = target.parent
    while True:
        if any(
            (current / marker).exists() or (current / marker).is_symlink()
            for marker in markers
        ):
            return current
        if current == workspace:
            return None
        parent = current.parent
        if parent == current or (
            parent != workspace and workspace not in parent.parents
        ):
            return None
        current = parent


def _run_formatter_command(
    command: str,
    *,
    cwd: Path,
    workspace: Path,
    cfg: Config,
):
    from .sandbox.env_policy import active_environment
    from .sandbox.policy import sandbox_execution_kwargs
    from .tools import (
        _bash_readable_paths,
        _bash_unreadable_paths,
        _effective_command_environment,
        bash,
    )

    effective_env, allow_login_shell = active_environment()
    if effective_env is None:
        effective_env, allow_login_shell = _effective_command_environment(cfg)
    return bash(
        command,
        cwd=str(cwd),
        timeout=int(getattr(cfg, "formatter_timeout", 10)),
        bwrap_bin=cfg.bwrap_bin,
        unreadable_paths=_bash_unreadable_paths(str(workspace), cfg),
        readable_paths=_bash_readable_paths(cfg),
        effective_env=effective_env,
        allow_login_shell=allow_login_shell,
        **sandbox_execution_kwargs(cfg),
    )


def _filter_output(output: str, *, command: str, cfg: Config) -> str:
    from .tools import _filter_bash_output

    return _filter_bash_output(output, command, cfg)


def _git(root: Path, arguments: list[str]) -> bytes:
    env = os.environ.copy()
    env.update({
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_TERMINAL_PROMPT": "0",
        "LC_ALL": "C",
    })
    try:
        result = subprocess.run(
            ["git", "-c", "core.fsmonitor=false", *arguments],
            cwd=root,
            env=env,
            capture_output=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise FormatterAttributionError(
            f"Git attribution failed ({type(exc).__name__})"
        ) from exc
    if result.returncode != 0:
        raise FormatterAttributionError(
            f"Git attribution command exited {result.returncode}"
        )
    return result.stdout


def _dirty_paths(root: Path) -> frozenset[str]:
    top = Path(os.fsdecode(
        _git(root, ["rev-parse", "--show-toplevel"])
    ).strip())
    if top.resolve() != root.resolve():
        raise FormatterAttributionError(
            "formatter attribution requires the task directory to be the Git root"
        )
    unstaged = _git(root, [
        "diff",
        "--name-only",
        "-z",
        "--no-ext-diff",
        "--no-renames",
        "--",
    ])
    staged = _git(root, [
        "diff",
        "--cached",
        "--name-only",
        "-z",
        "--no-ext-diff",
        "--no-renames",
        "--",
    ])
    untracked = _git(root, [
        "ls-files",
        "--others",
        "--exclude-standard",
        "-z",
        "--",
    ])
    paths: set[str] = set()
    for raw in (
        *unstaged.split(b"\0"),
        *staged.split(b"\0"),
        *untracked.split(b"\0"),
    ):
        if not raw:
            continue
        value = os.fsdecode(raw)
        parsed = PurePosixPath(value)
        if not value or parsed.is_absolute() or ".." in parsed.parts:
            raise FormatterAttributionError(
                "Git returned an unsafe formatter attribution path"
            )
        paths.add(value)
    return frozenset(paths)


def _workspace_snapshot(
    root: Path,
    *,
    include_paths: tuple[str, ...],
) -> WorkspaceSnapshot:
    dirty = _dirty_paths(root)
    candidates = set(dirty)
    candidates.update(include_paths)
    if len(candidates) > MAX_ATTRIBUTION_PATHS:
        raise FormatterAttributionError(
            "formatter attribution exceeds the limit of "
            f"{MAX_ATTRIBUTION_PATHS} repository-visible paths"
        )
    total = [0]
    signatures = {
        path: _file_signature(root / path, total=total)
        for path in sorted(candidates)
    }
    return WorkspaceSnapshot(dirty_paths=dirty, signatures=signatures)


def _file_signature(path: Path, *, total: list[int]) -> FileSignature:
    try:
        inspected = path.lstat()
    except FileNotFoundError:
        return FileSignature("absent", "", 0)
    if stat.S_ISLNK(inspected.st_mode):
        data = os.fsencode(os.readlink(path))
        return FileSignature("symlink", hashlib.sha256(data).hexdigest(), len(data))
    if not stat.S_ISREG(inspected.st_mode):
        payload = f"{inspected.st_mode}:{inspected.st_size}".encode("ascii")
        return FileSignature(
            "other",
            hashlib.sha256(payload).hexdigest(),
            inspected.st_size,
        )
    total[0] += inspected.st_size
    if total[0] > MAX_ATTRIBUTION_BYTES:
        raise FormatterAttributionError(
            "formatter attribution exceeds the byte limit of "
            f"{MAX_ATTRIBUTION_BYTES}"
        )
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(_READ_CHUNK_BYTES):
                digest.update(chunk)
    except OSError as exc:
        raise FormatterAttributionError(
            f"cannot hash formatter attribution path ({type(exc).__name__})"
        ) from exc
    return FileSignature("file", digest.hexdigest(), inspected.st_size)


def _changed_paths(
    before: WorkspaceSnapshot,
    after: WorkspaceSnapshot,
) -> tuple[str, ...]:
    candidates = set(before.signatures) | set(after.signatures)
    candidates |= set(after.dirty_paths) - set(before.dirty_paths)
    return tuple(sorted(
        _display_text(path)
        for path in candidates
        if before.signatures.get(path) != after.signatures.get(path)
    ))


def _signature_sha(snapshot: WorkspaceSnapshot, path: str) -> str:
    signature = snapshot.signatures.get(path)
    if signature is None or signature.kind == "absent":
        return "absent"
    return signature.sha256


def _bound_output(value: str, limit: int) -> tuple[str, bool]:
    if len(value) <= limit:
        return value, False
    marker = f"\n[formatter output bounded: {len(value) - limit} chars omitted]\n"
    remaining = max(0, limit - len(marker))
    head = int(remaining * 0.6)
    tail = remaining - head
    return value[:head] + marker + (value[-tail:] if tail else ""), True


def _render_result(
    spec: FormatterSpec,
    *,
    status: str,
    workspace_path: str,
    project_root: str,
    command_status: str,
    exit_code: int | None,
    timed_out: bool,
    changed_paths: tuple[str, ...],
    before_sha256: str,
    after_sha256: str,
    process_output: str,
    output_chars: int,
    output_sha256: str,
    output_truncated: bool,
) -> str:
    exit_label = "unknown" if exit_code is None else str(exit_code)
    changed = "\n".join(f"- {_xml_body(path)}" for path in changed_paths)
    if not changed:
        changed = "none"
    if status in {"failed", "timed_out"}:
        conclusion = (
            "Formatter failed. The model edit remains applied. Formatter "
            "effects listed below may also remain."
        )
    else:
        conclusion = "Formatter completed."
    return (
        "\n\n"
        f'<formatter_run name="{_xml_attr(spec.name)}" '
        f'status="{status}" command_status="{command_status}" '
        f'exit_code="{exit_label}" timed_out="{str(timed_out).lower()}">\n'
        f"{conclusion}\n"
        f"Selected path: {_xml_body(workspace_path)}\n"
        f"Project root: {_xml_body(project_root)}\n"
        f"Pre-formatter SHA-256: {before_sha256}\n"
        f"Post-formatter SHA-256: {after_sha256}\n"
        "Formatter changed paths:\n"
        f"{changed}\n"
        f"Formatter output: chars={output_chars} sha256={output_sha256} "
        f"truncated={str(output_truncated).lower()}\n"
        f"{_xml_body(process_output)}\n"
        "</formatter_run>"
    )


def _permission_result(
    spec: FormatterSpec,
    *,
    workspace_path: str,
    project_root: str,
    decision: str,
    rule: str,
) -> FormatterResult:
    output = (
        "\n\n"
        f'<formatter_run name="{_xml_attr(spec.name)}" status="denied" '
        f'permission_decision="{_xml_attr(decision)}">\n'
        "Formatter did not run because the resolved bash permission did not "
        "allow it. The model edit remains applied.\n"
        f"Selected path: {_xml_body(workspace_path)}\n"
        f"Project root: {_xml_body(project_root)}\n"
        f"Permission rule: {_xml_body(rule)}\n"
        "Formatter changed paths: none\n"
        "</formatter_run>"
    )
    return FormatterResult(output=output, status="denied")


def _attribution_failure(
    spec: FormatterSpec,
    *,
    workspace_path: str,
    project_root: str,
    detail: str,
    command_ran: bool,
    command_status: str = "not_run",
    process_output: str = "",
    output_truncated: bool = False,
    exit_code: int | None = None,
    timed_out: bool = False,
) -> FormatterResult:
    status = "attribution_failed" if command_ran else "not_run"
    effect = (
        "The formatter command ran, so partial effects may remain and changed "
        "paths are unknown."
        if command_ran
        else "The formatter command did not run. The model edit remains applied."
    )
    exit_label = "unknown" if exit_code is None else str(exit_code)
    output = (
        "\n\n"
        f'<formatter_run name="{_xml_attr(spec.name)}" status="{status}" '
        f'command_status="{command_status}" exit_code="{exit_label}" '
        f'timed_out="{str(timed_out).lower()}">\n'
        f"Formatter attribution failed: {_xml_body(detail)}. {effect}\n"
        f"Selected path: {_xml_body(workspace_path)}\n"
        f"Project root: {_xml_body(project_root)}\n"
        "Formatter changed paths: unknown\n"
        f"Formatter output truncated: {str(output_truncated).lower()}\n"
        f"{_xml_body(process_output)}\n"
        "</formatter_run>"
    )
    return FormatterResult(output=output, status=status)


def _record_formatter_event(
    name: str,
    *,
    path: str,
    project_root: str,
    status: str,
    command_status: str,
    exit_code: int | None,
    timed_out: bool,
    changed_paths: tuple[str, ...],
    before_sha256: str,
    after_sha256: str,
    output: str,
    output_truncated: bool,
    permission_decision: str,
    permission_rule: str,
) -> None:
    from .savings import get_ledger

    get_ledger().record(
        bucket="formatter_run",
        layer="harness",
        mechanism=name,
        input_chars=0,
        output_chars=len(output),
        measure_type="exact",
        ctx={
            "path": path,
            "project_root": project_root,
            "status": status,
            "command_status": command_status,
            "exit_code": exit_code,
            "timed_out": timed_out,
            "changed_paths": list(changed_paths),
            "before_sha256": before_sha256,
            "after_sha256": after_sha256,
            "output_chars": len(output),
            "output_sha256": hashlib.sha256(
                output.encode("utf-8", errors="replace")
            ).hexdigest(),
            "output_truncated": output_truncated,
            "permission_decision": permission_decision,
            "permission_rule": permission_rule,
        },
    )


def _display_root(project_root: Path, workspace: Path) -> str:
    relative = project_root.relative_to(workspace)
    return relative.as_posix() or "."


def _display_text(value: str) -> str:
    return value.encode("utf-8", errors="backslashreplace").decode("utf-8")


def _xml_attr(value: str) -> str:
    return (
        _display_text(value)
        .replace("&", "&amp;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _xml_body(value: str) -> str:
    return (
        _display_text(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


__all__ = [
    "FormatterAttributionError",
    "FormatterResult",
    "MAX_ATTRIBUTION_BYTES",
    "MAX_ATTRIBUTION_PATHS",
    "format_edited_file",
]
