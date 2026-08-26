"""Installed CLI for Yuj coding sessions."""
from __future__ import annotations

import argparse
import contextlib
import getpass
import hashlib
import json
import os
import subprocess
import sys
import uuid
from importlib.metadata import PackageNotFoundError, version as distribution_version
from pathlib import Path

from ..llm_solver.config import (
    PROJECT_ROOT,
    ConfigLayerSpec,
    get_server_base_url,
    load_config,
    resolve_config,
)
from ..llm_solver.config_inspection import (
    build_error_document,
    build_inspection_document,
    render_inspection_human,
    render_inspection_json,
    sanitize_diagnostic_message,
    validate_configuration_references,
)
from ..llm_solver._permission_presets import (
    ASSISTANT_PERMISSION_PRESET_NAMES,
)
from ..llm_solver._shared.edit_formats import EDIT_FORMATS
from ..llm_solver._shared.paths import local_config_path
from ..llm_solver.models import resolve_model
from ..llm_solver.runtime_resources import validate_runtime_resources
from ..llm_solver.server.request_controls import THINKING_LEVELS
from ..llm_solver.harness.worktree_runtime import (
    WorktreeRuntimeError,
    inspect_session_worktree,
    remove_session_worktree,
)
from ..llm_solver.harness.clarifications import (
    ClarificationStateError,
    clarification_answer_path,
    clarification_state,
    record_clarification_answer,
)
from ..llm_solver.harness.corrections import (
    CorrectionState,
    CorrectionStateError,
    correction_path,
    create_correction,
    validate_correction_trace,
)
from ..llm_solver.harness._loop.interrupted_turn import append_trace_event_fsync
from ..llm_solver.harness._loop.trace_schema import TRACE_SCHEMA_VERSION
from ..llm_solver.harness.sandbox.policy import (
    SANDBOX_CHOICES,
    SandboxResolutionError,
    inspect_sandbox_selection,
    preflight_sandbox,
    probe_sandbox_capabilities,
    resolve_sandbox_selection,
)
from ._auth import (
    AuthBinding,
    CredentialMissingError,
    CredentialStore,
    ProviderAuthError,
    browser_sign_in,
    provider_spec,
)
from ._images import (
    ImageInputError,
    PendingImage,
    image_evidence,
    read_image_inputs,
    save_image_segment,
)
from .forking import ForkSessionError, fork_saved_session, validate_correction_owner
from .purge import (
    PurgePreview,
    PurgeSessionError,
    preview_session_purge,
    purge_archived_session,
)
from .progress import TraceFollower
from .runner import (
    _make_client,
    _record_auth_binding,
    approval_request_path,
    create_session,
    derive_live_state,
    load_approval_decisions,
    load_approval_request,
    load_interrupt_marker,
    mark_session_interrupted,
    prepare_smoke_repo,
    rewind_session,
    resolve_served_model,
    resolve_smoke_model,
    run_session,
    save_approval_decisions,
    save_approval_request,
    session_compact_summary,
    session_sandbox_provenance,
    session_trace_tail,
    session_turn_count,
    session_turn_tail,
    validate_image_capability,
)
from .store import (
    AmbiguousSessionRefError,
    SessionArchiveError,
    SessionLabelError,
    SessionLockedError,
    SessionPurgeInProgressError,
    SessionStore,
    is_full_session_id,
)
from .startup import preflight_assistant_startup, render_startup_preflight
from .usage import (
    UsageEvidenceError,
    aggregate_session_usage,
    render_session_usage,
)

CLI_NAME = "yuj"
_LATEST_SESSION_TOKENS = {"latest", "last"}
_PROVIDER_PRESETS = {
    "local": {"provider": "openai-compatible"},
    "openai": {
        "provider": "openai-compatible",
        "base_url": "https://api.openai.com/v1",
        "api_key": "$ENV:OPENAI_API_KEY",
    },
    "openrouter": {
        "provider": "openai-compatible",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": "$ENV:OPENROUTER_API_KEY",
    },
    "zai": {
        "provider": "openai-compatible",
        "base_url": "https://api.z.ai/api/paas/v4",
        "api_key": "$ENV:ZAI_API_KEY",
    },
    "anthropic": {
        "provider": "anthropic",
        "base_url": "https://api.anthropic.com/v1",
        "api_key": "$ENV:ANTHROPIC_API_KEY",
    },
    "claude": {
        "provider": "anthropic",
        "base_url": "https://api.anthropic.com/v1",
        "api_key": "yuj-host-credential",
    },
    "codex": {
        "provider": "openai-compatible",
        "base_url": "https://api.openai.com/v1",
        "api_key": "yuj-host-credential",
    },
    "custom": {"provider": "openai-compatible"},
}
_MANAGED_PROVIDERS = frozenset({"claude", "codex"})
_TREATMENT_CONFIG = PROJECT_ROOT / "configs/regimes/treatment.toml"
_PLAIN_CONFIG = PROJECT_ROOT / "configs/regimes/baselines/plain_long_solve.toml"


def _cli_version() -> str:
    try:
        return distribution_version("yuj")
    except PackageNotFoundError:
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog=CLI_NAME,
        description="Run a secondary Yuj command",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def _attach_run_args(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "task",
            nargs="*",
            help=(
                "task text; use this form instead of --prompt-text or "
                "--prompt-file"
            ),
        )
        p.add_argument(
            "-C", "--cd", "--cwd",
            dest="cwd",
            metavar="DIR",
            type=Path,
            default=Path.cwd(),
            help="working directory to edit (default: current directory)",
        )
        p.add_argument("--prompt-text", help="literal task prompt")
        p.add_argument(
            "--prompt-file",
            type=Path,
            help="read task prompt from a file, or use '-' for standard input",
        )
        p.add_argument(
            "-i", "--image",
            type=Path,
            action="append",
            default=[],
            help="local image to attach; repeat for more images",
        )
        p.add_argument("-m", "--model", help="model name or short alias")
        p.add_argument(
            "--thinking", choices=THINKING_LEVELS,
            help="per-request reasoning effort",
        )
        p.add_argument(
            "--plan-mode", choices=("off", "required"),
            help="require an explicit .solver/plan.md before implementation",
        )
        p.add_argument(
            "--permission-preset",
            choices=ASSISTANT_PERMISSION_PRESET_NAMES,
            help="fixed assistant permission preset for this session",
        )
        p.add_argument(
            "--edit-format", choices=EDIT_FORMATS,
            help="override the selected model profile's edit dialect",
        )
        p.add_argument(
            "--provider",
            choices=sorted(_PROVIDER_PRESETS),
            help=(
                "model service: local, claude, codex, openai, anthropic, "
                "zai, openrouter, or custom"
            ),
        )
        p.add_argument(
            "--base-url",
            help=(
                "model API base URL; Claude and Codex managed endpoints "
                "cannot be changed"
            ),
        )
        p.add_argument(
            "--api-key-env",
            help="environment variable containing the API key; stored as an env reference, not the key",
        )
        p.add_argument("-c", "--config", type=Path, action="append", default=[],
                       help="extra TOML settings file; repeat to apply more files")
        p.add_argument("--system-prompt", type=Path, default=None,
                       help="file to prepend to the system prompt")
        p.add_argument(
            "--treatment",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="use the treatment package (default: enabled)",
        )
        p.add_argument(
            "--context",
            default=None,
            help="context mode (default: halflife with treatment, full without)",
        )
        p.add_argument(
            "--dry-run",
            action="store_true",
            help="validate local startup through the model-network boundary, then exit",
        )
        p.add_argument(
            "-V", "--version",
            action="version",
            version=f"%(prog)s {_cli_version()}",
        )
        p.set_defaults(func=cmd_run)

    config_parser = sub.add_parser(
        "config",
        help="validate and explain the resolved configuration without model work",
    )
    config_parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="write stable machine-readable JSON",
    )
    config_parser.add_argument("--model", "-m", help="model name or short alias")
    config_parser.add_argument(
        "--thinking",
        choices=THINKING_LEVELS,
        help="per-request reasoning effort",
    )
    config_parser.add_argument(
        "--plan-mode",
        choices=("off", "required"),
        help="planning-phase setting to validate",
    )
    config_parser.add_argument(
        "--permission-preset",
        choices=ASSISTANT_PERMISSION_PRESET_NAMES,
        help="assistant permission preset to validate",
    )
    config_parser.add_argument(
        "--edit-format",
        choices=EDIT_FORMATS,
        help="model edit dialect to validate",
    )
    config_parser.add_argument(
        "--provider",
        choices=sorted(_PROVIDER_PRESETS),
        help="model service setting to validate",
    )
    config_parser.add_argument("--base-url", help="model API base URL override")
    config_parser.add_argument(
        "--api-key-env",
        help="environment variable containing the API key",
    )
    config_parser.add_argument(
        "--config",
        "-c",
        type=Path,
        action="append",
        default=[],
        help="extra TOML settings file; repeat to apply more files",
    )
    config_parser.add_argument(
        "--treatment",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="inspect the treatment package (default: enabled)",
    )
    config_parser.add_argument(
        "--context",
        default=None,
        help="context mode to validate (base default when omitted)",
    )
    config_parser.add_argument(
        "--agent",
        action="append",
        default=[],
        help="named agent descriptor to validate; repeat for more agents",
    )
    config_parser.set_defaults(func=cmd_config)

    setup_parser = sub.add_parser("setup", help="save model settings for this machine")
    setup_parser.add_argument(
        "--provider",
        choices=sorted(_PROVIDER_PRESETS),
        help="model service to save (interactive if omitted)",
    )
    setup_parser.add_argument("--model", "-m", help="default model ID to save")
    setup_parser.add_argument(
        "--permission-preset",
        choices=ASSISTANT_PERMISSION_PRESET_NAMES,
        help="fixed assistant permission preset to save",
    )
    setup_parser.add_argument(
        "--sandbox",
        choices=sorted(SANDBOX_CHOICES),
        help="sandbox choice to save (default: bwrap)",
    )
    setup_parser.add_argument(
        "--sandbox-image",
        help="local image reference for Docker, Podman, or auto",
    )
    setup_parser.add_argument("--base-url", help="API base URL to save")
    setup_parser.add_argument(
        "--auth",
        choices=("api-key", "subscription"),
        help="Claude or Codex authentication method",
    )
    setup_parser.add_argument("--api-key", help="API key to save")
    setup_parser.add_argument(
        "--api-key-env",
        help="save an environment-variable reference instead of the key",
    )
    setup_parser.add_argument(
        "--force",
        action="store_true",
        help="replace an existing config.local.toml",
    )
    setup_parser.set_defaults(func=cmd_setup)

    login_parser = sub.add_parser(
        "login", help="save and select a Claude or Codex credential"
    )
    login_parser.add_argument(
        "--provider", choices=sorted(_MANAGED_PROVIDERS), required=True
    )
    login_parser.add_argument(
        "--auth",
        choices=("api-key", "subscription"),
        default="subscription",
    )
    login_parser.add_argument("--api-key")
    login_parser.add_argument("--api-key-env")
    login_parser.set_defaults(func=cmd_login)

    logout_parser = sub.add_parser(
        "logout", help="remove one Claude or Codex credential"
    )
    logout_parser.add_argument(
        "--provider", choices=sorted(_MANAGED_PROVIDERS)
    )
    logout_parser.set_defaults(func=cmd_logout)

    auth_status_parser = sub.add_parser(
        "auth-status", help="show selected provider authentication without secrets"
    )
    auth_status_parser.set_defaults(func=cmd_auth_status)

    models_parser = sub.add_parser("models", help="list models from the selected service")
    models_parser.add_argument("--config", "-c", type=Path, action="append", default=[],
                               help="extra TOML settings file; repeat to apply more files")
    models_parser.set_defaults(func=cmd_models)

    doctor_parser = sub.add_parser(
        "doctor", help="check settings, sandbox, model service, and Git"
    )
    doctor_parser.add_argument("--config", "-c", type=Path, action="append", default=[],
                               help="extra TOML settings file; repeat to apply more files")
    doctor_parser.set_defaults(func=cmd_doctor)

    smoke_parser = sub.add_parser("smoke", help="run one small coding check")
    smoke_parser.add_argument("--root", type=Path, default=None,
                              help="throwaway directory (default: new temporary directory)")
    smoke_parser.add_argument("--assist-home", type=Path, default=None,
                              help="session root (default: normal HARNESS_ASSIST_HOME)")
    smoke_parser.add_argument("--model", "-m", help="preferred model alias or exact model id")
    smoke_parser.add_argument(
        "--thinking", choices=THINKING_LEVELS,
        help="per-request reasoning effort",
    )
    smoke_parser.add_argument(
        "--plan-mode", choices=("off", "required"),
        help="require an explicit .solver/plan.md before implementation",
    )
    smoke_parser.add_argument(
        "--permission-preset",
        choices=ASSISTANT_PERMISSION_PRESET_NAMES,
        help="fixed assistant permission preset for this session",
    )
    smoke_parser.add_argument(
        "--edit-format", choices=EDIT_FORMATS,
        help="override the selected model profile's edit dialect",
    )
    smoke_parser.add_argument(
        "--provider",
        choices=sorted(_PROVIDER_PRESETS),
        help=(
            "model service: local, claude, codex, openai, anthropic, "
            "zai, openrouter, or custom"
        ),
    )
    smoke_parser.add_argument(
        "--base-url",
        help=(
            "model API base URL; Claude and Codex managed endpoints "
            "cannot be changed"
        ),
    )
    smoke_parser.add_argument(
        "--api-key-env",
        help="environment variable containing the API key; stored as an env reference, not the key",
    )
    smoke_parser.add_argument("--config", "-c", type=Path, action="append", default=[],
                              help="extra TOML settings file; repeat to apply more files")
    smoke_parser.add_argument("--system-prompt", type=Path, default=None,
                              help="file to prepend to the system prompt")
    smoke_parser.add_argument(
        "--treatment",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use the treatment package (default: enabled)",
    )
    smoke_parser.add_argument(
        "--context",
        default=None,
        help="context mode (default: halflife with treatment, full without)",
    )
    smoke_parser.set_defaults(func=cmd_smoke)

    resume_parser = sub.add_parser("resume", help="continue a saved coding session")
    resume_parser.add_argument(
        "session_id",
        nargs="?",
        default="latest",
        help=(
            "coding-session ID, exact label, unique ID prefix, or 'latest' "
            "(default: latest incomplete session)"
        ),
    )
    resume_parser.add_argument("--prompt-text", help="literal follow-up text")
    resume_parser.add_argument(
        "--prompt-file",
        type=Path,
        help="read follow-up text from a file, or use '-' for standard input",
    )
    resume_parser.add_argument(
        "--image",
        type=Path,
        action="append",
        default=[],
        help="local image to attach to the follow-up; repeat for more images",
    )
    resume_parser.set_defaults(func=cmd_resume)

    correct_parser = sub.add_parser(
        "correct", help="record one correction for a stopped session"
    )
    correct_parser.add_argument(
        "session_id",
        help="coding-session ID, exact label, or unique ID prefix",
    )
    correct_parser.add_argument(
        "correction", help="exact operator correction to send on resume"
    )
    correct_parser.set_defaults(func=cmd_correct)

    answer_parser = sub.add_parser(
        "answer", help="record one answer for a pending clarification"
    )
    answer_parser.add_argument(
        "session_id",
        help="coding-session ID, exact label, or unique ID prefix",
    )
    answer_parser.add_argument(
        "request_id", help="exact clarification request ID shown by status"
    )
    answer_parser.add_argument(
        "answer", help="exact operator answer to record"
    )
    answer_parser.set_defaults(func=cmd_answer)

    rewind_parser = sub.add_parser(
        "rewind",
        help="restore a saved session to an earlier conversation and tree turn",
    )
    rewind_parser.add_argument(
        "session_id",
        help="coding-session ID, exact label, or unique ID prefix",
    )
    rewind_parser.add_argument("turn", type=int, help="completed turn to restore")
    rewind_parser.add_argument(
        "--reason",
        default="operator_cli",
        help="reason recorded in the append-only trace",
    )
    rewind_parser.set_defaults(func=cmd_rewind)

    approve_parser = sub.add_parser("approve", help="allow a pending shell action")
    approve_parser.add_argument(
        "session_id",
        nargs="?",
        default="latest",
        help=(
            "coding-session ID, exact label, unique ID prefix, or 'latest' "
            "(default: latest waiting request)"
        ),
    )
    approve_parser.add_argument("--always", action="store_true",
                                help="approve the exact same tool and command in this session")
    approve_parser.set_defaults(func=cmd_approve)

    reject_parser = sub.add_parser("reject", help="refuse a pending shell action")
    reject_parser.add_argument(
        "session_id",
        nargs="?",
        default="latest",
        help=(
            "coding-session ID, exact label, unique ID prefix, or 'latest' "
            "(default: latest waiting request)"
        ),
    )
    reject_parser.add_argument("--reason", default="operator rejected the action",
                               help="reason shown to the model after resume")
    reject_parser.add_argument("--always", action="store_true",
                               help="reject the exact same tool and command in this session")
    reject_parser.set_defaults(func=cmd_reject)

    sessions_parser = sub.add_parser("sessions", help="list saved coding sessions")
    sessions_parser.add_argument(
        "--limit", type=int, default=20,
        help="number of recent sessions to list (default: 20)",
    )
    sessions_parser.add_argument(
        "--archived",
        action="store_true",
        help="list archived sessions instead of ordinary sessions",
    )
    sessions_parser.set_defaults(func=cmd_sessions)

    fork_parser = sub.add_parser(
        "fork", help="create an isolated child of one stopped saved session"
    )
    fork_parser.add_argument(
        "session_id",
        help="coding-session ID, exact label, or unique ID prefix",
    )
    fork_parser.set_defaults(func=cmd_fork)

    archive_parser = sub.add_parser(
        "archive", help="hide one stopped saved session from ordinary selection"
    )
    archive_parser.add_argument(
        "session_id",
        help="coding-session ID, exact label, or unique ID prefix",
    )
    archive_parser.set_defaults(func=cmd_archive)

    unarchive_parser = sub.add_parser(
        "unarchive", help="restore one archived session to ordinary selection"
    )
    unarchive_parser.add_argument(
        "session_id",
        help="coding-session ID, exact label, or unique ID prefix",
    )
    unarchive_parser.set_defaults(func=cmd_unarchive)

    purge_parser = sub.add_parser(
        "purge",
        help="preview or permanently remove one archived coding session",
    )
    purge_parser.add_argument(
        "session_id",
        help="one full immutable coding-session ID",
    )
    purge_parser.add_argument(
        "--preview",
        action="store_true",
        help="list the exact owned entries and estimated bytes without mutation",
    )
    purge_parser.add_argument(
        "--confirm",
        metavar="FULL_SESSION_ID",
        help="permanently purge by repeating the exact full session ID",
    )
    purge_parser.set_defaults(func=cmd_purge)

    label_parser = sub.add_parser(
        "label", help="set, replace, or clear a saved session label"
    )
    label_parser.add_argument(
        "session_id",
        help="coding-session ID, exact label, or unique ID prefix",
    )
    label_parser.add_argument(
        "label",
        nargs="?",
        help="exact manual label to set",
    )
    label_parser.add_argument(
        "--clear",
        action="store_true",
        help="clear the current label",
    )
    label_parser.set_defaults(func=cmd_label)

    status_parser = sub.add_parser("status", help="show the status of one coding session")
    status_parser.add_argument(
        "session_id",
        nargs="?",
        default="latest",
        help=(
            "coding-session ID, exact label, unique ID prefix, or 'latest' "
            "(default: latest session)"
        ),
    )
    status_parser.set_defaults(func=cmd_status)
    current_parser = sub.add_parser(
        "current",
        help="show the active or newest coding session",
    )
    current_parser.set_defaults(func=cmd_current)

    show_parser = sub.add_parser("show", help="show one coding session and recent activity")
    show_parser.add_argument(
        "session_id",
        nargs="?",
        default="latest",
        help=(
            "coding-session ID, exact label, unique ID prefix, or 'latest' "
            "(default: latest session)"
        ),
    )
    show_parser.add_argument("--turns", type=int, default=5,
                             help="number of recent turns to show")
    show_parser.add_argument("--trace-lines", type=int, default=10,
                             help="number of recent trace events to show")
    show_parser.set_defaults(func=cmd_show)

    usage_parser = sub.add_parser(
        "usage", help="show exact persisted usage for one coding session"
    )
    usage_parser.add_argument(
        "session_id",
        nargs="?",
        default="latest",
        help=(
            "coding-session ID, exact label, unique ID prefix, or 'latest' "
            "(default: latest session)"
        ),
    )
    usage_parser.set_defaults(func=cmd_usage)

    worktree_parser = sub.add_parser(
        "worktree", help="inspect or remove retained session worktrees"
    )
    worktree_sub = worktree_parser.add_subparsers(
        dest="worktree_command", required=True
    )
    worktree_rm = worktree_sub.add_parser(
        "rm", help="remove one Yuj-owned worktree and its branch"
    )
    worktree_rm.add_argument(
        "session_id",
        help="coding-session ID, exact label, or unique ID prefix",
    )
    worktree_rm.add_argument(
        "--force",
        action="store_true",
        help="discard uncommitted files and unmerged commits",
    )
    worktree_rm.set_defaults(func=cmd_worktree_rm)

    if not argv or argv[0] not in sub.choices:
        command_rows = "\n".join(
            f"  {action.dest:<12} {action.help}"
            for action in sub._choices_actions
        )
        session_parser = argparse.ArgumentParser(
            prog=CLI_NAME,
            description="Start a Yuj coding session",
            epilog=(
                f"Commands:\n{command_rows}\n\n"
                "Run 'yuj COMMAND --help' for one command's options."
            ),
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        _attach_run_args(session_parser)
        args = session_parser.parse_args(argv)
    else:
        args = parser.parse_args(argv)
    try:
        return args.func(args)
    except ProviderAuthError as exc:
        raise SystemExit(f"{exc.code}: {exc}") from exc
    except ImageInputError as exc:
        raise SystemExit(f"image input error: {exc}") from exc
    except KeyboardInterrupt:
        sys.stderr.write("\n")
        return 130
    except EOFError:
        raise SystemExit("input closed") from None


def cmd_run(args) -> int:
    pending_images = read_image_inputs(args.image)
    if not args.dry_run:
        _maybe_offer_first_run_setup(args)
    prompt_text, prompt_source = _resolve_prompt_input(args)
    config_paths, context_mode = _effective_run_settings(args)

    auth_store = CredentialStore()
    auth_store.require_outside_target(args.cwd)
    auth_binding = _auth_binding_for_args(args, store=auth_store)
    transport_overrides = _transport_overrides_from_args(
        args, auth_binding=auth_binding
    )
    if pending_images:
        provisional_overrides = {
            "runtime_mode": "assistant",
            "max_sessions": 1,
            **transport_overrides,
        }
        if args.model:
            provisional_overrides["model"] = resolve_model(args.model)
        provisional_model = load_config(
            user_config=config_paths,
            overrides=provisional_overrides,
        ).model
        validate_image_capability(
            config_paths,
            model=provisional_model,
            config_overrides=transport_overrides,
            auth_binding=auth_binding,
        )
    try:
        preflight = preflight_assistant_startup(
            config_paths=config_paths,
            cwd=args.cwd,
            context_mode=context_mode,
            requested_model=args.model,
            config_overrides=transport_overrides,
            system_prompt_file=args.system_prompt,
            auth_binding=auth_binding,
            auth_store=auth_store,
        )
    except Exception as exc:
        raise SystemExit(f"startup preflight failed: {exc}") from exc
    if args.dry_run:
        sys.stdout.write(render_startup_preflight(preflight))
        return 0

    store = SessionStore()
    model, served = _resolve_model_or_exit(
        config_paths,
        requested_model=args.model,
        config_overrides=transport_overrides,
        auth_binding=auth_binding,
        auth_store=auth_store,
    )
    if pending_images:
        validate_image_capability(
            config_paths,
            model=model,
            config_overrides=transport_overrides,
            auth_binding=auth_binding,
        )
    record = create_session(
        store,
        cwd=args.cwd.resolve(),
        prompt_text=prompt_text,
        prompt_source=prompt_source,
        model=model,
        config_paths=config_paths,
        system_prompt_path=args.system_prompt.resolve() if args.system_prompt else None,
        context_mode=context_mode,
        auth_binding=auth_binding,
    )
    record = _persist_session_config_overlay(
        store,
        record,
        base_config_paths=config_paths,
        transport_overrides=transport_overrides,
    )
    if pending_images:
        save_image_segment(
            record.artifact_path,
            segment_number=1,
            prompt_text=prompt_text,
            images=pending_images,
        )
    _print_session_start(
        record,
        action="starting",
        served_models=served,
    )
    try:
        with _session_lock(store, record), TraceFollower(record.artifact_path):
            success, finish_reason = run_session(store, record, resume=False)
    except KeyboardInterrupt:
        return _handle_keyboard_interrupt(store, record)
    refreshed = store.get_session(record.session_id)
    _print_session_result(refreshed or record, success, finish_reason)
    return 0 if success else 1


def cmd_config(args) -> int:
    """Validate and explain assistant startup settings without side effects."""
    resolved = None
    try:
        config_paths, context_mode = _effective_run_settings(args)
        from ..llm_solver.harness.context_strategies import resolve_context_class

        resolve_context_class(context_mode)
        transport_overrides = _transport_overrides_from_args(args)
        overrides = {
            "runtime_mode": "assistant",
            "max_sessions": 1,
            **transport_overrides,
        }
        if args.model:
            overrides["model"] = resolve_model(args.model)
        base_label = "treatment" if args.treatment else "plain"
        layer_specs = [
            ConfigLayerSpec(
                path=config_paths[0],
                layer_id="base",
                kind="base",
                label=base_label,
            ),
            *[
                ConfigLayerSpec(
                    path=path,
                    layer_id=f"overlay-{index}",
                    kind="overlay",
                    label=f"--config[{index}]",
                )
                for index, path in enumerate(config_paths[1:], 1)
            ],
        ]
        resolved = resolve_config(
            user_config=config_paths,
            overrides=overrides,
            layer_specs=layer_specs,
        )
        references = validate_configuration_references(
            resolved.config,
            named_agents=args.agent,
        )
        resource_report = validate_runtime_resources().to_dict()
        resource_report["root"] = "<yuj-root>"
        references["runtime_resources"] = resource_report
        sandbox_resolution = inspect_sandbox_selection(resolved.config)
        document = build_inspection_document(
            resolved,
            success=True,
            selection={
                "base": base_label,
                "treatment": bool(args.treatment),
                "context_mode": context_mode,
                "context_source": "command-line" if args.context else "base",
                "sandbox": sandbox_resolution.as_dict(),
            },
            references=references,
        )
    except SandboxResolutionError as exc:
        if resolved is None:
            document = build_error_document(str(exc), code="sandbox_unavailable")
        else:
            capabilities = probe_sandbox_capabilities(
                bwrap_bin=resolved.config.bwrap_bin
            )
            document = build_inspection_document(
                resolved,
                success=False,
                selection={
                    "sandbox": {
                        **capabilities.as_dict(),
                        "selected": resolved.config.sandbox_backend,
                        "resolved": None,
                        "explicit_unsandboxed": False,
                    },
                },
                diagnostics=[{
                    "level": "error",
                    "code": "sandbox_unavailable",
                    "message": str(exc),
                }],
            )
        output = (
            render_inspection_json(document)
            if args.json_output
            else render_inspection_human(document)
        )
        sys.stdout.write(output)
        return 1
    except (Exception, SystemExit) as exc:
        detail = exc.code if isinstance(exc, SystemExit) else exc
        message = sanitize_diagnostic_message(detail, resolved=resolved)
        document = build_error_document(message)
        output = (
            render_inspection_json(document)
            if args.json_output
            else render_inspection_human(document)
        )
        sys.stdout.write(output)
        return 1

    output = (
        render_inspection_json(document)
        if args.json_output
        else render_inspection_human(document)
    )
    sys.stdout.write(output)
    return 0


def cmd_smoke(args) -> int:
    _maybe_offer_first_run_setup(args)
    smoke_root = prepare_smoke_repo(args.root)
    config_paths, context_mode = _effective_run_settings(args)
    auth_store = CredentialStore()
    auth_store.require_outside_target(smoke_root)
    auth_binding = _auth_binding_for_args(args, store=auth_store)
    transport_overrides = _transport_overrides_from_args(
        args, auth_binding=auth_binding
    )
    try:
        preflight_assistant_startup(
            config_paths=config_paths,
            cwd=smoke_root,
            context_mode=context_mode,
            requested_model=args.model,
            config_overrides=transport_overrides,
            system_prompt_file=args.system_prompt,
            auth_binding=auth_binding,
            auth_store=auth_store,
        )
    except Exception as exc:
        raise SystemExit(f"startup preflight failed: {exc}") from exc
    model, served = _resolve_smoke_model_or_exit(
        config_paths,
        requested_model=args.model,
        config_overrides=transport_overrides,
        auth_binding=auth_binding,
        auth_store=auth_store,
    )
    prompt_text = (
        "Fix the bug in calc.py so tests/test_calc.py passes. "
        "Make the smallest correct code change, run the relevant test, then finish."
    )
    store = SessionStore(args.assist_home.resolve()) if args.assist_home else SessionStore()
    record = create_session(
        store,
        cwd=smoke_root,
        prompt_text=prompt_text,
        prompt_source="smoke",
        model=model,
        config_paths=config_paths,
        system_prompt_path=args.system_prompt.resolve() if args.system_prompt else None,
        context_mode=context_mode,
        auth_binding=auth_binding,
    )
    record = _persist_session_config_overlay(
        store,
        record,
        base_config_paths=config_paths,
        transport_overrides=transport_overrides,
    )
    print(f"smoke_repo: {smoke_root}")
    _print_session_start(
        record,
        action="starting smoke session",
        served_models=served,
    )
    try:
        with _session_lock(store, record):
            success, finish_reason = run_session(store, record, resume=False)
    except KeyboardInterrupt:
        return _handle_keyboard_interrupt(store, record)
    refreshed = store.get_session(record.session_id)
    final_record = refreshed or record
    _print_session_result(final_record, success, finish_reason)

    acceptance_ok, reasons = _smoke_acceptance_check(smoke_root, final_record)
    if not acceptance_ok:
        print("smoke acceptance failed:")
        for reason in reasons:
            print(f"  - {reason}")
        print(f"smoke_repo: {smoke_root}")
        print(f"session_id: {final_record.session_id}")
        print(f"artifacts: {final_record.artifact_dir}")
        print(f"status: {final_record.status}")
        if finish_reason:
            print(f"finish_reason: {finish_reason}")
        return 1
    return 0 if success else 1


def _smoke_acceptance_check(smoke_root: Path, record) -> tuple[bool, list[str]]:
    """Return whether the edit, tests, and interactive gates are complete."""
    reasons: list[str] = []
    if record.worktree_path:
        smoke_root = Path(record.worktree_path)

    calc_path = Path(smoke_root) / "calc.py"
    if not calc_path.is_file():
        reasons.append(f"{calc_path} is missing")
    else:
        contents = calc_path.read_text()
        if "return a + b" not in contents:
            reasons.append(f"{calc_path} does not contain the fixed 'return a + b' body")

    if not reasons:
        test_result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/test_calc.py", "-q"],
            cwd=str(smoke_root),
            capture_output=True,
            text=True,
            timeout=60,
        )
        if test_result.returncode != 0:
            tail = (test_result.stdout or test_result.stderr).strip().splitlines()[-5:]
            reasons.append("tests/test_calc.py failed: " + " | ".join(tail))

    approval = load_approval_request(record.artifact_path)
    if approval is not None and approval.get("status") == "pending":
        reasons.append("session has a pending approval request")
    try:
        clarification = clarification_state(record.artifact_path)
    except ClarificationStateError as exc:
        reasons.append(f"session has invalid clarification evidence: {exc}")
    else:
        if clarification.phase in {"input_required", "input_ready"}:
            reasons.append(
                "session has an unresolved clarification exchange"
            )

    return (not reasons, reasons)


def cmd_resume(args) -> int:
    resume_prompt_text, resume_prompt_source, pending_images = (
        _resolve_resume_input(args)
    )
    store = SessionStore()
    record = _resolve_session_record(store, args.session_id, selector="resumable")
    try:
        _correction_state_for_record(record)
    except CorrectionStateError as exc:
        raise SystemExit(f"invalid correction evidence: {exc}") from exc
    try:
        clarification = clarification_state(record.artifact_path)
    except ClarificationStateError as exc:
        raise SystemExit(f"invalid clarification evidence: {exc}") from exc
    if clarification.phase == "input_required":
        assert clarification.request is not None
        raise SystemExit(
            "session has a pending clarification; run "
            f"{CLI_NAME} answer {record.short_id} "
            f"{clarification.request['request_id']} '<answer>' first"
        )
    approval = load_approval_request(record.artifact_path)
    if approval is not None and approval.get("status") == "pending":
        raise SystemExit(
            "session has a pending approval request; run "
            f"{CLI_NAME} approve {record.session_id} first"
        )
    if record.status == "completed":
        if resume_prompt_text is not None:
            raise SystemExit("cannot add follow-up input to a completed session")
        _print_session_result(record, True, record.last_finish_reason)
        return 0
    if pending_images:
        validate_image_capability(
            [Path(path) for path in record.config_paths],
            model=record.model,
            auth_binding=_record_auth_binding(record),
        )
    store.set_active_session(record.cwd, record.session_id)
    _print_session_start(record, action="resuming")
    try:
        with _session_lock(store, record), TraceFollower(record.artifact_path):
            live = derive_live_state(record.artifact_path)
            next_session_number = max(1, live.session_number + 1)
            if resume_prompt_text is not None:
                assert resume_prompt_source is not None
                _record_resume_prompt_source(
                    record.artifact_path,
                    session_number=next_session_number,
                    prompt_text=resume_prompt_text,
                    prompt_source=resume_prompt_source,
                )
            if pending_images:
                assert resume_prompt_text is not None
                save_image_segment(
                    record.artifact_path,
                    segment_number=next_session_number,
                    prompt_text=resume_prompt_text,
                    images=pending_images,
                )
            if resume_prompt_text is not None:
                success, finish_reason = run_session(
                    store,
                    record,
                    resume=True,
                    resume_prompt_text=resume_prompt_text,
                )
            else:
                success, finish_reason = run_session(
                    store, record, resume=True
                )
    except KeyboardInterrupt:
        return _handle_keyboard_interrupt(store, record)
    refreshed = store.get_session(record.session_id)
    _print_session_result(refreshed or record, success, finish_reason)
    return 0 if success else 1


def cmd_correct(args) -> int:
    if args.session_id.lower() in _LATEST_SESSION_TOKENS:
        raise SystemExit("correct requires an explicit session reference")
    store = SessionStore()
    record = _resolve_session_record(
        store, args.session_id, selector="latest"
    )
    try:
        with _session_lock(store, record):
            state = _correction_state_for_record(record)
            if state.phase == "pending":
                raise CorrectionStateError(
                    "session already has a pending correction"
                )
            if state.phase == "consumed":
                raise CorrectionStateError(
                    "this session already has a correction"
                )
            if record.status == "completed":
                raise CorrectionStateError(
                    "cannot correct a completed session"
                )
            if record.status == "archived":
                raise CorrectionStateError(
                    "cannot correct an archived session"
                )
            known_resumable_rows = {
                "paused",
                "error",
                "approval_pending",
                "input_required",
                "input_ready",
                "running",
            }
            if record.status not in known_resumable_rows:
                raise CorrectionStateError(
                    f"session status {record.status!r} is not resumable"
                )
            live = derive_live_state(record.artifact_path)
            status = live.status or record.status
            if status == "running":
                raise CorrectionStateError(
                    "cannot correct an active session"
                )
            if status not in {
                "paused",
                "error",
                "approval_pending",
                "input_required",
                "input_ready",
            }:
                raise CorrectionStateError(
                    f"session status {status!r} is not resumable"
                )
            if live.session_number <= 0:
                raise CorrectionStateError(
                    "session has no stopped run boundary to correct"
                )
            correction = create_correction(
                record.artifact_path,
                correction_id=f"corr-{uuid.uuid4().hex[:12]}",
                session_id=record.session_id,
                after_session_number=live.session_number,
                text=args.correction,
            )
            append_trace_event_fsync(
                record.artifact_path / ".trace.jsonl",
                {
                    "event": "correction_created",
                    "trace_schema_version": TRACE_SCHEMA_VERSION,
                    "session_number": live.session_number,
                    "correction_id": correction["correction_id"],
                    "text_sha256": correction["text_sha256"],
                    "text_chars": len(correction["text"]),
                },
            )
    except CorrectionStateError as exc:
        raise SystemExit(str(exc)) from exc
    print(f"corrected: {record.session_id}")
    print(f"session_ref: {record.short_id}")
    print("correction: pending")
    print(f"correction_id: {correction['correction_id']}")
    print(f"correction_file: {correction_path(record.artifact_path)}")
    print(f"resume with: {CLI_NAME} resume {record.short_id}")
    return 0


def cmd_answer(args) -> int:
    store = SessionStore()
    record = _resolve_session_record(
        store, args.session_id, selector="latest"
    )
    try:
        with _session_lock(store, record):
            request_state = clarification_state(record.artifact_path)
            if request_state.phase != "input_required":
                if request_state.phase in {"input_ready", "consumed"}:
                    raise ClarificationStateError(
                        "clarification already has an answer"
                    )
                raise ClarificationStateError(
                    "session has no pending clarification"
                )
            answer = record_clarification_answer(
                record.artifact_path,
                session_id=record.session_id,
                request_id=args.request_id,
                answer=args.answer,
            )
            assert request_state.request is not None
            request = request_state.request
            append_trace_event_fsync(
                record.artifact_path / ".trace.jsonl",
                {
                    "event": "clarification_answer",
                    "trace_schema_version": TRACE_SCHEMA_VERSION,
                    "session_number": request["session_number"],
                    "turn_number": request["turn_number"],
                    "request_id": request["request_id"],
                    "answer_sha256": answer["answer_sha256"],
                    "answer_chars": len(answer["answer"]),
                },
            )
            store.update_session(
                record.session_id,
                status="input_ready",
                last_finish_reason="input_answered",
            )
    except ClarificationStateError as exc:
        raise SystemExit(str(exc)) from exc
    print(f"answered: {record.session_id}")
    print(f"session_ref: {record.short_id}")
    print(f"request_id: {answer['request_id']}")
    print(f"answer_file: {clarification_answer_path(record.artifact_path)}")
    print(f"resume with: {CLI_NAME} resume {record.short_id}")
    return 0


def cmd_rewind(args) -> int:
    store = SessionStore()
    record = _resolve_session_record(
        store, args.session_id, selector="latest"
    )
    try:
        with _session_lock(store, record):
            correction = _correction_state_for_record(record)
            if correction.phase == "pending":
                raise CorrectionStateError(
                    "session has a pending correction; resume it before rewind"
                )
            event = rewind_session(
                store,
                record,
                turn=args.turn,
                reason=args.reason,
            )
    except (CorrectionStateError, RuntimeError, WorktreeRuntimeError) as exc:
        raise SystemExit(str(exc)) from exc
    print(f"rewound: {record.session_id}")
    print(f"session_ref: {record.short_id}")
    print(f"from_turn: {event['from_turn']}")
    print(f"to_turn: {event['to_turn']}")
    print(f"commit: {event['commit']}")
    print(f"reason: {event['reason']}")
    print(f"resume with: {CLI_NAME} resume {record.short_id}")
    return 0


def cmd_setup(args) -> int:
    config_path = _config_local_path()
    if config_path.exists() and not args.force:
        if not _is_interactive():
            raise SystemExit(f"{config_path} already exists; pass --force to overwrite")
        answer = input(f"{config_path} already exists. Overwrite? [y/N]: ").strip().lower()
        if answer not in {"y", "yes"}:
            print(f"unchanged: {config_path}")
            return 0

    provider = args.provider or _prompt_choice(
        "Provider",
        choices=[
            "local",
            "claude",
            "codex",
            "openai",
            "anthropic",
            "openrouter",
            "zai",
            "custom",
        ],
        default="local",
    )
    permission_preset = getattr(args, "permission_preset", None)
    if permission_preset is None and _is_interactive():
        selected = _prompt_choice(
            "Permission preset",
            choices=["none", *ASSISTANT_PERMISSION_PRESET_NAMES],
            default="none",
        )
        permission_preset = "" if selected == "none" else selected
    permission_preset = permission_preset or ""
    sandbox_choice = getattr(args, "sandbox", None)
    if sandbox_choice is None:
        sandbox_choice = _prompt_choice(
            "Sandbox",
            choices=["bwrap", "auto", "docker", "podman", "none"],
            default="bwrap",
        )
    sandbox_image = str(getattr(args, "sandbox_image", None) or "")
    if (
        sandbox_choice in {"docker", "podman"}
        and not sandbox_image
        and _is_interactive()
    ):
        sandbox_image = _prompt_required("Local sandbox image reference")
    if sandbox_choice in {"docker", "podman"} and not sandbox_image:
        raise SystemExit(
            f"--sandbox-image is required with --sandbox {sandbox_choice}"
        )
    capabilities = probe_sandbox_capabilities(bwrap_bin="/usr/bin/bwrap")
    try:
        sandbox_resolution = resolve_sandbox_selection(
            sandbox_choice, capabilities
        )
    except SandboxResolutionError as exc:
        print(f"sandbox_platform: {capabilities.platform}")
        print(
            "sandbox_supported: "
            + (", ".join(capabilities.supported) or "none")
        )
        print(
            "sandbox_installed: "
            + (", ".join(capabilities.installed) or "none")
        )
        print(
            "sandbox_available: "
            + (", ".join(capabilities.available) or "none")
        )
        print(
            "sandbox_unavailable: "
            + (", ".join(capabilities.unavailable) or "none")
        )
        print(f"sandbox_selected: {sandbox_choice}")
        print("sandbox_resolved: unavailable")
        raise SystemExit(f"sandbox setup failed: {exc}") from exc
    if (
        sandbox_resolution.resolved in {"docker", "podman"}
        and not sandbox_image
    ):
        if _is_interactive():
            sandbox_image = _prompt_required("Local sandbox image reference")
        else:
            raise SystemExit(
                "--sandbox-image is required because --sandbox "
                f"{sandbox_choice} resolves to "
                f"{sandbox_resolution.resolved} on this host"
            )
    preset = dict(_PROVIDER_PRESETS[provider])
    auth_method: str | None = None
    binding: AuthBinding | None = None
    managed_store: CredentialStore | None = None
    if provider in _MANAGED_PROVIDERS:
        managed_store = CredentialStore()
        managed_store.require_outside_current_repository()
        auth_choice = args.auth or _prompt_choice(
            "Authentication",
            choices=["api-key", "subscription"],
            default="api-key",
        )
        auth_method = auth_choice.replace("-", "_")
        spec = provider_spec(provider)
        base_url = (
            spec.subscription_base_url
            if auth_method == "subscription"
            else spec.api_key_base_url
        )
        if args.base_url and args.base_url.rstrip("/") != base_url.rstrip("/"):
            raise SystemExit(
                f"--base-url cannot change the {provider} {auth_method} endpoint"
            )
        model = args.model or _prompt_required("Default model id")
        binding = _save_managed_login(
            provider,
            auth_method=auth_method,
            api_key=args.api_key,
            api_key_env=args.api_key_env,
            store=managed_store,
        )
        api_key = "yuj-host-credential"
    elif args.auth:
        raise SystemExit("--auth is supported only with --provider claude or codex")
    elif provider == "local":
        base_url = args.base_url or _prompt_default(
            "Local OpenAI-compatible base URL",
            "http://localhost:8080/v1",
        )
        api_key = args.api_key or _api_key_ref_or_value(args.api_key_env, default="local")
        model = args.model or _prompt_default(
            "Default model id/alias",
            "qwen3-vl-8b",
        )
    elif provider == "custom":
        base_url = args.base_url or _prompt_required("API base URL")
        api_key = args.api_key or _api_key_ref_or_value(args.api_key_env)
        model = args.model or _prompt_required("Default model id")
        preset["provider"] = "openai-compatible"
    else:
        base_url = args.base_url or str(preset["base_url"])
        api_key = args.api_key or _api_key_ref_or_value(args.api_key_env)
        model = args.model or _prompt_required("Default model id")

    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(_render_local_config(
        provider=str(preset["provider"]),
        base_url=base_url,
        api_key=api_key,
        model=model,
        permission_preset=permission_preset,
        sandbox=sandbox_choice,
        sandbox_image=sandbox_image,
    ))
    if binding is not None:
        assert managed_store is not None
        managed_store.select(binding)
    print(f"wrote: {config_path}")
    print(f"provider: {provider}")
    if auth_method is not None:
        print(f"authentication: {auth_method}")
    else:
        CredentialStore().clear_selection()
    print(f"model: {model}")
    if permission_preset:
        print(f"permission_preset: {permission_preset}")
    print(f"sandbox_platform: {capabilities.platform}")
    print(
        "sandbox_supported: "
        + (", ".join(capabilities.supported) or "none")
    )
    print(
        "sandbox_installed: "
        + (", ".join(capabilities.installed) or "none")
    )
    print(
        "sandbox_available: "
        + (", ".join(capabilities.available) or "none")
    )
    print(
        "sandbox_unavailable: "
        + (", ".join(capabilities.unavailable) or "none")
    )
    print(f"sandbox_selected: {sandbox_resolution.selected}")
    print(f"sandbox_resolved: {sandbox_resolution.resolved}")
    return 0


def cmd_login(args) -> int:
    auth_method = args.auth.replace("-", "_")
    store = CredentialStore()
    store.require_outside_current_repository()
    binding = _save_managed_login(
        args.provider,
        auth_method=auth_method,
        api_key=args.api_key,
        api_key_env=args.api_key_env,
        store=store,
    )
    store.select(binding)
    print(f"active_provider: {binding.provider}")
    print(f"authentication: {binding.auth_method}")
    print("credential: stored")
    return 0


def cmd_logout(args) -> int:
    store = CredentialStore()
    provider = args.provider
    if provider is None:
        active = store.active_binding()
        if active is None:
            raise CredentialMissingError(
                "active", "no provider credential is selected"
            )
        provider = active.provider
    removed = store.logout(provider)
    print(f"provider: {provider}")
    print(f"credential_removed: {str(removed).lower()}")
    return 0


def cmd_auth_status(_args) -> int:
    store = CredentialStore()
    binding = store.active_binding()
    if binding is None:
        print("active_provider: none")
        print("authentication: none")
        print("credential: missing")
        return 1
    store.load(binding.provider, expected_binding=binding)
    print(f"active_provider: {binding.provider}")
    print(f"authentication: {binding.auth_method}")
    print("credential: stored")
    return 0


def _save_managed_login(
    provider: str,
    *,
    auth_method: str,
    api_key: str | None,
    api_key_env: str | None,
    store: CredentialStore | None = None,
) -> AuthBinding:
    store = store or CredentialStore()
    if auth_method == "subscription":
        if api_key or api_key_env:
            raise SystemExit(
                "subscription authentication does not accept an API key"
            )
        return browser_sign_in(provider, store=store)
    if auth_method != "api_key":
        raise SystemExit(f"unsupported authentication method: {auth_method}")
    if api_key and api_key_env:
        raise SystemExit("provide only one of --api-key or --api-key-env")
    if not api_key and not api_key_env:
        if not _is_interactive():
            raise SystemExit(
                "API-key authentication requires --api-key or --api-key-env"
            )
        api_key = getpass.getpass("API key: ").strip()
        if not api_key:
            raise SystemExit("API key is required")
    return store.save_api_key(
        provider,
        secret=api_key,
        environment=api_key_env,
    )


def cmd_models(args) -> int:
    cfg = _load_assistant_config(args.config)
    auth_store = CredentialStore()
    auth_binding = auth_store.active_binding()
    client = _make_client(
        cfg,
        profile=None,
        auth_binding=auth_binding,
        auth_store=auth_store,
    )
    models = client.health_check()
    print(f"provider: {cfg.provider}")
    print(f"base_url: {cfg.base_url}")
    if not models:
        print("(no models returned)")
        return 1
    for model in models:
        marker = " *" if model == cfg.model else ""
        print(f"{model}{marker}")
    return 0


def cmd_doctor(args) -> int:
    failures = 0
    cfg = None
    try:
        cfg = _load_assistant_config(args.config)
        print(f"config: ok ({_config_local_path()})")
        print(f"provider: {cfg.provider}")
        print(f"base_url: {cfg.base_url}")
        print(f"model: {cfg.model}")
    except Exception as exc:
        failures += 1
        print(f"config: fail ({exc})")

    if cfg is not None:
        capabilities = probe_sandbox_capabilities(
            bwrap_bin=str(getattr(cfg, "bwrap_bin", "/usr/bin/bwrap"))
        )
        print(f"sandbox_platform: {capabilities.platform}")
        print(
            "sandbox_supported: "
            + (", ".join(capabilities.supported) or "none")
        )
        print(
            "sandbox_installed: "
            + (", ".join(capabilities.installed) or "none")
        )
        print(
            "sandbox_available: "
            + (", ".join(capabilities.available) or "none")
        )
        print(
            "sandbox_unavailable: "
            + (", ".join(capabilities.unavailable) or "none")
        )
        print(
            "sandbox_selected: "
            + str(getattr(cfg, "sandbox_backend", "bwrap"))
        )
        try:
            sandbox_resolution = preflight_sandbox(
                cfg, capabilities=capabilities
            )
        except SandboxResolutionError as exc:
            failures += 1
            print("sandbox_resolved: unavailable")
            print(f"sandbox: fail ({exc})")
        else:
            print(f"sandbox_resolved: {sandbox_resolution.resolved}")
            print(
                "sandbox: ok "
                f"(engaged={str(bool(sandbox_resolution.engaged)).lower()}, "
                "explicit_unsandboxed="
                f"{str(sandbox_resolution.explicit_unsandboxed).lower()})"
            )

        try:
            auth_store = CredentialStore()
            models = _make_client(
                cfg,
                profile=None,
                auth_binding=auth_store.active_binding(),
                auth_store=auth_store,
            ).health_check()
            print(f"models: ok ({len(models)} returned)")
            if cfg.model in models:
                print("selected_model: ok")
            elif cfg.provider == "openai-compatible" and cfg.base_url.startswith("http://localhost"):
                failures += 1
                print("selected_model: fail (not served by local /v1/models)")
            else:
                print("selected_model: warn (not listed; hosted providers may still accept explicit ids)")
        except Exception as exc:
            failures += 1
            print(f"models: fail ({exc})")

    if _config_local_path().exists():
        print("local_config: ok")
    else:
        print(f"local_config: warn (missing; run {CLI_NAME} setup)")

    if Path.cwd().joinpath(".git").exists():
        print("git_repo: ok")
    else:
        print("git_repo: warn (current directory is not a git repo root)")

    return 1 if failures else 0


def cmd_sessions(args) -> int:
    store = SessionStore()
    sessions = store.list_sessions(limit=args.limit, archived=bool(args.archived))
    if not sessions:
        noun = (
            "archived assistant sessions"
            if args.archived
            else "assistant sessions"
        )
        print(f"(no {noun})")
        return 0
    current_cwd = str(Path.cwd().resolve())
    active_ids = store.list_active_session_ids()
    locked_ids = store.list_locked_session_ids()
    print(
        "session_id                             status     "
        "label                 ref       flags               model  cwd"
    )
    for record in sessions:
        flags: list[str] = []
        if record.session_id in active_ids:
            flags.append("active")
        if record.session_id in locked_ids:
            flags.append("locked")
        if record.cwd == current_cwd:
            flags.append("cwd")
        if record.archived_at is not None:
            flags.append("archived")
        flag_text = ",".join(flags) if flags else "-"
        label_text = record.label if record.label is not None else "-"
        print(
            f"{record.session_id}  {record.status:9s}  {label_text:20s}  "
            f"{record.short_id:8s}  {flag_text:18s}  {record.model}  {record.cwd}"
        )
        if record.last_finish_reason:
            print(f"    last_finish_reason={record.last_finish_reason}")
        if record.archived_at is not None:
            print(f"    archived_at={record.archived_at}")
    return 0


def cmd_archive(args) -> int:
    if args.session_id.lower() in _LATEST_SESSION_TOKENS:
        raise SystemExit("archive requires an explicit session reference")
    store = SessionStore()
    record = _resolve_session_record(
        store,
        args.session_id,
        selector="latest",
        allow_archived=True,
    )
    if record.archived_at is None:
        live = derive_live_state(record.artifact_path)
        if (live.status or record.status) == "running":
            raise SystemExit("cannot archive a running session")
        approval = load_approval_request(record.artifact_path)
        if approval is not None and approval.get("status") == "pending":
            raise SystemExit("cannot archive a session with a pending approval")
        try:
            clarification = clarification_state(record.artifact_path)
        except ClarificationStateError as exc:
            raise SystemExit(f"invalid clarification evidence: {exc}") from exc
        if clarification.phase in {"input_required", "input_ready"}:
            raise SystemExit(
                "cannot archive a session with a pending clarification"
            )
        try:
            correction = _correction_state_for_record(record)
        except CorrectionStateError as exc:
            raise SystemExit(f"invalid correction evidence: {exc}") from exc
        if correction.phase == "pending":
            raise SystemExit("cannot archive a session with a pending correction")
    try:
        archived, changed = store.archive_session(record.session_id)
    except SessionArchiveError as exc:
        raise SystemExit(str(exc)) from exc
    print(f"session_id: {archived.session_id}")
    print(f"session_ref: {archived.short_id}")
    print("archive: archived")
    print(f"archived_at: {archived.archived_at}")
    print(f"changed: {'yes' if changed else 'no'}")
    print(f"next: {CLI_NAME} unarchive {archived.short_id}")
    return 0


def cmd_fork(args) -> int:
    if args.session_id.lower() in _LATEST_SESSION_TOKENS:
        raise SystemExit("fork requires an explicit session reference")
    store = SessionStore()
    source = _resolve_session_record(
        store,
        args.session_id,
        selector="latest",
        allow_archived=True,
    )
    try:
        result = fork_saved_session(store, source)
    except ForkSessionError as exc:
        raise SystemExit(str(exc)) from exc
    child = result.child
    print(f"forked: {child.session_id}")
    print(f"session_ref: {child.short_id}")
    print(f"parent_session_id: {source.session_id}")
    print(f"parent_session_ref: {source.short_id}")
    print(f"status: {child.status}")
    print(f"cwd: {child.cwd}")
    print(f"artifacts: {child.artifact_dir}")
    if child.worktree_path is not None:
        print(f"worktree: {child.worktree_path}")
        print(f"worktree_branch: {child.worktree_branch}")
    print(f"source_artifacts_sha256: {result.source_artifact_sha256}")
    print(f"resume with: {CLI_NAME} resume {child.short_id}")
    return 0


def cmd_unarchive(args) -> int:
    if args.session_id.lower() in _LATEST_SESSION_TOKENS:
        raise SystemExit("unarchive requires an explicit session reference")
    store = SessionStore()
    record = _resolve_session_record(
        store,
        args.session_id,
        selector="latest",
        allow_archived=True,
    )
    try:
        unarchived, changed = store.unarchive_session(record.session_id)
    except SessionArchiveError as exc:
        raise SystemExit(str(exc)) from exc
    print(f"session_id: {unarchived.session_id}")
    print(f"session_ref: {unarchived.short_id}")
    print("archive: unarchived")
    print("archived_at: -")
    print(f"changed: {'yes' if changed else 'no'}")
    print("next: none")
    return 0


def cmd_purge(args) -> int:
    if not is_full_session_id(args.session_id):
        raise SystemExit("purge requires one full immutable session ID")
    if args.preview and args.confirm is not None:
        raise SystemExit("--preview and --confirm are mutually exclusive")
    if not args.preview and args.confirm is None:
        raise SystemExit("purge requires --preview or --confirm FULL_SESSION_ID")
    if args.confirm is not None and args.confirm != args.session_id:
        raise SystemExit(
            "purge confirmation must be the same full immutable session ID"
        )

    try:
        if args.preview:
            try:
                store = SessionStore(read_only=True)
            except FileNotFoundError as exc:
                raise SystemExit("no assistant sessions found") from exc
            preview = preview_session_purge(store, args.session_id)
            _print_purge_preview(preview, mutation="none")
            if preview.state != "completed":
                print(
                    f"next: {CLI_NAME} purge {args.session_id} "
                    f"--confirm {args.session_id}"
                )
            return 0

        store = SessionStore()
        preview = purge_archived_session(store, args.session_id)
    except PurgeSessionError as exc:
        if exc.preview is not None:
            _print_purge_preview(exc.preview, mutation="incomplete")
        raise SystemExit(str(exc)) from exc

    print(f"purged: {preview.session_id}")
    _print_purge_preview(preview, mutation="completed")
    return 0


def _print_purge_preview(preview: PurgePreview, *, mutation: str) -> None:
    print(f"session_id: {preview.session_id}")
    print(f"purge_state: {preview.state}")
    print(f"artifact_entries: {preview.entry_count}")
    print(f"estimated_bytes: {preview.estimated_bytes}")
    print(f"remaining_entries: {preview.remaining_entries}")
    print(f"remaining_bytes: {preview.remaining_bytes}")
    if preview.failure_detail is not None:
        print(f"last_failure: {preview.failure_detail}")
    for entry in preview.entries:
        relative = json.dumps(entry.relative, ensure_ascii=True)
        print(f"artifact: {entry.kind} {relative} bytes={entry.size}")
    print(f"mutation: {mutation}")


def cmd_label(args) -> int:
    if args.clear and args.label is not None:
        raise SystemExit("label value and --clear are mutually exclusive")
    if not args.clear and args.label is None:
        raise SystemExit("provide a label or --clear")

    store = SessionStore()
    record = _resolve_session_record(store, args.session_id, selector="latest")
    try:
        if args.clear:
            store.clear_session_label(record.session_id)
            print(f"label_cleared: {record.session_id}")
            print(f"session_ref: {record.short_id}")
            print("label: -")
            return 0
        assert args.label is not None
        store.set_session_label(record.session_id, args.label)
    except SessionLabelError as exc:
        raise SystemExit(str(exc)) from exc
    print(f"labeled: {record.session_id}")
    print(f"session_ref: {record.short_id}")
    print(f"label: {args.label}")
    return 0


def cmd_approve(args) -> int:
    store = SessionStore()
    record = _resolve_session_record(store, args.session_id, selector="pending_approval")
    approval = load_approval_request(record.artifact_path)
    if approval is None or approval.get("status") != "pending":
        raise SystemExit(f"no pending approval request for session: {record.session_id}")
    approval["status"] = "approved"
    if args.always:
        approval["always"] = True
        _save_approval_decision(record.artifact_path, approval, "approved")
    save_approval_request(record.artifact_path, approval)
    print(f"approved: {record.session_id}")
    print(f"session_ref: {record.short_id}")
    if args.always:
        print("decision: always approve matching action in this session")
    print(f"request_file: {approval_request_path(record.artifact_path)}")
    print(f"resume with: {CLI_NAME} resume {record.short_id}")
    return 0


def cmd_reject(args) -> int:
    store = SessionStore()
    record = _resolve_session_record(store, args.session_id, selector="pending_approval")
    approval = load_approval_request(record.artifact_path)
    if approval is None or approval.get("status") != "pending":
        raise SystemExit(f"no pending approval request for session: {record.session_id}")
    approval["status"] = "rejected"
    approval["rejection_reason"] = args.reason
    if args.always:
        approval["always"] = True
        _save_approval_decision(record.artifact_path, approval, "rejected")
    save_approval_request(record.artifact_path, approval)
    print(f"rejected: {record.session_id}")
    print(f"session_ref: {record.short_id}")
    print(f"reason: {args.reason}")
    if args.always:
        print("decision: always reject matching action in this session")
    print(f"request_file: {approval_request_path(record.artifact_path)}")
    print(f"resume with: {CLI_NAME} resume {record.short_id}")
    return 0


def cmd_status(args) -> int:
    store = SessionStore()
    record = _resolve_session_record(
        store,
        args.session_id,
        selector="latest",
        allow_archived=True,
    )
    live = derive_live_state(record.artifact_path)
    status = live.status or record.status
    finish_reason = live.finish_reason if live.status else record.last_finish_reason
    turns = session_turn_count(record.artifact_path)
    approval = load_approval_request(record.artifact_path)
    clarification = clarification_state(record.artifact_path)
    try:
        correction = _correction_state_for_record(record)
    except CorrectionStateError as exc:
        raise SystemExit(f"invalid correction evidence: {exc}") from exc
    lock = store.get_session_lock(record.session_id)
    interrupt = load_interrupt_marker(record.artifact_path)
    sandbox = session_sandbox_provenance(record.artifact_path)

    print(f"session_id: {record.session_id}")
    if record.parent_session_id is not None:
        print(f"parent_session_id: {record.parent_session_id}")
    print(f"label: {record.label if record.label is not None else '-'}")
    print(f"session_ref: {record.short_id}")
    print(f"status: {status}")
    print(f"archived: {'yes' if record.archived_at is not None else 'no'}")
    if record.archived_at is not None:
        print(f"archived_at: {record.archived_at}")
    if finish_reason:
        print(f"finish_reason: {finish_reason}")
    print(f"turns: {turns}")
    print(f"cwd: {record.cwd}")
    print(f"model: {record.model}")
    if record.provider:
        print(f"provider: {record.provider}")
        print(f"authentication: {record.auth_method}")
    _print_sandbox_provenance(sandbox)
    _print_image_evidence(record.artifact_path)
    _print_correction_evidence(correction)
    if approval is not None and approval.get("status") == "pending":
        print("approval: pending")
    else:
        print("approval: none")
    if clarification.phase == "none":
        print("clarification: none")
    else:
        clarification_label = {
            "input_required": "pending",
            "input_ready": "answered",
        }.get(clarification.phase, clarification.phase)
        print(f"clarification: {clarification_label}")
        assert clarification.request is not None
        print(f"clarification_request_id: {clarification.request['request_id']}")
        print(f"question: {clarification.request['question']}")
    if lock is not None:
        print(f"lock: pid={lock.owner_pid} host={lock.owner_host}")
    else:
        print("lock: none")
    if interrupt is not None:
        print("interrupt: interrupted")
    else:
        print("interrupt: none")

    if record.archived_at is not None:
        print(f"next: {CLI_NAME} unarchive {record.short_id}")
    elif clarification.phase == "input_required":
        assert clarification.request is not None
        print(
            f"next: {CLI_NAME} answer {record.short_id} "
            f"{clarification.request['request_id']} '<answer>'"
        )
    elif approval is not None and approval.get("status") == "pending":
        print(f"next: {CLI_NAME} approve {record.short_id}")
    elif status in {"paused", "approval_pending", "input_ready"}:
        print(f"next: {CLI_NAME} resume {record.short_id}")
    elif status == "running":
        print(f"next: {CLI_NAME} show {record.short_id}")
    else:
        print("next: none")
    return 0


def cmd_current(_args) -> int:
    # Mirror `status latest` explicitly for a faster operator path.
    return cmd_status(argparse.Namespace(session_id="latest"))


def cmd_show(args) -> int:
    store = SessionStore()
    record = _resolve_session_record(
        store,
        args.session_id,
        selector="latest",
        allow_archived=True,
    )
    turns = session_turn_count(record.artifact_path)
    live = derive_live_state(record.artifact_path)
    status = live.status or record.status
    finish_reason = live.finish_reason if live.status else record.last_finish_reason
    print(f"session_id: {record.session_id}")
    if record.parent_session_id is not None:
        print(f"parent_session_id: {record.parent_session_id}")
    print(f"label: {record.label if record.label is not None else '-'}")
    print(f"session_ref: {record.short_id}")
    print(f"status: {status}")
    print(f"archived: {'yes' if record.archived_at is not None else 'no'}")
    if record.archived_at is not None:
        print(f"archived_at: {record.archived_at}")
    if live.session_number:
        print(f"current_session: {live.session_number}")
    print(f"created_at: {record.created_at}")
    print(f"updated_at: {record.updated_at}")
    print(f"cwd: {record.cwd}")
    print(f"artifacts: {record.artifact_dir}")
    print(f"model: {record.model}")
    if record.provider:
        print(f"provider: {record.provider}")
        print(f"authentication: {record.auth_method}")
    _print_sandbox_provenance(
        session_sandbox_provenance(record.artifact_path)
    )
    _print_image_evidence(record.artifact_path)
    print(f"context: {record.context_mode}")
    print(f"prompt_source: {record.prompt_source}")
    if record.system_prompt_path:
        print(f"system_prompt: {record.system_prompt_path}")
    if finish_reason:
        print(f"finish_reason: {finish_reason}")
    print(f"turns: {turns}")
    approval = load_approval_request(record.artifact_path)
    clarification = clarification_state(record.artifact_path)
    try:
        correction = _correction_state_for_record(record)
    except CorrectionStateError as exc:
        raise SystemExit(f"invalid correction evidence: {exc}") from exc
    lock = store.get_session_lock(record.session_id)
    interrupt = load_interrupt_marker(record.artifact_path)
    if approval is None:
        print("approval: none")
    else:
        print(f"approval: {approval.get('status')}")
        print(f"approval_reason: {approval.get('reason')}")
        print(f"approval_action: {approval.get('tool_name')}({approval.get('args_summary') or approval.get('cmd') or ''})")
    _print_correction_evidence(correction)
    if clarification.phase == "none":
        print("clarification: none")
    else:
        clarification_label = {
            "input_required": "pending",
            "input_ready": "answered",
        }.get(clarification.phase, clarification.phase)
        print(f"clarification: {clarification_label}")
        assert clarification.request is not None
        print(
            "clarification_request_id: "
            f"{clarification.request['request_id']}"
        )
        print(
            "clarification_question: "
            f"{clarification.request['question']}"
        )
    if record.archived_at is not None:
        print(f"next: {CLI_NAME} unarchive {record.short_id}")
    elif clarification.phase == "input_required":
        assert clarification.request is not None
        print(
            f"next: {CLI_NAME} answer {record.short_id} "
            f"{clarification.request['request_id']} '<answer>'"
        )
    elif approval is not None and approval.get("status") == "pending":
        print(f"next: {CLI_NAME} approve {record.short_id}")
    elif status in {"paused", "approval_pending", "input_ready"}:
        print(f"next: {CLI_NAME} resume {record.short_id}")
    elif status == "running":
        print(f"next: {CLI_NAME} show {record.short_id}")
    else:
        print("next: none")
    if lock is None:
        print("lock: none")
    else:
        print(f"lock: pid={lock.owner_pid} host={lock.owner_host} since={lock.acquired_at}")
    if interrupt is None:
        print("interrupt: none")
    else:
        print(f"interrupt: {interrupt.get('finish_reason')} at {interrupt.get('interrupted_at')}")
    turn_lines = session_turn_tail(record.artifact_path, limit=args.turns)
    if not turn_lines:
        print("recent_turns: (empty)")
    else:
        print("recent_turns:")
        for line in turn_lines:
            print(f"  {line}")
    trace_lines = session_trace_tail(record.artifact_path, limit=args.trace_lines)
    if not trace_lines:
        print("trace_tail: (empty)")
        return 0
    print("trace_tail:")
    for line in trace_lines:
        print(f"  {line}")
    return 0


def _print_sandbox_provenance(
    sandbox: dict[str, object] | None,
) -> None:
    if sandbox is None:
        print("sandbox_selected: unknown")
        print("sandbox_resolved: unknown")
        print("sandbox_engaged: unknown")
        print("sandbox_explicit_unsandboxed: unknown")
        return
    print(f"sandbox_selected: {sandbox.get('selected') or 'unknown'}")
    print(f"sandbox_resolved: {sandbox.get('resolved') or 'unknown'}")
    engaged = sandbox.get("engaged")
    if isinstance(engaged, bool):
        print(f"sandbox_engaged: {'yes' if engaged else 'no'}")
    else:
        print("sandbox_engaged: unknown")
    explicit = sandbox.get("explicit_unsandboxed")
    if isinstance(explicit, bool):
        print(
            "sandbox_explicit_unsandboxed: "
            f"{'yes' if explicit else 'no'}"
        )
    else:
        print("sandbox_explicit_unsandboxed: unknown")


def _print_image_evidence(artifact_dir: Path) -> None:
    evidence = image_evidence(artifact_dir)
    if not evidence:
        return
    total_bytes = sum(item.size_bytes for item in evidence)
    noun = "image" if len(evidence) == 1 else "images"
    print(f"attachments: {len(evidence)} {noun}, {total_bytes} bytes")
    for item in evidence:
        print(
            "attachment: "
            f"segment={item.segment_number} "
            f"image={item.image_number} "
            f"name={item.display_name} "
            f"media_type={item.media_type} "
            f"bytes={item.size_bytes} "
            f"sha256={item.sha256} "
            f"dimensions={item.width}x{item.height}"
        )


def _correction_state_for_record(record) -> CorrectionState:
    state = validate_correction_trace(record.artifact_path)
    validate_correction_owner(record, state)
    return state


def _print_correction_evidence(state: CorrectionState) -> None:
    if state.phase == "none":
        print("correction: none")
        return
    assert state.correction is not None
    correction = state.correction
    print(f"correction: {state.phase}")
    print(f"correction_id: {correction['correction_id']}")
    print(f"correction_sha256: {correction['text_sha256']}")
    print(f"correction_chars: {len(correction['text'])}")
    print(f"correction_preview: {_bounded_correction_preview(correction['text'])}")


def _bounded_correction_preview(text: str, *, max_chars: int = 160) -> str:
    rendered = json.dumps(text, ensure_ascii=False)
    if len(rendered) <= max_chars:
        return rendered
    preview = ""
    for character in text:
        candidate = json.dumps(preview + character + "...", ensure_ascii=False)
        if len(candidate) > max_chars:
            break
        preview += character
    return json.dumps(preview + "...", ensure_ascii=False)


def cmd_usage(args) -> int:
    """Render persisted usage without opening a writable store or provider."""
    try:
        store = SessionStore(read_only=True)
    except FileNotFoundError as exc:
        raise SystemExit("no assistant sessions found") from exc
    record = _resolve_session_record(
        store,
        args.session_id,
        selector="latest",
        allow_archived=True,
    )
    try:
        usage = aggregate_session_usage([record.artifact_path / ".trace.jsonl"])
    except UsageEvidenceError as exc:
        raise SystemExit(f"corrupt session usage evidence: {exc}") from exc
    print(f"session_id: {record.session_id}")
    print(f"session_ref: {record.short_id}")
    for line in render_session_usage(usage):
        print(line)
    return 0


def cmd_worktree_rm(args) -> int:
    """Remove only the owned worktree recorded for an assistant session."""
    store = SessionStore()
    record = _resolve_session_record(store, args.session_id, selector="latest")
    if store.get_session_lock(record.session_id) is not None:
        raise SystemExit("refusing to remove the worktree of a locked session")
    if not all(
        (
            record.worktree_path,
            record.worktree_branch,
            record.worktree_base_commit,
        )
    ):
        raise SystemExit(f"session has no retained worktree: {record.session_id}")
    try:
        inspected = inspect_session_worktree(Path(record.cwd), record.session_id)
        expected = (
            Path(str(record.worktree_path)).resolve(),
            str(record.worktree_branch),
            str(record.worktree_base_commit),
        )
        actual = (
            inspected.worktree_path.resolve(),
            inspected.branch,
            inspected.base_commit,
        )
        if actual != expected:
            raise WorktreeRuntimeError(
                "saved worktree identity does not match the owned Git worktree"
            )
        removed = remove_session_worktree(
            Path(record.cwd), record.session_id, force=bool(args.force)
        )
    except WorktreeRuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    store.clear_active_session(record.cwd, session_id=record.session_id)
    print(f"removed_worktree: {removed.worktree_path}")
    print(f"removed_branch: {removed.branch}")
    if removed.forced:
        print("force: uncommitted files and unmerged commits were discarded")
    return 0


def _save_approval_decision(artifact_dir: Path, approval: dict, decision: str) -> None:
    tool_name = str(approval.get("tool_name") or "")
    action_key = str(approval.get("action_key") or "")
    cmd = str(approval.get("cmd") or "")
    if not tool_name or not (action_key or cmd):
        return
    decisions = load_approval_decisions(artifact_dir)
    if action_key:
        decisions[action_key] = decision
    if cmd:
        decisions[f"{tool_name}:{cmd}"] = decision
    save_approval_decisions(artifact_dir, decisions)


def _resolve_prompt_input(args) -> tuple[str, str]:
    has_prompt_text = args.prompt_text is not None
    has_prompt_file = args.prompt_file is not None
    has_task = bool(args.task)
    provided = int(has_prompt_text) + int(has_prompt_file) + int(has_task)
    if provided == 0:
        if _is_interactive():
            return _prompt_multiline("Task"), "interactive"
        raise SystemExit(
            "provide a task as positional text, with --prompt-text, or with --prompt-file"
        )
    if provided > 1:
        raise SystemExit(
            "provide exactly one prompt source: positional task text, --prompt-text, or --prompt-file"
        )
    if has_prompt_file:
        return _read_prompt_file(args.prompt_file)
    if has_prompt_text:
        return _require_prompt_text(args.prompt_text, source="inline"), "inline"
    return (
        _require_prompt_text(" ".join(args.task).strip(), source="positional"),
        "inline-positional",
    )


def _resolve_resume_input(
    args,
) -> tuple[str | None, str | None, tuple[PendingImage, ...]]:
    paths = tuple(getattr(args, "image", ()) or ())
    prompt_text = getattr(args, "prompt_text", None)
    prompt_file = getattr(args, "prompt_file", None)
    if prompt_text is not None and prompt_file is not None:
        raise SystemExit("provide exactly one resume follow-up text source")
    if prompt_file is not None:
        resolved_text, prompt_source = _read_prompt_file(prompt_file)
    elif prompt_text is not None:
        resolved_text = _require_prompt_text(
            str(prompt_text), source="resume follow-up"
        )
        prompt_source = "inline"
    else:
        resolved_text = None
        prompt_source = None
    if paths and resolved_text is None:
        raise SystemExit("resume images require follow-up text")
    return resolved_text, prompt_source, read_image_inputs(paths)


def _read_prompt_file(prompt_file: Path) -> tuple[str, str]:
    if str(prompt_file) == "-":
        return _require_prompt_text(sys.stdin.read(), source="standard input"), "stdin"
    prompt_path = prompt_file.resolve()
    return (
        _require_prompt_text(prompt_path.read_text(), source=str(prompt_path)),
        str(prompt_path),
    )


def _require_prompt_text(text: str, *, source: str) -> str:
    if not text.strip():
        raise SystemExit(f"{source} prompt is empty")
    return text


def _record_resume_prompt_source(
    artifact_dir: Path,
    *,
    session_number: int,
    prompt_text: str,
    prompt_source: str,
) -> None:
    append_trace_event_fsync(
        artifact_dir / ".trace.jsonl",
        {
            "event": "operator_followup",
            "trace_schema_version": TRACE_SCHEMA_VERSION,
            "session_number": session_number,
            "prompt_source": prompt_source,
            "text_sha256": hashlib.sha256(prompt_text.encode("utf-8")).hexdigest(),
            "text_chars": len(prompt_text),
        },
    )


def _config_local_path() -> Path:
    return local_config_path()


def _load_assistant_config(config_paths: list[Path]):
    return load_config(
        user_config=config_paths,
        overrides={"runtime_mode": "assistant", "max_sessions": 1},
    )


def _effective_run_settings(args) -> tuple[list[Path], str]:
    """Return the selected package first and user overlays last."""
    treatment = bool(getattr(args, "treatment", True))
    package = _TREATMENT_CONFIG if treatment else _PLAIN_CONFIG
    context_mode = getattr(args, "context", None) or (
        "halflife" if treatment else "full"
    )
    return [package, *list(getattr(args, "config", []))], context_mode


def _needs_first_run_setup() -> bool:
    return not _config_local_path().exists()


def _is_interactive() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def _maybe_offer_first_run_setup(args) -> None:
    if not _needs_first_run_setup() or not _is_interactive():
        return
    if getattr(args, "provider", None) or getattr(args, "base_url", None) or getattr(args, "api_key_env", None):
        return
    if getattr(args, "config", None):
        return
    answer = input("No config.local.toml found. Run yuj setup now? [Y/n]: ").strip().lower()
    if answer in {"", "y", "yes"}:
        rc = cmd_setup(argparse.Namespace(
            provider=None,
            model=getattr(args, "model", None),
            base_url=None,
            api_key=None,
            api_key_env=None,
            auth=None,
            permission_preset=None,
            sandbox=None,
            sandbox_image=None,
            force=False,
        ))
        if rc == 0:
            # config.py bakes base+local layering into _LAYERED at import
            # time, so the command we're about to continue into would run
            # against the pre-setup snapshot and ignore the config.local.toml
            # setup just wrote. Re-exec so the run imports fresh config.
            os.execv(sys.executable, [sys.executable, *sys.argv])


def _prompt_choice(label: str, *, choices: list[str], default: str) -> str:
    if not sys.stdin.isatty():
        return default
    prompt = f"{label} ({'/'.join(choices)}) [{default}]: "
    while True:
        value = input(prompt).strip().lower() or default
        if value in choices:
            return value
        print(f"choose one of: {', '.join(choices)}")


def _prompt_required(label: str) -> str:
    while True:
        value = input(f"{label}: ").strip()
        if value:
            return value
        print(f"{label} is required")


def _prompt_multiline(label: str) -> str:
    print(f"{label} (finish with Ctrl-D on an empty line):")
    return _require_prompt_text(sys.stdin.read(), source=label.lower())


def _prompt_default(label: str, default: str) -> str:
    if not sys.stdin.isatty():
        return default
    return input(f"{label} [{default}]: ").strip() or default


def _api_key_ref_or_value(api_key_env: str | None, *, default: str | None = None) -> str:
    if api_key_env:
        return f"$ENV:{api_key_env}"
    if default is not None:
        if not sys.stdin.isatty():
            return default
        entered = getpass.getpass(f"API key [{default}]: ").strip()
        return entered or default
    return getpass.getpass("API key: ").strip() or _prompt_required("API key")


def _render_local_config(
    *,
    provider: str,
    base_url: str,
    api_key: str,
    model: str,
    permission_preset: str = "",
    sandbox: str = "",
    sandbox_image: str = "",
) -> str:
    rendered = (
        "# Generated by `yuj setup`. This file is gitignored.\n"
        "[server]\n"
        f'provider = "{_toml_escape(provider)}"\n'
        f'base_url = "{_toml_escape(base_url)}"\n'
        f'api_key = "{_toml_escape(api_key)}"\n'
        "\n"
        "[model]\n"
        f'name = "{_toml_escape(model)}"\n'
    )
    if permission_preset:
        rendered += (
            "\n[assistant]\n"
            f'permission_preset = "{_toml_escape(permission_preset)}"\n'
        )
    if sandbox:
        rendered += (
            "\n[sandbox]\n"
            f'backend = "{_toml_escape(sandbox)}"\n'
        )
        if sandbox_image:
            rendered += (
                "container_image = "
                f'"{_toml_escape(sandbox_image)}"\n'
            )
    return rendered


def _toml_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _auth_binding_for_args(
    args,
    *,
    store: CredentialStore | None = None,
) -> AuthBinding | None:
    store = store or CredentialStore()
    selected_provider = getattr(args, "provider", None)
    if selected_provider and selected_provider not in _MANAGED_PROVIDERS:
        return None
    active = store.active_binding()
    if selected_provider in _MANAGED_PROVIDERS:
        if active is None:
            raise CredentialMissingError(
                selected_provider,
                "no credential is selected; run "
                f"`yuj login --provider {selected_provider}`",
            )
        if active.provider != selected_provider:
            raise CredentialMissingError(
                selected_provider,
                "a different provider credential is selected; "
                "select this provider explicitly",
            )
    return active


def _transport_overrides_from_args(
    args,
    *,
    auth_binding: AuthBinding | None = None,
) -> dict:
    provider = getattr(args, "provider", None)
    base_url = getattr(args, "base_url", None)
    api_key_env = getattr(args, "api_key_env", None)
    thinking_level = getattr(args, "thinking", None)
    plan_mode = getattr(args, "plan_mode", None)
    edit_format = getattr(args, "edit_format", None)
    permission_preset = getattr(args, "permission_preset", None)
    if (
        not provider
        and not base_url
        and not api_key_env
        and not thinking_level
        and not plan_mode
        and not edit_format
        and not permission_preset
    ):
        return {}
    if provider == "custom" and not base_url:
        raise SystemExit("--provider custom requires --base-url")

    if provider in _MANAGED_PROVIDERS:
        binding = auth_binding or _auth_binding_for_args(args)
        assert binding is not None
        if api_key_env:
            raise SystemExit(
                f"--api-key-env cannot replace the selected {provider} credential"
            )
        spec = provider_spec(provider)
        expected_base = (
            spec.subscription_base_url
            if binding.auth_method == "subscription"
            else spec.api_key_base_url
        )
        if base_url and base_url.rstrip("/") != expected_base.rstrip("/"):
            raise SystemExit(
                f"--base-url cannot change the {provider} {binding.auth_method} endpoint"
            )
        overrides = {
            "provider": spec.core_provider,
            "base_url": expected_base,
            "api_key": "yuj-host-credential",
        }
    else:
        overrides = (
            dict(_PROVIDER_PRESETS.get(provider or "custom", {}))
            if provider or base_url or api_key_env
            else {}
        )

    if base_url and provider not in _MANAGED_PROVIDERS:
        overrides["base_url"] = base_url
    if api_key_env and provider not in _MANAGED_PROVIDERS:
        overrides["api_key"] = f"$ENV:{api_key_env}"
    if (
        provider
        and provider not in _MANAGED_PROVIDERS
        and provider != "local"
        and overrides.get("api_key", "").startswith("$ENV:")
    ):
        env_name = overrides["api_key"].split(":", 1)[1]
        if env_name not in os.environ:
            raise SystemExit(
                f"--provider {provider} expects {env_name}; set it or pass --api-key-env"
            )
    if thinking_level:
        overrides["thinking_level"] = thinking_level
    if plan_mode:
        overrides["plan_mode"] = plan_mode
    if edit_format:
        overrides["tools_edit_format"] = edit_format
    if permission_preset:
        overrides["assistant_permission_preset"] = permission_preset
    return overrides


def _persist_session_config_overlay(
    store: SessionStore,
    record,
    *,
    base_config_paths: list[Path],
    transport_overrides: dict,
):
    if not transport_overrides:
        return record
    overlay_path = record.artifact_path / "provider.toml"
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    overlay_path.write_text(_render_provider_overlay(transport_overrides))
    config_paths = [*base_config_paths, overlay_path]
    store.update_session_config_paths(record.session_id, config_paths)
    return store.get_session(record.session_id) or record


def _render_provider_overlay(overrides: dict) -> str:
    lines: list[str] = []
    server_lines: list[str] = []
    for key in ("provider", "base_url", "api_key"):
        if key in overrides and overrides[key] is not None:
            value = str(overrides[key]).replace("\\", "\\\\").replace('"', '\\"')
            server_lines.append(f'{key} = "{value}"')
    if server_lines:
        lines.extend(["[server]", *server_lines])
    if overrides.get("thinking_level") is not None:
        if lines:
            lines.append("")
        value = str(overrides["thinking_level"])
        lines.extend(["[model]", f'thinking_level = "{value}"'])
    if overrides.get("plan_mode") is not None:
        if lines:
            lines.append("")
        value = str(overrides["plan_mode"])
        lines.extend(["[loop]", f'plan_mode = "{value}"'])
    if overrides.get("tools_edit_format") is not None:
        if lines:
            lines.append("")
        value = str(overrides["tools_edit_format"])
        lines.extend(["[tools]", f'edit_format = "{value}"'])
    if overrides.get("assistant_permission_preset") is not None:
        if lines:
            lines.append("")
        value = str(overrides["assistant_permission_preset"])
        lines.extend([
            "[assistant]",
            f'permission_preset = "{value}"',
        ])
    return "\n".join(lines) + "\n"


def _print_session_start(
    record,
    *,
    action: str,
    served_models: list[str] | None = None,
) -> None:
    print(f"{action}: {record.session_id}")
    print(f"ref: {record.short_id}")
    print(f"cwd: {record.cwd}")
    print(f"model: {record.model}")
    print(f"context: {record.context_mode}")
    print(f"artifacts: {record.artifact_dir}")
    if served_models is not None:
        print(f"served_models: {', '.join(served_models)}")


def _print_session_result(record, success: bool, finish_reason: str | None) -> None:
    turns = session_turn_count(record.artifact_path)
    print(f"session_id: {record.session_id}")
    print(f"session_ref: {record.short_id}")
    print(f"status: {record.status}")
    print(f"cwd: {record.cwd}")
    print(f"artifacts: {record.artifact_dir}")
    print(f"model: {record.model}")
    if finish_reason:
        print(f"finish_reason: {finish_reason}")
    print(f"turns: {turns}")
    _print_run_compact_summary(record)
    if not success and record.status != "completed":
        print(f"resume with: {CLI_NAME} resume {record.short_id}")


def _print_run_compact_summary(record) -> None:
    summary = session_compact_summary(record.artifact_path)
    changed_files = list(summary.get("changed_files", []))
    last_test_cmd = str(summary.get("last_test_cmd") or "")
    last_test_result = str(summary.get("last_test_result") or "unknown")
    cache_metrics_present = bool(summary.get("cache_metrics_present"))
    cache_hit_ratio = summary.get("cache_hit_ratio")

    if not changed_files and not last_test_cmd and not cache_metrics_present:
        return

    print("summary:")
    if changed_files:
        shown = changed_files[:5]
        tail = " ..." if len(changed_files) > 5 else ""
        print(f"  changed_files: {', '.join(shown)}{tail}")
    else:
        print("  changed_files: none observed")

    if last_test_cmd:
        print(f"  last_test: {last_test_cmd}")
        print(f"  last_test_result: {last_test_result}")
    else:
        print("  last_test: none observed")
    if cache_metrics_present:
        if isinstance(cache_hit_ratio, (int, float)):
            print(f"  cache_hit_ratio: {float(cache_hit_ratio):.1%}")
        else:
            print("  cache_hit_ratio: unknown")


def _handle_keyboard_interrupt(store: SessionStore, record) -> int:
    mark_session_interrupted(record.artifact_path)
    store.update_session(
        record.session_id,
        status="paused",
        last_finish_reason="interrupted",
    )
    refreshed = store.get_session(record.session_id)
    final_record = refreshed or record
    print("interrupted: session paused cleanly")
    _print_session_result(final_record, False, "interrupted")
    return 130


def _resolve_model_or_exit(
    config_paths: list[Path],
    *,
    requested_model: str | None,
    config_overrides: dict | None = None,
    auth_binding: AuthBinding | None = None,
    auth_store: CredentialStore | None = None,
):
    try:
        return resolve_served_model(
            config_paths,
            requested_model=requested_model,
            config_overrides=config_overrides,
            auth_binding=auth_binding,
            auth_store=auth_store,
        )
    except Exception as exc:
        friendly = _friendly_model_resolution_error(exc)
        if friendly is None:
            raise
        raise SystemExit(friendly) from exc


def _resolve_smoke_model_or_exit(
    config_paths: list[Path],
    *,
    requested_model: str | None,
    config_overrides: dict | None = None,
    auth_binding: AuthBinding | None = None,
    auth_store: CredentialStore | None = None,
):
    try:
        return resolve_smoke_model(
            config_paths,
            requested_model=requested_model,
            config_overrides=config_overrides,
            auth_binding=auth_binding,
            auth_store=auth_store,
        )
    except Exception as exc:
        friendly = _friendly_model_resolution_error(exc)
        if friendly is None:
            raise
        raise SystemExit(friendly) from exc


def _friendly_model_resolution_error(exc: Exception) -> str | None:
    base_url = get_server_base_url()
    if isinstance(exc, KeyError) and "environment variable" in str(exc):
        return str(exc)
    if exc.__class__.__name__ in {"APIConnectionError", "APITimeoutError"}:
        return (
            f"could not reach the local model server at {base_url} while resolving /v1/models; "
            "start the server or fix the base_url setting"
        )
    if isinstance(exc, RuntimeError):
        return f"could not resolve a served model from {base_url}: {exc}"
    return None


@contextlib.contextmanager
def _session_lock(store: SessionStore, record):
    try:
        store.acquire_session_lock(record.session_id)
    except SessionLockedError as exc:
        raise SystemExit(
            f"session {record.session_id} is already locked by "
            f"pid {exc.lock.owner_pid} on {exc.lock.owner_host} since {exc.lock.acquired_at}"
        ) from exc
    try:
        yield
    finally:
        store.release_session_lock(record.session_id)


def _resolve_session_record(
    store: SessionStore,
    session_ref: str,
    *,
    selector: str,
    allow_archived: bool = False,
):
    if session_ref.lower() not in _LATEST_SESSION_TOKENS:
        try:
            record = store.resolve_session_ref(session_ref)
        except (AmbiguousSessionRefError, SessionPurgeInProgressError) as exc:
            raise SystemExit(str(exc)) from exc
        if record is None:
            raise SystemExit(f"unknown session: {session_ref}")
        if record.archived_at is not None and not allow_archived:
            raise SystemExit(
                "session is archived; run "
                f"{CLI_NAME} unarchive {record.short_id} first"
            )
        return record

    current_cwd = Path.cwd().resolve()
    sessions = store.list_sessions(limit=200)
    if not sessions:
        raise SystemExit("no assistant sessions found")

    active_record = store.get_active_session(current_cwd)
    if selector == "latest" and active_record is not None:
        return active_record
    if selector == "resumable" and active_record is not None and active_record.status != "completed":
        return active_record
    if selector == "pending_approval" and active_record is not None:
        approval = load_approval_request(active_record.artifact_path)
        if approval is not None and approval.get("status") == "pending":
            return active_record

    current_cwd_str = str(current_cwd)
    scoped = [record for record in sessions if record.cwd == current_cwd_str]

    if selector == "resumable":
        for records in (scoped, sessions):
            resumable = [record for record in records if record.status != "completed"]
            if resumable:
                return resumable[0]
        raise SystemExit("no resumable assistant session found")

    if selector == "pending_approval":
        for records in (scoped, sessions):
            for record in records:
                approval = load_approval_request(record.artifact_path)
                if approval is not None and approval.get("status") == "pending":
                    return record
        raise SystemExit("no pending approval request found")

    candidates = scoped or sessions
    return candidates[0]


if __name__ == "__main__":
    sys.exit(main())
