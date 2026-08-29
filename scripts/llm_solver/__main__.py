"""CLI entry point — python -m scripts.llm_solver <run_dir> [options]."""
import argparse
import hashlib
import logging
import os
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from .config import (
    load_config,
    PROJECT_ROOT,
    require_runtime_mode,
    resolve_transformation_context_mode,
)
from ._shared.edit_formats import EDIT_FORMATS
from .harness import collect_pending, solve_task
from .harness.context_strategies import (
    list_context_modes,
    resolve_context_class,
)
from .harness._loop.model_role_runtime import (
    build_model_role_runtime,
    validate_model_role_profiles,
)
from .harness.worktree_runtime import (
    create_session_worktree,
    inspect_session_worktree,
)
from .harness.sandbox.policy import (
    SandboxResolutionError,
    bind_sandbox_resolution,
    preflight_sandbox,
)
from .models import resolve_model
from .server import LlamaClient, load_profile
from .server.request_controls import THINKING_LEVELS

# Helpers extracted to keep this file under the 500-line cap.
from ._main_helpers import (
    _build_run_metadata,
    _harden_process,
    _write_server_metadata,
    _write_session_json,
)

log = logging.getLogger(__name__)


# Pre-main hardening fires at import time, before any subprocess can be
# spawned by the importing process. Equivalent to a Rust ctor.
_harden_process()


def _model_log_tag(value: str | None) -> str:
    """Return a bounded filename component without changing the wire model ID."""
    raw = str(value or "default")
    leaf = raw.rsplit("/", 1)[-1] or "model"
    safe = "".join(
        char if char.isalnum() or char in "._-" else "-"
        for char in leaf
    ).strip(".-_") or "model"
    if safe == raw and len(safe) <= 80:
        return safe
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
    return f"{safe[:64]}-{digest}"


def _build_client(cfg, profile):
    if os.environ.get("YUJ_CODEX_HEADLESS") == "1":
        from .analysis.codex_yuj_client import CodexHeadlessYujClient

        return CodexHeadlessYujClient.from_env(cfg, profile=profile)
    return LlamaClient(cfg, profile=profile)


def _prepare_task_worktree(
    cfg,
    *,
    run_dir: Path,
    source_cwd: Path,
    resume: bool,
    multi_task: bool,
):
    """Create/reuse a stable direct-run worktree before fixing sandbox cwd."""
    mode = str(getattr(cfg, "runtime_worktree", "off") or "off").strip()
    if mode == "off":
        return None
    source = Path(source_cwd).resolve()
    identity = f"{Path(run_dir).resolve()}\0{source}"
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
    run_id = f"direct-{digest}"
    effective_mode = mode
    if multi_task and mode != "auto":
        effective_mode = f"{mode}-{digest[:8]}"
    if resume:
        inspect_session_worktree(source, run_id)
    return create_session_worktree(
        source,
        mode=effective_mode,
        run_id=run_id,
        reuse=resume,
    )


def main(argv: list[str] | None = None) -> int:
    cli_argv = list(sys.argv[1:] if argv is None else argv)
    context_was_explicit = any(
        token == "--context" or token.startswith("--context=")
        for token in cli_argv
    )
    parser = argparse.ArgumentParser(
        description="Run fixed coding measurements through the Yuj harness"
    )
    parser.add_argument(
        "run_dir", type=Path,
        help="directory for run-level records; multi-task mode reads RUN_DIR/repos",
    )
    parser.add_argument("--model", "-m", help="model ID or known short name")
    parser.add_argument(
        "--thinking", choices=THINKING_LEVELS,
        help="per-request reasoning effort",
    )
    parser.add_argument(
        "--plan-mode", choices=("off", "required"),
        help="require an explicit .solver/plan.md before implementation",
    )
    parser.add_argument(
        "--edit-format", choices=EDIT_FORMATS,
        help="override the selected model profile's edit dialect",
    )
    parser.add_argument("--port", "-p", type=int, help="use this model-server port")
    parser.add_argument("--config", "-c", type=Path, action="append", default=[],
                        help="TOML settings file; repeat to apply more files; later values win")
    parser.add_argument("--max-sessions", type=int,
                        help="largest number of solver sessions for each task")
    parser.add_argument("--task", type=Path, help="run this one task repository")
    parser.add_argument("--prompt-file", type=Path, default=None,
                        help="single-task mode: read the task prompt from this file")
    parser.add_argument("--prompt-text", default=None,
                        help="single-task mode: use this text as the task prompt")
    parser.add_argument("--system-prompt", type=Path, default=None,
                        help="file to add before the normal system prompt")
    _context_modes = list_context_modes()
    parser.add_argument(
        "--context",
        choices=_context_modes,
        default="full",
        help="context mode (default: full)",
    )
    parser.add_argument("--prompt-addendum", default=None,
                        help="text to add to the task prompt")
    parser.add_argument("--variant-name", default=None,
                        help="label to save with this measurement")
    parser.add_argument("--tool-desc", default=None, choices=["minimal"],
                        help="model-tool description mode; the public release supports minimal")
    parser.add_argument("--rumination-threshold", type=int, default=None,
                        help="no-change prompt percentage; requires rumination_enabled")
    parser.add_argument("--duplicate-abort", type=int, default=None,
                        help="repeated-call limit; requires duplicate_guard_enabled")
    parser.add_argument("--require-intent", action="store_true", default=None,
                        help="reject a tool call that has no assistant text")
    parser.add_argument("--transcript-dir", type=Path, default=None,
                        help="single-task mode: save model-message records here")
    parser.add_argument("--savings-dir", type=Path, default=None,
                        help="single-task mode: save context-change records here")
    parser.add_argument("--resume", type=Path, default=None,
                        help="resume this transcript without adding a message")
    parser.add_argument("--resume-message-file", type=Path, default=None,
                        help="with --resume, add this explicit user message")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="write run metadata, print settings and tasks, then exit without a task run",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="print debug-level process information",
    )
    parser.add_argument("--replay-from", type=Path, default=None,
                        help="replay mode: read this source run directory")
    parser.add_argument("--replay-stop-turn", type=int, default=0,
                        help="for a source with one solver session, stop after trace turn N "
                             "(0 = full replay)")
    parser.add_argument("--replay-allow-divergence", action="store_true",
                        help="record a divergence and continue instead of stopping")
    parser.add_argument("--replay-continue-live", action="store_true",
                        help="after the stop turn, request a live-model handover")
    parser.add_argument("--replay-overlay", type=Path, default=None,
                        help="TOML settings file to apply at live handover")
    parser.add_argument("--replay-watch-turns", type=int, default=0,
                        help="pass the intended live-turn limit (0 = no change); "
                             "the current loop does not enforce it")
    parser.add_argument("--replay-extra-config", type=Path, action="append", default=[],
                        help="measurement-only settings appended after source settings; "
                             "repeatable; request parity is not currently checked")
    args = parser.parse_args(cli_argv)
    if args.prompt_file is not None and args.prompt_text is not None:
        parser.error("--prompt-file and --prompt-text are mutually exclusive")
    if (args.prompt_file is not None or args.prompt_text is not None) and args.task is None:
        parser.error("--prompt-file/--prompt-text require --task")
    if args.resume is not None and args.task is None:
        parser.error("--resume requires --task (single-task mode)")
    if args.resume_message_file is not None and args.resume is None:
        parser.error("--resume-message-file requires --resume")

    # Logging — stderr + file in run_dir
    level = logging.DEBUG if args.verbose else logging.INFO
    log_fmt = "[%(asctime)s] %(levelname)s %(name)s: %(message)s"
    log_datefmt = "%H:%M:%S"
    logging.basicConfig(level=level, format=log_fmt, datefmt=log_datefmt)

    run_dir = args.run_dir.resolve()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_tag = _model_log_tag(args.model)
    log_path = run_dir / f"harness_{model_tag}_{ts}.log"
    run_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_path)
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter(log_fmt, datefmt=log_datefmt))
    logging.getLogger().addHandler(fh)
    log.info("Log file: %s", log_path)

    # Replay config adoption (docs/replay_mode_spec.md "Config parity"):
    # the recording's session.json supplies model + config layers; user
    # --config/--model are refused in replay mode rather than merged.
    _replay_prov = None
    if args.replay_from is not None:
        from .server.replay_client import load_replay_provenance
        if (
            args.config
            or args.model
            or args.thinking is not None
            or args.plan_mode is not None
            or args.edit_format is not None
        ):
            parser.error("--replay-from adopts the recording's config; "
                         "--config/--model/--thinking/--plan-mode/"
                         "--edit-format are not "
                         "allowed in replay mode")
        try:
            _replay_prov = load_replay_provenance(args.replay_from)
        except (OSError, ValueError) as e:
            log.error("replay refused: cannot load source provenance: %s", e)
            return 2
        args.config = [Path(p) for p in _replay_prov["config_paths"]] \
            + [Path(p) for p in (args.replay_extra_config or [])]
        args.model = _replay_prov["model"]
        src_mode = _replay_prov.get("context_mode") or ""
        if src_mode:
            if args.context == parser.get_default("context"):
                # flag not explicitly set: adopt the recording's mode
                args.context = src_mode
                log.info("replay adopts context mode %r from the recording", src_mode)
            elif args.context != src_mode:
                log.error("replay refused: source context mode %r != requested %r",
                          src_mode, args.context)
                return 2

    # Build overrides from CLI flags
    overrides: dict = {}
    if args.model:
        overrides["model"] = resolve_model(args.model)
    if args.thinking is not None:
        overrides["thinking_level"] = args.thinking
    if args.plan_mode is not None:
        overrides["plan_mode"] = args.plan_mode
    if args.edit_format is not None:
        overrides["tools_edit_format"] = args.edit_format
    if args.port:
        # Reuse scheme+host from [server] base_url; only the port changes.
        from urllib.parse import urlparse, urlunparse
        from .config import get_server_base_url
        parsed = urlparse(get_server_base_url())
        netloc = f"{parsed.hostname or 'localhost'}:{args.port}"
        overrides["base_url"] = urlunparse(parsed._replace(netloc=netloc))
    if args.max_sessions:
        overrides["max_sessions"] = args.max_sessions
    if args.rumination_threshold is not None:
        overrides["rumination_nudge_threshold"] = args.rumination_threshold
    if args.duplicate_abort is not None:
        overrides["duplicate_abort"] = args.duplicate_abort
    if args.require_intent is not None:
        overrides["require_intent"] = args.require_intent
    if args.prompt_addendum is not None:
        overrides["prompt_addendum"] = args.prompt_addendum
    if args.variant_name is not None:
        overrides["variant_name"] = args.variant_name
    if args.tool_desc is not None:
        overrides["tool_desc"] = args.tool_desc

    cfg = load_config(user_config=args.config, overrides=overrides)
    require_runtime_mode(cfg, expected="measurement", caller="scripts.llm_solver")
    if bool(getattr(cfg, "transformations_explicit", False)):
        try:
            args.context = resolve_transformation_context_mode(
                cfg,
                args.context,
                requested_explicitly=(
                    context_was_explicit or args.replay_from is not None
                ),
            )
        except ValueError as exc:
            parser.error(
                str(exc).replace("context choice", "--context")
            )
    else:
        cfg = replace(cfg, halflife_context=args.context == "halflife")
    try:
        sandbox_resolution = preflight_sandbox(cfg)
    except SandboxResolutionError as exc:
        log.error("Sandbox startup failed before model contact: %s", exc)
        return 2
    cfg = bind_sandbox_resolution(cfg, sandbox_resolution)
    started_at = datetime.now(timezone.utc).isoformat()

    # Echo resolved config for every run (not just --dry-run): reproducibility.
    log.info(
        "Config: model=%s ctx=%d max_turns=%d max_sessions=%d tool_desc=%s "
        "edit_format=%s variant=%s sandbox_selected=%s sandbox_resolved=%s",
        cfg.model, cfg.context_size, cfg.max_turns, cfg.max_sessions,
        cfg.tool_desc, cfg.tools_edit_format or "profile",
        cfg.variant_name or "(none)",
        sandbox_resolution.selected,
        sandbox_resolution.resolved,
    )

    # Capture a preliminary run envelope before dry-run can return. Real
    # executions rewrite this after server context/max_tokens derivation and
    # server_meta.json capture, then thread the same envelope into each
    # task's metrics.json.
    try:
        run_metadata = _build_run_metadata(
            run_dir=run_dir,
            cfg=cfg,
            args=args,
            overrides=overrides,
            started_at=started_at,
        )
        _write_session_json(run_dir, run_metadata)
    except Exception as e:
        log.warning("session.json capture failed: %s", e)
        run_metadata = {}

    # Load model profile
    profiles_dir = PROJECT_ROOT / "profiles"
    profile = None
    if profiles_dir.is_dir():
        try:
            profile_key = cfg.profile_name or cfg.model
            profile = load_profile(profile_key, profiles_dir)
            log.info("Loaded profile: %s (inherits=%s)", profile.name, profile.inherits)
        except FileNotFoundError:
            log.info("No profile found for '%s', using legacy mode", profile_key)
    validate_model_role_profiles(
        cfg=cfg, main_profile=profile, profiles_dir=profiles_dir,
    )

    try:
        run_metadata = _build_run_metadata(
            run_dir=run_dir,
            cfg=cfg,
            args=args,
            overrides=overrides,
            started_at=started_at,
            profile_loaded=profile.name if profile else None,
        )
        _write_session_json(run_dir, run_metadata)
    except Exception as e:
        log.warning("session.json profile update failed: %s", e)

    if args.dry_run:
        print(f"Config: {cfg}")
        print(f"Profile: {profile.name if profile else 'none (legacy)'}")
        print(f"System prompt: {args.system_prompt or '(default)'}")
        print(f"Context: {args.context}")
        if args.task:
            print(f"Task: {args.task}")
        else:
            pending = collect_pending(args.run_dir)
            print(f"Pending: {len(pending)} tasks")
            for p in pending:
                print(f"  {p.name}")
        return 0

    # Wire server layer
    if args.replay_from is not None:
        if args.task is None:
            # replay re-executes recorded commands against a task working
            # copy; it is single-task by nature. The caller provisions a
            # FRESH copy of the same task the recording ran against (the
            # bench tooling already does this per cell).
            log.error("--replay-from requires --task <fresh task repo dir> "
                      "(replay is single-task; see docs/replay_mode_spec.md)")
            return 2
        from .server.replay_client import ReplayClient, resolve_replay_source
        # The adoption block above has already refused --model/--config
        # and loaded the source model, context mode, and config paths.
        transcript, source_trace, _mode = resolve_replay_source(args.replay_from)
        client = ReplayClient(transcript, stop_turn=args.replay_stop_turn,
                              strict_fidelity=not args.replay_allow_divergence,
                              source_trace_path=source_trace)
        if args.replay_continue_live and client.has_recorded_clarification:
            log.error(
                "--replay-continue-live is unavailable for a source with a "
                "recorded clarification; replay must remain offline"
            )
            return 2
        if args.replay_continue_live:
            from .harness._loop.replay_handover import arm
            arm(client,
                live_client_factory=lambda: _build_client(cfg, profile),
                overlay_path=str(args.replay_overlay) if args.replay_overlay else "",
                watch_turns=args.replay_watch_turns)
        log.info("REPLAY MODE: source=%s stop_turn=%s strict=%s continue_live=%s",
                 transcript, args.replay_stop_turn or "(end)",
                 not args.replay_allow_divergence, args.replay_continue_live)
    else:
        client = _build_client(cfg, profile)
    if hasattr(client, "set_session_id"):
        client.set_session_id(
            str(run_metadata.get("session_id") or f"{run_dir.resolve()}:{started_at}")
        )
    server_metadata_path: Path | None = None
    server_metadata_sha256: str | None = None
    try:
        server_metadata_path, server_metadata_sha256 = _write_server_metadata(
            run_dir, client,
        )
        if server_metadata_path is not None:
            log.info(
                "Server metadata: %s sha256=%s",
                server_metadata_path,
                server_metadata_sha256,
            )
    except Exception as e:
        log.warning("server_meta.json capture failed: %s", e)

    # Query server for effective context size
    server_ctx = client.query_server_context()
    if server_ctx:
        effective_ctx = min(cfg.context_size, server_ctx) if cfg.context_size > 0 else server_ctx
        if effective_ctx != cfg.context_size:
            log.info(
                "Context: config=%d, server=%d → effective=%d",
                cfg.context_size, server_ctx, effective_ctx,
            )
            cfg = replace(cfg, context_size=effective_ctx)
        else:
            log.info("Context: %d (config matches server)", effective_ctx)
    else:
        log.warning("Could not query server context — using config value %d", cfg.context_size)

    # Derive max_tokens from effective context size. The hardcoded
    # max_tokens=16384 was correct only at ctx=65,536 (16384/65536≈0.25);
    # any other server ctx gave the wrong shape (32k server → still 16k
    # generation, eating half the context). max_tokens_fraction=0.25 by
    # default preserves the prior generation budget at the canonical ctx
    # while scaling correctly elsewhere.
    derived_max_tokens = int(cfg.context_size * cfg.max_tokens_fraction)
    if derived_max_tokens != cfg.max_tokens:
        log.info(
            "max_tokens from ctx=%d × %.2f = %d (was %d)",
            cfg.context_size, cfg.max_tokens_fraction,
            derived_max_tokens, cfg.max_tokens,
        )
        cfg = replace(cfg, max_tokens=derived_max_tokens)

    # Derive char budgets from effective context size.
    # Rolling window + single tool result must fit within ~45% of the
    # token budget (rest goes to system prompt, state.json, task prompt,
    # generation headroom). At ~4 chars/token this gives the char caps.
    _ROLLING_WINDOW_RATIO = 0.45   # fraction of token budget for rolling window
    _MAX_OUTPUT_RATIO = 0.40       # fraction of token budget for single tool result
    _CHARS_PER_TOKEN = 4
    token_budget = int(cfg.context_size * cfg.context_fill_ratio)
    derived_recent = int(token_budget * _ROLLING_WINDOW_RATIO * _CHARS_PER_TOKEN)
    derived_output = int(token_budget * _MAX_OUTPUT_RATIO * _CHARS_PER_TOKEN)
    if derived_recent != cfg.recent_tool_results_chars or derived_output != cfg.max_output_chars:
        log.info(
            "Char budgets from ctx=%d (%.0f%% fill): "
            "recent_tool_results %d→%d, max_output %d→%d",
            cfg.context_size, cfg.context_fill_ratio * 100,
            cfg.recent_tool_results_chars, derived_recent,
            cfg.max_output_chars, derived_output,
        )
        cfg = replace(cfg, recent_tool_results_chars=derived_recent, max_output_chars=derived_output)

    # Re-bind the client's cfg reference. dataclasses.replace builds a NEW
    # Config object each call, so the LlamaClient instance constructed at
    # line 144 is still holding the ORIGINAL cfg with the placeholder
    # max_tokens=0 from _extract_config_fields. Without this re-bind, a
    # request can still use the placeholder after Yuj derives the real value.
    client.cfg = cfg
    if args.replay_from is None:
        build_model_role_runtime(
            cfg=cfg,
            main_client=client,
            profiles_dir=profiles_dir,
            client_factory=lambda role_cfg, role_profile: LlamaClient(
                role_cfg, profile=role_profile
            ),
        )

    try:
        run_metadata = _build_run_metadata(
            run_dir=run_dir,
            cfg=cfg,
            args=args,
            overrides=overrides,
            started_at=started_at,
            profile_loaded=profile.name if profile else None,
            server_metadata_path=server_metadata_path,
            server_metadata_sha256=server_metadata_sha256,
        )
        _write_session_json(run_dir, run_metadata)
    except Exception as e:
        log.warning("session.json final update failed: %s", e)

    # Context strategy (single-source registry in context_strategies).
    context_class = resolve_context_class(args.context)

    # Single task mode
    if args.task:
        initial_prompt = None
        if args.prompt_file is not None:
            initial_prompt = args.prompt_file.read_text()
        elif args.prompt_text is not None:
            initial_prompt = args.prompt_text
        # Resume mode: --resume alone restores a balanced request boundary
        # without adding a message. --resume-message-file selects the older
        # explicit-handoff path and supplies its next user message.
        if args.resume is not None:
            if args.resume_message_file is not None:
                initial_prompt = args.resume_message_file.read_text()
                if not initial_prompt.strip():
                    parser.error(
                        "--resume-message-file is empty; omit it for "
                        "transparent resume"
                    )
        worktree_info = _prepare_task_worktree(
            cfg,
            run_dir=run_dir,
            source_cwd=args.task,
            resume=args.resume is not None,
            multi_task=False,
        )
        task_cwd = (
            worktree_info.session_cwd
            if worktree_info is not None else args.task
        )
        ok = solve_task(
            task_cwd, cfg, client,
            system_prompt_file=args.system_prompt,
            context_class=context_class,
            profile_path=(profile.profile_dir / "profile.toml")
                         if profile and profile.profile_dir else None,
            initial_prompt=initial_prompt,
            transcript_dir=args.transcript_dir,
            savings_dir=args.savings_dir,
            resume_path=args.resume,
            transparent_resume=(
                args.resume is not None and args.resume_message_file is None
            ),
            run_metadata=run_metadata,
            worktree_info=worktree_info,
        )
        return 0 if ok else 1

    # Multi-task mode
    pending = collect_pending(run_dir)
    if not pending:
        print("No pending tasks.")
        return 0

    print(f"Solving {len(pending)} tasks (model={cfg.model})")
    results = {}
    for repo_dir in pending:
        worktree_info = _prepare_task_worktree(
            cfg,
            run_dir=run_dir,
            source_cwd=repo_dir,
            resume=False,
            multi_task=True,
        )
        task_cwd = (
            worktree_info.session_cwd
            if worktree_info is not None else repo_dir
        )
        ok = solve_task(
            task_cwd, cfg, client,
            system_prompt_file=args.system_prompt,
            context_class=context_class,
            profile_path=(profile.profile_dir / "profile.toml")
                         if profile and profile.profile_dir else None,
            run_metadata=run_metadata,
            worktree_info=worktree_info,
        )
        results[repo_dir.name] = ok
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {repo_dir.name}")

    passed = sum(1 for v in results.values() if v)
    print(f"\n{passed}/{len(results)} tasks completed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
