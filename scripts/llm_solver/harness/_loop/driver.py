"""solve_task — outer loop driver.

Coordinates the per-task lifecycle: prompt + provenance load, sandbox /
runtime envelope, the for-session loop (pretest → context → session →
aggregate → terminate), and final run-metrics write.

Setup helpers own setup state; trace events remain here so their order is
visible at one read-site. Names patched by tests are late-bound through the
public ``loop`` module rather than imported directly.
"""
from __future__ import annotations
import logging
import os
import time
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING

from ...config import Config
from ...server.request_controls import CacheUsageAccumulator
from ..context import ContextManager
from ..solver import write_checkpoint, write_run_metrics
from ..state_writer import write_state_from_trace
from . import (
    _normalize_repo_timestamps,
    _pretest_is_green,
    build_resume_prompt,
    run_pretest,
)
from ._driver_setup import (
    compute_runtime_envelope_fields,
    load_session_injections, load_system_prompt_and_provenance,
    load_transforms_and_estimator,
    resolve_task_format, resolve_run_paths,
    setup_run_outputs, thinking_trace_fields,
)
from ._session_setup import build_context_manager, inject_resume_messages
from .handoff_integration import apply_pending_handoff, maybe_prepare_boundary_handoff
from .interrupted_turn import RecoveryPlan, recover_interrupted_trace
from . import model_role_runtime
from .resume import _load_trace_events, _next_session_number, build_resume_prompt_from_trace
from .trace_schema import emit_trace_event as _emit_trace_event

if TYPE_CHECKING:
    from ..loop import SessionResult, TaskSpec
log = logging.getLogger(__name__)

def solve_task(
    repo_dir: Path, cfg: Config, client,
    system_prompt_file: Path | None = None,
    context_class: type[ContextManager] | None = None,
    profile_path: Path | None = None,
    initial_prompt: str | None = None,
    task_spec: "TaskSpec | None" = None,
    transcript_dir: Path | None = None,
    savings_dir: Path | None = None,
    resume_path: Path | None = None,
    run_metadata: dict | None = None,
    artifacts_dir: Path | None = None,
    resume_from_artifacts: bool = False,
    worktree_info=None,
) -> bool:
    """Outer loop: run sessions until done or max_sessions exhausted.

    client: any object with chat() and build_assistant_message() methods
           (e.g. LlamaClient from server layer). Injected for swappability.
    system_prompt_file: optional file whose content is prepended to the system prompt.
    context_class: if provided, instantiate this instead of default SolverStateContext.
    profile_path: optional path to profile.toml for provenance hashing.
    """
    # Late-bind names that tests patch on the public ``loop`` module.
    # See module docstring; same pattern as run_step.py.
    from .. import loop as _loop_mod
    Session = _loop_mod.Session
    SessionResult = _loop_mod.SessionResult
    _auto_commit = _loop_mod._auto_commit
    log = _loop_mod.log  # so test patches on loop.log intercept these emits

    work_dir, artifact_dir, trace_path = resolve_run_paths(repo_dir, artifacts_dir)
    checkpoint_store = None
    if getattr(cfg, "tools_file_checkpoints_enabled", False):
        from ..workspace_checkpoints import (
            WorkspaceCheckpointStore,
            default_shadow_dir,
        )
        candidate = artifact_dir / ".shadow_git"
        try:
            candidate.resolve().relative_to(work_dir.resolve())
        except ValueError:
            shadow_dir = candidate
        else:
            shadow_dir = default_shadow_dir(work_dir)
        checkpoint_store = WorkspaceCheckpointStore(
            work_dir,
            shadow_dir=shadow_dir,
            excludes=getattr(cfg, "tools_file_checkpoints_exclude", ()),
        )
        cfg = replace(
            cfg,
            unreadable_paths=tuple(cfg.unreadable_paths)
            + checkpoint_store.sandbox_unreadable_paths,
        )
    if getattr(cfg, "rewind_enabled", False):
        if checkpoint_store is None:
            raise RuntimeError(
                "conversation rewind requires workspace checkpoints"
            )
        from ..turn_snapshots import rewind_snapshot_dir
        rewind_dir = rewind_snapshot_dir(work_dir, artifact_dir)
        rewind_dir.mkdir(parents=True, exist_ok=True)
        rewind_dir.chmod(0o700)
        cfg = replace(
            cfg,
            unreadable_paths=tuple(cfg.unreadable_paths) + (str(rewind_dir),),
        )
    from ..tools import _effective_command_environment
    resolved_env, allow_login_shell = _effective_command_environment(cfg)
    # One immutable snapshot is shared by every session and command surface;
    # later host-process environment mutations cannot change this run.
    effective_env = MappingProxyType(resolved_env)
    from ..sandbox.ignore_policy import load_ignore_policy
    ignore_policy = load_ignore_policy(
        work_dir,
        enabled=getattr(cfg, "state_ignore_file_enabled", True),
        file_names=getattr(
            cfg, "state_ignore_file_names", (".yujignore",)
        ),
    )
    # Context discovery happens before the first model call. Give it the same
    # startup view as the tools while retaining the immutable compiled policy
    # for dynamic path decisions later in the run.
    prompt_unreadable_paths = tuple(dict.fromkeys((
        *tuple(cfg.unreadable_paths),
        *ignore_policy.sandbox_unreadable_paths(),
    )))
    prompt_file = artifact_dir / "prompt.txt"
    pretest_script = task_spec.pretest_script if task_spec is not None else None
    task_prompt = initial_prompt
    if task_prompt is None and task_spec is not None:
        task_prompt = task_spec.prompt_text
    if task_prompt is None:
        if not prompt_file.exists():
            log.error("No prompt.txt in %s", artifact_dir)
            return False
        task_prompt = prompt_file.read_text()

    start_time = time.time()
    (system_prompt, provenance, context_contract,
     prompt_metadata) = load_system_prompt_and_provenance(
        cfg, client, work_dir, system_prompt_file, profile_path, run_metadata,
        context_class,
        unreadable_paths=prompt_unreadable_paths,
    )
    resolved_injections, injection_import_tree = load_session_injections(
        cfg, work_dir, unreadable_paths=prompt_unreadable_paths,
    )
    if injection_import_tree:
        prompt_metadata = replace(
            prompt_metadata,
            prompt_import_tree=(
                *prompt_metadata.prompt_import_tree,
                *injection_import_tree,
            ),
        )
    prev_session: "Session | None" = None
    prev_result: "SessionResult | None" = None
    pending_handoff = None
    agg_prompt = 0
    agg_completion = 0
    agg_turns = 0
    agg_done_blocked = 0
    # Cross-session totals for the per-session counters. Previously only
    # done_blocked_count surfaced; every other counter died with the session.
    agg_intent_block = 0
    agg_gate_block = 0
    agg_contract_block = 0
    agg_same_class_error = 0
    agg_commit_violation = 0
    agg_mutation_repeat = 0
    agg_verify_repeat = 0
    # Count sessions that start with a corrupt trace mirror.
    agg_trace_corrupt = 0
    agg_length_continuations = 0
    cache_usage = CacheUsageAccumulator()
    role_usage = model_role_runtime.role_token_ledger(client)
    done_loop_aborted = False
    sessions_used = 0
    success = False
    task_description = task_prompt
    # Mechanical state.json is rebuilt from the trace at session boundaries.
    state_json_path = artifact_dir / ".solver" / "state.json"
    state_path: Path | None = state_json_path if cfg.state_writer_enabled else None
    # Multilingual: resolve analysis_task_format="auto" to the repo's
    # actual runner before any consumer (verification detection, output
    # parsing, detector fields) reads it. No-op when a concrete format
    # is pinned. Must precede load_transforms_and_estimator.
    cfg = resolve_task_format(cfg, work_dir)
    if "cfg" in getattr(client, "__dict__", {}):
        client.cfg = cfg
    setup_run_outputs(
        cfg, client, work_dir, artifact_dir,
        assistant_artifacts=artifacts_dir is not None,
        savings_dir=savings_dir,
        transcript_dir=transcript_dir,
        system_prompt=system_prompt,
        system_prompt_file=system_prompt_file,
        prompt_metadata=prompt_metadata,
    )

    (output_control, universal_rewrites, forbidden_rules, redirect_rules,
     redactions, output_parser, token_estimator) = load_transforms_and_estimator(
         cfg, client, work_dir
     )

    # YUJ_HOLD_UNTIL — explicit pre-launch gate (same env-var contract style
    # as YUJ_CONTAINER). An orchestrator that pipelines tasks can start this
    # solver early, let ALL startup above (imports, config, transforms,
    # tokenizer) complete while the previous task still owns the GPU, and
    # release it by creating the signal file the moment the GPU frees. Every
    # step above this line is CPU-only; the first server request happens
    # after it. Unset => no behavior change. A signal that never arrives is
    # a loud RuntimeError after YUJ_HOLD_TIMEOUT_S (default 1800), not a hang.
    _hold = os.environ.get("YUJ_HOLD_UNTIL", "")
    if _hold:
        _deadline = time.time() + float(os.environ.get("YUJ_HOLD_TIMEOUT_S", "1800"))
        log.info("hold_until: startup complete, waiting for signal %s", _hold)
        while not os.path.exists(_hold):
            if time.time() > _deadline:
                raise RuntimeError(f"YUJ_HOLD_UNTIL signal never arrived: {_hold}")
            time.sleep(0.05)
        log.info("hold_until: released")

    # Pretest-parity baseline captured at session 1 (when parser is loaded).
    # Holds {'failing': set, 'passing': set}; passed unchanged to every
    # Session so subsequent sessions inherit the baseline. None when not
    # populated or not applicable (no parser / empty pretest).
    pretest_parsed_verdict: dict | None = None
    from ..savings import close_ledger
    from ..system_log import close_system_log
    artifact_dir.mkdir(parents=True, exist_ok=True)
    recovery_plan = RecoveryPlan(recovered=False)
    if resume_from_artifacts or resume_path is not None:
        recovery_plan = recover_interrupted_trace(
            trace_path,
            mode=getattr(cfg, "interrupted_turn_mode", "mechanical"),
        )
    start_session_num = _next_session_number(trace_path) if resume_from_artifacts else 1
    end_session_num = start_session_num + cfg.max_sessions - 1

    # Resolve the selected backend before pretests or model tool calls. A
    # successful container preflight pins execution to the inspected image ID;
    # a non-strict failure degrades once, loudly, rather than retrying an
    # unavailable runtime on every command.
    env_fields = compute_runtime_envelope_fields(cfg, work_dir)
    if (
        getattr(cfg, "sandbox_backend", "bwrap") == "container"
        and cfg.sandbox_bash
    ):
        if env_fields["sandbox_engaged"]:
            cfg = replace(
                cfg,
                sandbox_container_image=env_fields["container_image_digest"],
            )
        elif not getattr(cfg, "sandbox_required", False):
            log.warning(
                "container_preflight: %s — running without sandbox because "
                "sandbox_required=false",
                env_fields.get("container_preflight_error") or "unavailable",
            )
            cfg = replace(cfg, sandbox_bash=False)
        if "cfg" in getattr(client, "__dict__", {}):
            client.cfg = cfg

    with open(trace_path, "a") as trace_file:
        # First-event-of-task envelope: records the runtime conditions that
        # the rest of the trace is conditioned on. Without this, post-hoc
        # analysis cannot distinguish a sandboxed run from a silently
        # degraded unsandboxed run. Emitted only
        # when the trace was empty at task start (i.e. session 1 of a fresh
        # task, not a resumed task whose prior sessions already wrote).
        if trace_path.stat().st_size == 0:
            _emit_trace_event(trace_file, "runtime_envelope", **env_fields)
            log.info(
                "runtime_envelope: sandbox_mode=%s backend=%s engaged=%s "
                "bwrap=%s preflight=%s container=%r runtime=%r digest=%r",
                env_fields["sandbox_mode"], env_fields["sandbox_backend"],
                env_fields["sandbox_engaged"],
                env_fields["bwrap_present"], env_fields["bwrap_preflight_passed"],
                env_fields["yuj_container"],
                env_fields["container_runtime"],
                env_fields["container_image_digest"],
            )
            if env_fields["bwrap_preflight_error"] and not env_fields["yuj_container"]:
                log.warning("bwrap_preflight: %s", env_fields["bwrap_preflight_error"])
            # Strict mode: refuse to start the session loop unsandboxed.
            # _run_in_sandbox would also catch this on the first bash call,
            # but failing here is louder and avoids any pretest noise.
        if (getattr(cfg, "sandbox_required", False)
                and cfg.sandbox_bash
                and not env_fields["sandbox_engaged"]):
            raise RuntimeError(
                f"sandbox_required=true but sandbox_engaged=false "
                f"(sandbox_mode={env_fields['sandbox_mode']}, "
                f"sandbox_backend={env_fields['sandbox_backend']!r}, "
                f"bwrap_bin={cfg.bwrap_bin!r}, "
                f"bwrap_present={env_fields['bwrap_present']}, "
                f"bwrap_preflight_passed={env_fields['bwrap_preflight_passed']}, "
                f"bwrap_preflight_error={env_fields['bwrap_preflight_error']!r}, "
                f"container_runtime={env_fields['container_runtime']!r}, "
                f"container_preflight_error="
                f"{env_fields['container_preflight_error']!r}, "
                f"yuj_container={env_fields['yuj_container']!r}). Refusing "
                "to start a session that would run model commands unsandboxed."
            )
        for session_num in range(start_session_num, end_session_num + 1):
            # Pretest: run failing tests BEFORE every session. Verdict becomes
            # the first block of the session's first user message. On sessions
            # 2+ we short-circuit to success if the pretest already exits
            # green — no model invocation needed.
            _pretest_t0 = time.time()
            pretest_block = run_pretest(
                work_dir,
                pretest_script=pretest_script,
                pretest_timeout=cfg.pretest_timeout,
                pretest_head_chars=cfg.pretest_head_chars,
                pretest_tail_chars=cfg.pretest_tail_chars,
            )
            _pretest_duration_ms = int((time.time() - _pretest_t0) * 1000)
            # Record a pretest_run trace event so replay tools can answer
            # what pretest said at session N
            # start?" without re-executing the script.
            _emit_trace_event(
                trace_file, "pretest_run",
                session_number=session_num,
                duration_ms=_pretest_duration_ms,
                chars=len(pretest_block),
            )
            # Record the pretest-block character cost on the savings ledger as
            # `pretest_block` per
            # session — mirrors the existing protocol_commandments per-
            # task entry.
            if pretest_block:
                from ..savings import get_ledger as _get_ledger
                _get_ledger().record(
                    bucket="pretest_block",
                    layer="harness",
                    mechanism=f"session_{session_num}",
                    input_chars=0,
                    output_chars=len(pretest_block),
                    measure_type="exact",
                    ctx={"duration_ms": _pretest_duration_ms},
                )
            # Parse pretest output on session 1 ONLY — this is the task's
            # ground-truth baseline. Subsequent sessions inherit the same
            # baseline (pretest on session 2+ may look different after
            # mid-task progress but should not re-seed).
            if (session_num == start_session_num
                    and output_parser is not None
                    and pretest_block
                    and cfg.done_require_pretest_parity):
                try:
                    from ...bash_quirks import parse_structured
                    pre_parsed = parse_structured(pretest_block, output_parser)
                    tests = pre_parsed.get("tests") or {}
                    failing = {t for t, v in tests.items() if v in ("FAILED", "FAIL", "ERROR")}
                    passing = {t for t, v in tests.items() if v in ("PASSED", "PASS")}
                    if failing or passing:
                        pretest_parsed_verdict = {"failing": failing, "passing": passing}
                        log.info("Pretest parity baseline: %d failing, %d passing",
                                 len(failing), len(passing))
                    else:
                        log.info("Pretest not structurally parseable — done_guard falls back to heuristic")
                except Exception as e:
                    log.debug("Pretest parse failed: %s", e)
            if session_num > start_session_num and _pretest_is_green(pretest_block):
                log.info("Pretest exited green at session start — short-circuiting.")
                write_checkpoint(artifact_dir, cfg.model, "completed")
                success = True
                sessions_used = session_num - start_session_num
                break

            # Normalize file timestamps before the first model turn. A
            # pretest can change directory times through temporary files.
            # Changing times can change `ls -la` output even with fixed
            # model sampling. Later sessions must keep the model's changes.
            if session_num == start_session_num:
                _normalize_repo_timestamps(work_dir)

            if session_num == start_session_num:
                if resume_from_artifacts:
                    initial = build_resume_prompt_from_trace(
                        trace_path, cfg, task_description
                    ) or task_prompt
                    if recovery_plan.recovered:
                        initial = (
                            recovery_plan.resume_prompt_line
                            + "\n\n"
                            + initial
                        )
                else:
                    initial = task_prompt
                if cfg.prompt_addendum and not resume_from_artifacts:
                    initial = initial.rstrip() + "\n\n" + cfg.prompt_addendum
                if not resume_from_artifacts:
                    task_description = initial
            else:
                mechanical_resume = build_resume_prompt(
                    prev_result, prev_session, cfg, task_description
                )
                initial = apply_pending_handoff(
                    mechanical_resume, task=task_description,
                    handoff=pending_handoff,
                )
                pending_handoff = None

            if pretest_block:
                initial = pretest_block + "\n" + initial
            log.info("[session %d/%d] %s", session_num, end_session_num, work_dir.name)
            model_binding = model_role_runtime.begin_model_session(client, cfg)
            session_client, session_cfg = model_binding.client, model_binding.config
            thinking_fields = thinking_trace_fields(session_cfg, session_client)
            # Trace: session start
            _emit_trace_event(
                trace_file, "session_start",
                session_number=session_num,
                context_contract=context_contract,
                sandbox_backend=env_fields["sandbox_backend"],
                container_runtime=env_fields["container_runtime"],
                container_image_digest=env_fields["container_image_digest"],
                sandbox_env_names=list(effective_env),
                **prompt_metadata.trace_fields(),
                **thinking_fields,
                **model_binding.trace_fields(),
                **ignore_policy.trace_fields(),
                **(
                    worktree_info.session_start_fields()
                    if worktree_info is not None else {}
                ),
            )
            if state_path is not None:
                write_state_from_trace(trace_path, state_path,
                                       max_result_chars=session_cfg.max_output_chars)

            ctx = build_context_manager(
                context_class, session_cfg, work_dir, initial, session_num,
                model_binding.token_estimator or token_estimator,
            )
            if getattr(cfg, "turn_snapshots_enabled", False):
                from ..turn_snapshots import ensure_snapshot_setup
                ensure_snapshot_setup(work_dir)
            session = Session(
                session_cfg, session_client, system_prompt, initial, str(work_dir),
                context_manager=ctx, trace_file=trace_file, session_number=session_num,
                trace_path=trace_path, state_path=state_path,
                output_control=output_control,
                universal_rewrites=universal_rewrites,
                forbidden_rules=forbidden_rules,
                redirect_rules=redirect_rules,
                redactions=redactions,
                output_parser=output_parser,
                pretest_parsed=pretest_parsed_verdict,
                checkpoint_store=checkpoint_store,
                injections=resolved_injections,
                artifact_dir=artifact_dir,
                adaptive_control_baseline_config_paths=tuple(
                    (run_metadata or {}).get("config_paths", ())
                    or getattr(cfg, "adaptive_control_baseline_config_paths", ())
                ),
                ignore_policy=ignore_policy,
                effective_env=effective_env,
                allow_login_shell=allow_login_shell,
            )
            session._cache_usage_accumulator = cache_usage
            model_role_runtime.bind_session_model_roles(
                session, session_client, role_usage,
            )
            if resume_from_artifacts and getattr(
                session_cfg, "rewind_enabled", False
            ):
                from ..turn_snapshots import apply_pending_rewind_resume
                apply_pending_rewind_resume(session)
            if session_num == start_session_num and resume_path is not None:
                inject_resume_messages(
                    session,
                    resume_path,
                    initial,
                    recovery=recovery_plan,
                )
            # Emit resolved thresholds so trace replay across config changes
            # is reproducible. At
            # session 2+ also include the prior session's terminal
            # counters so a reader can anchor cross-session resets.
            _g = getattr(session, "_guards", None)
            if _g is not None:
                _emit_trace_event(
                    trace_file, "guardrail_init",
                    session_number=session_num,
                    rumination_nudge_threshold=getattr(_g, "rumination_nudge_threshold", 0),
                    rumination_nudge_threshold_post_mutation=getattr(_g, "rumination_nudge_threshold_post_mutation", 0),
                    rumination_arm_threshold=getattr(_g, "rumination_arm_threshold", 0),
                    deque_maxlen=_g.recent_calls.maxlen if _g.recent_calls is not None else 0,
                    prior_session_done_blocked=agg_done_blocked,
                    prior_session_intent_block=agg_intent_block,
                    prior_session_gate_block=agg_gate_block,
                    prior_session_contract_block=agg_contract_block,
                    prior_session_same_class_error=agg_same_class_error,
                    prior_session_commit_violation=agg_commit_violation,
                    prior_session_mutation_repeat=agg_mutation_repeat,
                    prior_session_verify_repeat=agg_verify_repeat,
                )
            result = session.run()
            log.info(
                "Session ended: %s (turns=%d, prompt_tokens=%d)",
                result.finish_reason, result.turns, result.total_prompt_tokens,
            )

            # Aggregate metrics. done_blocked_count is read off the
            # session's GuardrailState (per-session counter; the failsafe
            # that converted to END will already have left it at the
            # threshold value, so summing across sessions gives a
            # comparison-ready total).
            agg_prompt += result.total_prompt_tokens
            agg_completion += result.total_completion_tokens
            agg_turns += result.turns
            agg_length_continuations += int(
                getattr(session, "_length_continuation_count", 0) or 0
            )
            _guards = getattr(session, "_guards", None)
            if _guards is not None:
                agg_done_blocked += getattr(_guards, "done_blocked_count", 0)
                agg_intent_block += getattr(_guards, "intent_block_count", 0)
                agg_gate_block += getattr(_guards, "gate_block_count", 0)
                agg_contract_block += getattr(_guards, "contract_block_count", 0)
                agg_same_class_error += getattr(_guards, "same_class_error_count", 0)
                agg_commit_violation += getattr(_guards, "commit_violation_count", 0)
                agg_mutation_repeat += getattr(_guards, "mutation_repeat_count", 0)
                agg_verify_repeat += getattr(_guards, "verify_repeat_count", 0)
            if getattr(session, "_trace_corrupted", False):
                agg_trace_corrupt += 1
            if result.finish_reason == "done_loop":
                done_loop_aborted = True
            sessions_used = session_num - start_session_num + 1

            # Trace: session end
            _emit_trace_event(
                trace_file, "session_end",
                session_number=session_num,
                finish_reason=result.finish_reason,
                turns=result.turns,
                total_prompt_tokens=result.total_prompt_tokens,
            )
            if state_path is not None:
                write_state_from_trace(trace_path, state_path,
                                       max_result_chars=cfg.max_output_chars)

            if result.done:
                _auto_commit(work_dir, session_num, result.finish_reason)
                write_checkpoint(artifact_dir, cfg.model, "completed")
                success = True
                break

            if result.finish_reason == "error":
                # Commit mutations performed before a fatal API error too.
                # Previously only the non-error path called _auto_commit, so the post-
                # mortem digest lost the mutation list for any task that
                # ended in an upstream API failure.
                _auto_commit(work_dir, session_num, result.finish_reason)
                write_checkpoint(artifact_dir, cfg.model, "error")
                break

            if result.finish_reason == "approval_required":
                write_checkpoint(artifact_dir, cfg.model, "paused")
                break

            # Auto-commit for non-error sessions.
            _auto_commit(work_dir, session_num, result.finish_reason)

            # The per-task wall-clock budget bounds how long one task can run.
            # It is checked after the session's
            # auto-commit so the partial work survives.
            wall_limit = int(getattr(cfg, "task_wall_clock_limit_s", 0) or 0)
            if wall_limit > 0:
                elapsed = time.time() - start_time
                if elapsed >= wall_limit:
                    log.warning(
                        "task_wall_clock: elapsed=%.0fs limit=%ds; ending task",
                        elapsed, wall_limit,
                    )
                    write_checkpoint(artifact_dir, cfg.model, "error")
                    # Mark the loop as terminated by wall budget. Use a
                    # synthetic SessionResult only for finish_reason
                    # propagation — the real session already ended cleanly.
                    result = SessionResult(
                        result.turns,
                        "task_wall_clock",
                        done=False,
                        total_prompt_tokens=result.total_prompt_tokens,
                        total_completion_tokens=result.total_completion_tokens,
                    )
                    break

            pending_handoff = maybe_prepare_boundary_handoff(
                cfg=cfg, client=client, task=task_description,
                trace_path=trace_path, trace_file=trace_file,
                state_path=state_path, session_number=session_num,
                finish_reason=result.finish_reason,
                has_next_session=session_num < end_session_num,
                tokenizer=getattr(session, "_tokenizer", None),
            )
            # NORMAL_LIFECYCLE and MODEL_STUCK → continue to next session
            # (different preamble generated by build_resume_prompt)
            prev_session = session
            prev_result = result
        else:
            log.warning("Max sessions (%d) exhausted for %s", cfg.max_sessions, work_dir.name)
            write_checkpoint(artifact_dir, cfg.model, "error")

    # Write run metrics (#57, #60)
    wall_clock = time.time() - start_time
    total_tokens = agg_prompt + agg_completion
    metrics: dict = {
        "total_prompt_tokens": agg_prompt,
        "total_completion_tokens": agg_completion,
        "total_tokens": total_tokens,
        "wall_clock_seconds": round(wall_clock, 2),
        "sessions_used": sessions_used,
        "total_turns": agg_turns,
        "done_blocked_total": agg_done_blocked,
        # Cross-session totals for guardrail counters. Previously only
        # done_blocked_total surfaced; the rest died with the session.
        "intent_block_total": agg_intent_block,
        "gate_block_total": agg_gate_block,
        "contract_block_total": agg_contract_block,
        "same_class_error_total": agg_same_class_error,
        "commit_violation_total": agg_commit_violation,
        "mutation_repeat_total": agg_mutation_repeat,
        "verify_repeat_total": agg_verify_repeat,
        "trace_corrupt_count": agg_trace_corrupt,
        "length_continuations": agg_length_continuations,
        "done_loop_aborted": done_loop_aborted,
    }
    if sessions_used > 0:
        metrics["time_per_session_seconds"] = round(wall_clock / sessions_used, 2)
    if agg_turns > 0:
        metrics["tokens_per_turn"] = round(total_tokens / agg_turns, 2)
    metrics.update(cache_usage.metrics_fields())
    metrics.update(role_usage.metrics_fields())
    metrics.update(model_role_runtime.model_fallback_metrics(client))
    metrics["file_checkpoints"] = (
        checkpoint_store.metrics_payload()
        if checkpoint_store is not None
        else {
            "enabled": False,
            "count": 0,
            "total_duration_ms": 0.0,
            "per_call": [],
        }
    )
    write_run_metrics(artifact_dir, metrics, provenance)
    close_ledger()
    close_system_log()
    return success
