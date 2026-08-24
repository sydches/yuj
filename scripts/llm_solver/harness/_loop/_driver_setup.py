"""Pre-loop setup helpers for ``solve_task``.

Extracted from ``driver.py`` to keep the outer loop below the project's
500-line cap. The functions here compute setup state — none of them
emit trace events or interact with the per-session loop. Trace emit
sites (``runtime_envelope``, ``pretest_run``, ``session_start``,
``session_end``) all stay at the ``solve_task`` call-site so
state.json projection order is preserved.

For ``_compute_runtime_envelope_fields``: this returns a dict of
fields ready to splice into ``_emit_trace_event(trace_file,
"runtime_envelope", **fields)``. The emit + ``sandbox_required`` raise
stay in ``solve_task`` — both are load-bearing for trace-order and
loud-failure semantics.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...config import Config
from ..._shared.paths import expand_user_path
from ..._shared.telemetry_paths import ensure_telemetry_dir, trace_path
from ..context_contract import build_context_contract
from ..guardrails import build_guardrail_registry
from ..injections import Injection, load_injections_with_metadata
from ..stream_rules import StreamRule, load_stream_rules
from ..project_instructions import (
    discover_project_instructions,
    find_project_root,
    resolve_project_instruction_imports,
)
from ..solver import (
    assemble_system_prompt,
    collect_provenance,
    resolve_system_prompt_source,
)
from . import (
    _apply_profile_preamble,
    _load_bash_transforms,
    _record_session_start_costs,
    _resolve_token_estimator,
)

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PromptAssemblyMetadata:
    """Secret-free metadata for trace and exact prompt-component costing."""

    arm_label: str | None = None
    arm_chars: int = 0
    project_instruction_files: tuple[dict[str, object], ...] = ()
    project_instruction_bytes: int = 0
    project_instruction_imported_bytes: int = 0
    project_instruction_resolved_bytes: int = 0
    project_instruction_chars: int = 0
    project_instructions_truncated: bool = False
    prompt_import_tree: tuple[dict[str, object], ...] = ()
    loaded_skills: tuple[dict[str, object], ...] = ()
    skills_catalog_chars: int = 0

    def trace_fields(self) -> dict[str, object]:
        return {
            "project_instruction_files": [
                dict(record) for record in self.project_instruction_files
            ],
            "project_instruction_bytes": self.project_instruction_bytes,
            "project_instruction_imported_bytes": (
                self.project_instruction_imported_bytes
            ),
            "project_instruction_resolved_bytes": (
                self.project_instruction_resolved_bytes
            ),
            "project_instructions_truncated": (
                self.project_instructions_truncated
            ),
            "prompt_import_tree": [
                dict(record) for record in self.prompt_import_tree
            ],
            "loaded_skills": [
                dict(record) for record in self.loaded_skills
            ],
        }


def resolve_run_paths(
    repo_dir: Path,
    artifacts_dir: Path | None,
) -> tuple[Path, Path, Path]:
    """Return work, artifact, and trace paths for measurement or assistant use."""
    work_dir = Path(repo_dir)
    artifact_dir = Path(artifacts_dir) if artifacts_dir is not None else work_dir
    if artifacts_dir is None:
        ensure_telemetry_dir(work_dir)
        return work_dir, artifact_dir, trace_path(work_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return work_dir, artifact_dir, artifact_dir / ".trace.jsonl"


def setup_run_outputs(
    cfg: Config,
    client,
    work_dir: Path,
    artifact_dir: Path,
    *,
    assistant_artifacts: bool,
    savings_dir: Path | None,
    transcript_dir: Path | None,
    system_prompt: str,
    system_prompt_file: Path | None,
    prompt_metadata: PromptAssemblyMetadata,
) -> None:
    """Open the correct output files without placing assistant state in the repo."""
    if not assistant_artifacts:
        setup_savings_and_transcript(
            cfg, client, work_dir, savings_dir, transcript_dir,
            system_prompt, system_prompt_file, prompt_metadata,
        )
        return

    from ..savings import open_ledger
    from ..system_log import open_system_log
    open_ledger(artifact_dir / "savings.jsonl")
    open_system_log(artifact_dir / "system_log.jsonl").set_task(work_dir.name)
    _record_session_start_costs(
        cfg, client, system_prompt, system_prompt_file, prompt_metadata
    )
    if hasattr(client, "set_transcript"):
        client.set_transcript(artifact_dir / "transcript.log")


def load_system_prompt_and_provenance(
    cfg: Config,
    client,
    work_dir: Path,
    system_prompt_file: Path | None,
    profile_path: Path | None,
    run_metadata: dict | None,
    context_class,
    *,
    unreadable_paths: tuple[str, ...] | None = None,
    skill_catalog=None,
) -> tuple[str, dict, dict, PromptAssemblyMetadata]:
    """Build system_prompt, provenance, and context_contract.

    Returns the system prompt, provenance, context contract, and safe prompt
    component metadata. The provenance dict is mutated in place with
    variant_name + prompt_addendum if the config sets them.
    """
    prompt_unreadable_paths = (
        cfg.unreadable_paths
        if unreadable_paths is None
        else unreadable_paths
    )
    project_root = find_project_root(work_dir, cfg.project_root_markers)
    arm_roots: list[Path] = [project_root]
    if system_prompt_file is not None:
        arm_parent = system_prompt_file.resolve().parent
        if arm_parent not in arm_roots:
            arm_roots.append(arm_parent)
    arm = resolve_system_prompt_source(
        system_prompt_file,
        imports_enabled=cfg.imports_enabled,
        allowed_dirs=tuple(arm_roots),
        max_depth=cfg.imports_max_depth,
        unreadable_paths=prompt_unreadable_paths,
    )
    resolved_arm = arm.content if arm is not None else None
    prompt_import_tree: list[dict[str, object]] = []
    if arm is not None:
        prompt_import_tree.append(arm.trace_record())
    project_content = ""
    project_records: tuple[dict[str, object], ...] = ()
    project_bytes = 0
    project_imported_bytes = 0
    project_resolved_bytes = 0
    project_truncated = False
    if cfg.project_docs_enabled:
        global_dir = (
            expand_user_path(cfg.project_doc_global_dir)
            if cfg.project_doc_global_dir.strip()
            else None
        )
        project = discover_project_instructions(
            work_dir,
            global_dir=global_dir,
            doc_names=cfg.project_doc_names,
            max_bytes=cfg.project_doc_max_bytes,
            root_markers=cfg.project_root_markers,
            unreadable_paths=prompt_unreadable_paths,
            defer_byte_cap=True,
        )
        project = resolve_project_instruction_imports(
            project,
            enabled=cfg.imports_enabled,
            max_depth=cfg.imports_max_depth,
            unreadable_paths=prompt_unreadable_paths,
        )
        project_content = project.content
        project_records = tuple(project.trace_records())
        project_bytes = project.document_bytes
        project_imported_bytes = project.imported_bytes
        project_resolved_bytes = project.resolved_bytes
        project_truncated = project.truncated
        prompt_import_tree.extend(project.prompt_import_tree)
        for diagnostic in project.diagnostics:
            log.warning(
                "project_instruction_read: path=%s kind=%s message=%s",
                diagnostic.path,
                diagnostic.error_kind,
                diagnostic.message,
            )
    if skill_catalog is None:
        from ..skills import discover_skills
        skill_catalog = discover_skills(
            work_dir,
            enabled=getattr(cfg, "skills_enabled", False),
            skills_dirs=getattr(cfg, "skills_dirs", ()),
            skill_paths=getattr(cfg, "skill_paths", ()),
            root_markers=cfg.project_root_markers,
            unreadable_paths=prompt_unreadable_paths,
        )
    skills_block = skill_catalog.format_prompt_block()
    prompt_metadata = PromptAssemblyMetadata(
        arm_label=(
            arm.source if arm is not None else None
        ),
        arm_chars=len(resolved_arm.rstrip()) if resolved_arm is not None else 0,
        project_instruction_files=project_records,
        project_instruction_bytes=project_bytes,
        project_instruction_imported_bytes=project_imported_bytes,
        project_instruction_resolved_bytes=project_resolved_bytes,
        project_instruction_chars=len(project_content),
        project_instructions_truncated=project_truncated,
        prompt_import_tree=tuple(prompt_import_tree),
        loaded_skills=tuple(skill_catalog.trace_records()),
        skills_catalog_chars=len(skills_block),
    )
    system_prompt = _apply_profile_preamble(
        assemble_system_prompt(
            cfg.system_header,
            resolved_arm=resolved_arm,
            project_instructions=project_content,
            skills=skills_block,
        ),
        client,
    )
    # Pass the resolved system prompt so its sha256 lands in provenance.
    thinking_resolution = getattr(client, "__dict__", {}).get(
        "_thinking_resolution"
    )
    from .model_role_runtime import model_fallback_provenance
    provenance = collect_provenance(
        cfg, profile_path, resolved_system_prompt=system_prompt,
        run_metadata=run_metadata, thinking_resolution=thinking_resolution,
        fallback_provenance=model_fallback_provenance(client),
    )
    context_contract = build_context_contract(context_class, cfg)
    provenance["context_contract"] = context_contract
    if cfg.variant_name:
        provenance["variant_name"] = cfg.variant_name
        provenance["prompt_addendum"] = cfg.prompt_addendum
    return system_prompt, provenance, context_contract, prompt_metadata


def load_session_injections(
    cfg: Config,
    work_dir: Path,
    *,
    unreadable_paths: tuple[str, ...] | None = None,
) -> tuple[tuple[Injection, ...], tuple[dict[str, object], ...]]:
    """Resolve injection files before ``session_start`` is emitted."""
    if not cfg.injections_enabled:
        return (), ()
    project_root = find_project_root(work_dir, cfg.project_root_markers)
    prompt_unreadable_paths = (
        cfg.unreadable_paths
        if unreadable_paths is None
        else unreadable_paths
    )
    loaded = load_injections_with_metadata(
        work_dir / cfg.injections_dir,
        imports_enabled=cfg.imports_enabled,
        imports_max_depth=cfg.imports_max_depth,
        allowed_dirs=(project_root,),
        unreadable_paths=prompt_unreadable_paths,
    )
    return loaded.injections, loaded.prompt_import_tree


def load_session_stream_rules(
    cfg: Config,
    work_dir: Path,
) -> tuple[tuple[StreamRule, ...], tuple[dict[str, object], ...]]:
    """Validate stream rules once at task startup, before any model call."""
    if not cfg.stream_rules_enabled:
        return (), ()
    loaded = load_stream_rules(
        work_dir / cfg.stream_rules_dir,
        display_dir=cfg.stream_rules_dir,
        allowed_root=work_dir,
    )
    return loaded.rules, loaded.files


def thinking_trace_fields(cfg: Config, client) -> dict[str, object]:
    """Return effective per-run reasoning fields for session_start."""
    resolution = getattr(client, "__dict__", {}).get("_thinking_resolution")
    if resolution is not None:
        return resolution.trace_fields()
    return {"thinking_level": cfg.thinking_level}


def setup_savings_and_transcript(
    cfg: Config,
    client,
    repo_dir: Path,
    savings_dir: Path | None,
    transcript_dir: Path | None,
    system_prompt: str,
    system_prompt_file: Path | None,
    prompt_metadata: PromptAssemblyMetadata,
) -> None:
    """Open savings ledger, set client transcript, record session-start costs.

    Side-effectful only. Always-on; Bucket A observability. Written
    OUTSIDE repo_dir (the agent's sandbox cwd) so its mere existence in
    ``ls -la`` doesn't change with wall-clock mtime across runs — under
    temp=0 a single timestamp char flips the model's path. One file per
    task at run_dir/savings/<task>.jsonl.
    """
    from ..savings import open_ledger
    # Caller may override the savings dir with savings_dir kwarg; default is
    # the historical repo_dir.parent.parent / "savings/" layout (assumes
    # <run_dir>/repos/<iid>/ shape). Explicit override lets callers using
    # arbitrary --task paths (e.g. polyglot scratch trees) put artifacts
    # where they want without depending on this path-arithmetic accident.
    _savings_dir = savings_dir if savings_dir is not None else (repo_dir.parent.parent / "savings")
    _savings_dir.mkdir(parents=True, exist_ok=True)
    open_ledger(_savings_dir / f"{repo_dir.name}.jsonl")
    # System log (harness self-observations — see harness/system_log.py):
    # one file per run beside savings/ and transcripts/, outside the
    # sandbox cwd for the same mtime-determinism reason as the ledger.
    # Events are stamped with the task name so multi-task runs stay
    # attributable.
    from ..system_log import open_system_log
    open_system_log(_savings_dir.parent / "system_log.jsonl").set_task(repo_dir.name)
    _record_session_start_costs(
        cfg, client, system_prompt, system_prompt_file, prompt_metadata
    )

    # Keep one transcript of every HTTP exchange for the task. Write it
    # outside repo_dir so sandbox searches cannot read it. Keep the counter
    # monotonic across sessions.
    if hasattr(client, "set_transcript"):
        _tx_dir = transcript_dir if transcript_dir is not None else (repo_dir.parent.parent / "transcripts")
        _tx_dir.mkdir(parents=True, exist_ok=True)
        client.set_transcript(_tx_dir / f"{repo_dir.name}.log")


def resolve_task_format(cfg: Config, repo_dir: Path) -> Config:
    """Resolve ``analysis_task_format`` to the repo's actual runner.

    Multilingual entry point. When ``analysis_task_format`` is ``"auto"``
    (the multi-language default for benches whose tasks span go / js / ts
    / rust / python), detect the runner from ``repo_dir`` markers
    (``go.mod``, ``Cargo.toml``, ``package.json``, ``pyproject.toml`` ...)
    and return a copy of ``cfg`` with the resolved concrete format so that
    every downstream consumer — verification-command detection, structured
    output parsing, and thus the trace fields the hurdle detector reads —
    uses the task's real language.

    An explicit format (e.g. ``"pytest"``) is left untouched: pinning wins,
    and the existing ``task_format_mismatch`` warning still fires if it
    disagrees with the detected runner.
    """
    import dataclasses

    fmt = getattr(cfg, "analysis_task_format", "") or ""
    if fmt != "auto":
        return cfg
    try:
        from ...language_quirks import detect_runner
        detected = detect_runner(repo_dir)
    except Exception:
        log.warning("resolve_task_format: detection failed for %s; "
                    "falling back to pytest", repo_dir)
        detected = "pytest"
    log.info("resolve_task_format: analysis_task_format=auto resolved to "
             "%s for %s", detected, repo_dir)
    return dataclasses.replace(cfg, analysis_task_format=detected)


def load_transforms_and_estimator(cfg: Config, client, repo_dir: Path):
    """Return (output_control, universal_rewrites, forbidden_rules,
    redirect_rules, redactions, output_parser, token_estimator).

    Also emits two task-format or pagination warnings as side effects:
      - task_format_mismatch when the detected runner does not match
        cfg.analysis_task_format;
      - tool_quirks/glob caps disabled when search_pagination is off.
    """
    (
        output_control, universal_rewrites, forbidden_rules, redirect_rules,
        redactions, output_parser,
    ) = _load_bash_transforms(
        cfg,
        force_load_all=bool(getattr(cfg, "adaptive_policy_enabled", False)),
    )
    # Warn once per task when the detected runner does not match
    # cfg.analysis_task_format and
    # bash_transforms_task_format_enabled is on. _is_test_command will
    # use {fmt} patterns but run_tests will dispatch to {detected};
    # condense_output won't fire on detected-runner output.
    if cfg.bash_transforms_task_format_enabled:
        try:
            from ...language_quirks import detect_runner as _detect_runner_for_warn
            _detected = _detect_runner_for_warn(repo_dir)
            _fmt = getattr(cfg, "analysis_task_format", "")
            if _fmt and _detected and _detected != _fmt:
                log.warning(
                    "task_format_mismatch: detected runner=%s but "
                    "analysis_task_format=%s; output_control / output_parser "
                    "will not fire on detected-runner output",
                    _detected, _fmt,
                )
        except Exception:
            pass

    # Log when search_pagination is off. tool_quirks/glob caps depend on
    # the paginated envelope;
    # if pagination is off, glob refusals don't surface.
    if not getattr(cfg, "search_pagination_enabled", True):
        log.warning(
            "tool_quirks/glob caps disabled because search_pagination_enabled=false"
        )

    token_estimator = _resolve_token_estimator(client)
    return (
        output_control, universal_rewrites, forbidden_rules, redirect_rules,
        redactions, output_parser, token_estimator,
    )


def compute_runtime_envelope_fields(cfg: Config, repo_dir: Path) -> dict[str, Any]:
    """Build the dict of fields for the first-event ``runtime_envelope`` event.

    Returns a dict of kwargs ready to splice into
    ``_emit_trace_event(trace_file, "runtime_envelope", **fields)``.
    The caller emits the event, logs the summary, and applies the
    ``sandbox_required`` strict-mode raise — all kept at solve_task so
    the trace-event order and loud-failure surface stay one-shot
    visible at the dispatch site.

    Records the runtime conditions that the rest of the trace is
    conditioned on. Without this, post-hoc analysis cannot distinguish
    a sandboxed run from a silently degraded unsandboxed run.
    """
    _container_id = os.environ.get("YUJ_CONTAINER", "")
    _configured_backend = getattr(cfg, "sandbox_backend", "bwrap")
    _bwrap_present = Path(cfg.bwrap_bin).is_file()
    _container_runtime: str | None = None
    _container_image_digest: str | None = None
    _container_preflight_err: str | None = None
    # Bwrap PREFLIGHT (Codex S3): actually exec a tiny command in
    # a fresh user+net namespace and check exit. Catches the
    # silent class of failures where bwrap is INSTALLED but the
    # kernel rejects unprivileged userns clone (sysctl
    # kernel.unprivileged_userns_clone=0, AppArmor profile
    # changed, seccomp from a parent supervisor). Today these
    # produce a per-call warning + silent unsandboxed fallback.
    # Cached at module level: pays once per process. Skipped
    # entirely in container mode (docker exec doesn't use bwrap).
    if _configured_backend == "container":
        if _container_id:
            raise RuntimeError(
                "sandbox.backend='container' cannot be combined with legacy "
                "YUJ_CONTAINER; unset YUJ_CONTAINER or select "
                "sandbox.backend='bwrap'"
            )
        _ambient_unshare_net = None
        _bwrap_preflight_passed = None
        _bwrap_preflight_err = None
        _container_runtime = getattr(
            cfg, "sandbox_container_runtime", "docker"
        )
        if not cfg.sandbox_bash:
            _sandbox_mode = "none"
            _sandbox_engaged = False
        else:
            from ..sandbox.container_backend import (
                ContainerBackend,
                ContainerBackendError,
            )

            backend = ContainerBackend(
                runtime=_container_runtime,
                image=getattr(cfg, "sandbox_container_image", ""),
                flags=tuple(
                    getattr(cfg, "sandbox_container_flags", ()) or ()
                ),
            )
            try:
                runtime_bin = backend.resolve_runtime(
                    sandbox_required=bool(
                        getattr(cfg, "sandbox_required", False)
                    )
                )
                if runtime_bin is None:
                    _container_preflight_err = (
                        f"container runtime {_container_runtime!r} is missing"
                    )
                else:
                    _container_image_digest = backend.image_digest(runtime_bin)
            except ContainerBackendError as exc:
                _container_preflight_err = str(exc)
            _sandbox_engaged = bool(_container_image_digest)
            _sandbox_mode = "container" if _sandbox_engaged else "none"
    elif _configured_backend != "bwrap":
        raise ValueError(
            "sandbox.backend must be 'bwrap' or 'container'; "
            f"got {_configured_backend!r}"
        )
    elif _container_id:
        _sandbox_mode = "container"
        _sandbox_engaged = True
        _bwrap_preflight_passed: bool | None = None
        _bwrap_preflight_err: str | None = None
        # Ambient-container egress probe — when YUJ_CONTAINER=ambient,
        # the model's bash subprocess is wrapped in `unshare -n` if the
        # outer container has CAP_SYS_ADMIN. Probe once at startup so
        # the trace records the actual leak-isolation state.
        if _container_id == "ambient":
            from .._tools._run_in_sandbox import _probe_ambient_unshare_net
            _ambient_unshare_net = _probe_ambient_unshare_net()
        else:
            _ambient_unshare_net = None
    else:
        _ambient_unshare_net = None
        if _bwrap_present:
            from ..sandbox import bwrap_preflight as _bwrap_preflight
            _bwrap_preflight_passed, _bwrap_preflight_err = _bwrap_preflight(cfg.bwrap_bin)
        else:
            _bwrap_preflight_passed = False
            _bwrap_preflight_err = "bwrap binary not present (cfg.bwrap_bin)"
        if cfg.sandbox_bash and _bwrap_present and _bwrap_preflight_passed:
            _sandbox_mode = "bwrap"
            _sandbox_engaged = True
        else:
            _sandbox_mode = "none"
            _sandbox_engaged = False
    # Record the active guardrail name-to-qualname map so trace replay
    # across overrides is reproducible. This uses the no-override default.
    _gr = build_guardrail_registry()

    def _qn(fn):
        m = getattr(fn, "__module__", "?")
        n = getattr(fn, "__qualname__", getattr(fn, "__name__", "?"))
        return f"{m}.{n}"
    _guardrail_map = {
        "turn_pre":  {k: _qn(v) for k, v in _gr.turn_pre_dispatch.items()},
        "tool_pre":  {k: _qn(v) for k, v in _gr.tool_pre_dispatch.items()},
        "tool_post": {k: _qn(v) for k, v in _gr.tool_post_dispatch.items()},
        "observer":  {k: _qn(v) for k, v in _gr.observers.items()},
    }
    # Stamp content hashes of the quirk TOMLs and the detected
    # runner so trace replay can branch on quirk version + runner
    # without re-detecting from disk.
    _quirk_hashes: dict[str, str] = {}
    try:
        import hashlib as _hashlib
        from ... import bash_quirks as _bq
        from ... import tool_quirks as _tq
        for label, path in (
            ("bash_quirks/forbidden.toml", Path(_bq.__file__).parent / "forbidden.toml"),
            ("bash_quirks/redactions.toml", Path(_bq.__file__).parent / "redactions.toml"),
            ("bash_quirks/universal_rewrites.toml", Path(_bq.__file__).parent / "universal_rewrites.toml"),
            ("tool_quirks/glob.toml", Path(_tq.__file__).parent / "glob.toml"),
        ):
            if path.is_file():
                _quirk_hashes[label] = _hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    except Exception:
        pass
    _detected_runner = ""
    try:
        from ...language_quirks import detect_runner as _detect_runner
        _detected_runner = _detect_runner(repo_dir)
    except Exception:
        pass
    # Record expansion stats so a later check can answer whether the run had masks
    # applied?" without re-running glob expansion against a
    # config snapshot.
    _unreadable_n_files = 0
    _unreadable_n_dirs = 0
    _unreadable_zero_match: list[str] = []
    try:
        from ..sandbox import _expand_unreadable_paths, _is_specific_pattern
        import glob as _glob_check
        _patterns = tuple(getattr(cfg, "unreadable_paths", ()) or ())
        if _patterns:
            _, _unreadable_n_files, _unreadable_n_dirs = _expand_unreadable_paths(_patterns)
            for _p in _patterns:
                if _is_specific_pattern(_p):
                    try:
                        if not _glob_check.glob(_p, recursive=True, include_hidden=True):
                            _unreadable_zero_match.append(_p)
                    except Exception:
                        pass
    except Exception:
        pass
    return dict(
        session=1,
        sandbox_mode=_sandbox_mode,
        sandbox_engaged=_sandbox_engaged,
        sandbox_backend=_configured_backend,
        container_runtime=_container_runtime,
        container_image_digest=_container_image_digest,
        container_preflight_error=_container_preflight_err,
        sandbox_bash_cfg=bool(cfg.sandbox_bash),
        sandbox_required_cfg=bool(getattr(cfg, "sandbox_required", False)),
        bwrap_bin=cfg.bwrap_bin,
        bwrap_present=_bwrap_present,
        bwrap_preflight_passed=_bwrap_preflight_passed,
        bwrap_preflight_error=_bwrap_preflight_err,
        yuj_container=_container_id or None,
        # Egress isolation state for ambient mode (None if not ambient).
        # True = bash subprocess wrapped in `unshare -n` (network closed)
        # False = wrap unavailable, bash has host-level network
        ambient_unshare_net=_ambient_unshare_net,
        task_id=repo_dir.name,
        guardrail_map=_guardrail_map,
        quirk_hashes=_quirk_hashes,
        detected_runner=_detected_runner,
        unreadable_paths_n_files=_unreadable_n_files,
        unreadable_paths_n_dirs=_unreadable_n_dirs,
        unreadable_paths_zero_match_patterns=_unreadable_zero_match,
        # Bump this on any argv-shape change. Today's set includes
        # --unshare-{net,pid,ipc,uts,cgroup,user},
        # --tmpfs $cwd/.git/hooks, --setenv (5 entries),
        # unreadable_paths masks. Version 3 adds the first-class per-call
        # Docker/Podman backend with no-pull, no-network, read-only-root,
        # capability-drop, identical-cwd mount, and explicit environment.
        sandbox_policy_version=3,
    )
