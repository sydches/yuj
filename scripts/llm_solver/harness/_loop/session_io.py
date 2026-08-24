"""Session I/O helpers — extracted from loop.py."""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TYPE_CHECKING

from ..schemas import get_tool_schemas
from .profile_resolution import (
    _apply_profile_schema_simplify,
    _apply_profile_tool_cap,
    _filter_disabled_tools,
    apply_profile_to_schemas,
    build_tool_surface,
)

if TYPE_CHECKING:
    from ...config import Config

log = logging.getLogger(__name__)


def _summarize_args(args: dict, max_chars: int) -> str:
    """Short summary of tool arguments for logging.

    max_chars is required (wired from cfg.args_summary_chars). No default:
    shadow defaults in harness code drift silently from config.
    """
    parts = []
    for k, v in args.items():
        s = str(v)
        if len(s) > max_chars:
            s = s[:max_chars - 3] + "..."
        parts.append(f"{k}={s!r}")
    return ", ".join(parts)


def _truncate_for_trace(s: str, maxlen: int) -> str:
    """Truncate a string for trace logging.

    Now takes an explicit max length — callers pass 200 for the action/args
    summary (where a short repr is always enough) and the tools.py output
    cap (20000 by default, from Config.max_output_chars) for the result.
    Previously defaulted to 200 for both, which silently stubbed file
    reads and broke the stateful context strategy.
    """
    if len(s) <= maxlen:
        return s
    return s[:maxlen - 3] + "..."


def _auto_commit(repo_dir: Path, session_num: int, finish_reason: str) -> None:
    """Commit changes in repo_dir if working tree is dirty. Local only.

    Best-effort: any subprocess failure (CalledProcessError, missing
    git, dirty index it can't resolve) is logged at warning level and
    swallowed — the harness must not crash a task because the auto-
    commit checkpoint failed.
    """
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(repo_dir), capture_output=True, text=True,
        )
        if not status.stdout.strip():
            return
        subprocess.run(["git", "add", "-A"], cwd=str(repo_dir), check=True)
        msg = f"yuj: session {session_num} checkpoint ({finish_reason})"
        # Supply an identity because an extracted repository may not have
        # user.name or user.email configured.
        subprocess.run(
            ["git", "-c", "user.name=yuj-harness",
             "-c", "user.email=yuj@localhost",
             "commit", "-m", msg],
            cwd=str(repo_dir), capture_output=True, check=True,
        )
        log.info("Auto-commit: session %d (%s)", session_num, finish_reason)
    except (subprocess.CalledProcessError, FileNotFoundError, OSError) as e:
        log.warning(
            "auto_commit_failed: session=%d finish=%s err=%s",
            session_num, finish_reason, e,
        )


# Resume base prompt now lives in config.toml [prompts] resume_base.


# ── Pretest input ────────────────────────────────────────────────────────

# Pretest truncation limits are now in config.toml [output] section:
#   pretest_head_chars, pretest_tail_chars
# The _truncate_pretest_output function below receives them as arguments.
#
# The harness does not rewrite task-runner paths in pretest output. The
# outside task runner must remove any paths that the model must not see
# before it returns the text.

_STATUS_WORD_RE = re.compile(
    r'\b(?:passed|failed|error|warnings?|deselected|no tests ran|no tests collected)\b'
)
_TIMING_RE = re.compile(r'\s*in\s+\d+\.\d+s')


def _sanitize_runner_timing(output: str) -> str:
    """Strip wall-clock timing from pytest/unittest summary lines.

    Pytest embeds sub-second timing (``13 failed in 1.55s``) that varies
    per invocation.  Under deterministic inference (temp=0, top-k=1) a
    single changed character flips the sampled path.  Stripping timing
    makes the pretest block byte-identical across runs of the same task.

    Operates per-line so ANSI color codes between the status word and
    the timing fragment don't defeat the match.
    """
    out_lines = []
    for line in output.split('\n'):
        if _STATUS_WORD_RE.search(line):
            line = _TIMING_RE.sub('', line)
        out_lines.append(line)
    return '\n'.join(out_lines)


def _normalize_repo_timestamps(repo_dir: Path) -> None:
    """Set every file/dir mtime under repo_dir (except .git/) to a fixed epoch.

    Removes wall-clock leakage that appears in the agent's first
    ``ls -la`` and would otherwise flip the model's path under
    deterministic inference (temp=0, top-k=1).
    """
    epoch = "2020-01-01T00:00:00"
    try:
        # Exclude .git contents (corrupts index timestamps) but touch
        # the .git directory itself so its entry in `ls -la` is stable.
        subprocess.run(
            ["find", str(repo_dir), "-not", "-path", f"{repo_dir}/.git/*",
             "-exec", "touch", "-d", epoch, "{}", "+"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=30,
        )
    except (subprocess.SubprocessError, FileNotFoundError, PermissionError):
        # Non-fatal: determinism degrades but the run continues.
        pass




def _record_session_start_costs(cfg: Config, client, system_prompt: str,
                                 system_prompt_file: Path | None,
                                 prompt_metadata=None) -> None:
    """Record one-time per-task costs on the savings ledger.

    Captures prompt/tool component buckets:
      system_prompt         — tokens paid by the configured system_header.
      tool_surface          — tokens paid by tool schemas at the active
                              tool_desc mode.
      protocol_commandments — tokens paid by resolved --system-prompt content.
      project_instructions  — tokens paid by resolved project-document blocks.
      skills_catalog        — tokens paid by startup skill metadata only.
      profile_behavioral    — tokens paid by the profile's behavioral
                              suffix (probe: run denormalize on a
                              minimal Commandments-tagged message and
                              measure the content delta).
      profile_preamble      — tokens paid by the profile's preamble
                              text (recorded only when non-empty).

    All records positive delta_chars (cost paid).
    """
    from ..savings import get_ledger
    ledger = get_ledger()

    # System header (base system prompt before any --system-prompt append).
    ledger.record(
        bucket="system_prompt",
        layer="config",
        mechanism="system_header",
        input_chars=0,
        output_chars=len(cfg.system_header),
        measure_type="exact",
        ctx={"tool_desc": cfg.tool_desc},
    )

    # Tool surface: schema JSON emitted to the model (post profile knobs).
    try:
        # Use one helper for the filter→simplify→cap composition so both
        # call sites cannot drift.
        schemas = build_tool_surface(cfg, client).active_schemas
        schema_chars = sum(len(json.dumps(s, default=str)) for s in schemas)
        ledger.record(
            bucket="tool_surface",
            layer="harness",
            mechanism=f"tool_schemas:{cfg.tool_desc}:effective",
            input_chars=0,
            output_chars=schema_chars,
            measure_type="exact",
            ctx={"n_tools": len(schemas)},
        )
    except Exception as e:
        log.debug("Tool-surface cost record skipped: %s", e)

    # Protocol commandments: the resolved arm content that was actually
    # assembled, including legacy @imports. New callers pass metadata so this
    # path never re-reads a mutable source file after prompt resolution.
    arm_label = getattr(prompt_metadata, "arm_label", None)
    arm_chars = getattr(prompt_metadata, "arm_chars", None)
    if arm_label is not None and arm_chars is not None:
        ledger.record(
            bucket="protocol_commandments",
            layer="L4_protocol",
            mechanism=arm_label,
            input_chars=0,
            output_chars=arm_chars,
            measure_type="exact",
            ctx={"path": arm_label},
        )
    elif system_prompt_file is not None and Path(system_prompt_file).is_file():
        try:
            commandments_chars = len(Path(system_prompt_file).read_text())
            ledger.record(
                bucket="protocol_commandments",
                layer="L4_protocol",
                mechanism=Path(system_prompt_file).name,
                input_chars=0,
                output_chars=commandments_chars,
                measure_type="exact",
                ctx={"path": Path(system_prompt_file).name},
            )
        except OSError as e:
            log.debug("Protocol commandments cost record skipped: %s", e)

    project_chars = int(
        getattr(prompt_metadata, "project_instruction_chars", 0) or 0
    )
    if project_chars:
        records = tuple(
            getattr(prompt_metadata, "project_instruction_files", ()) or ()
        )
        ledger.record(
            bucket="project_instructions",
            layer="L4_protocol",
            mechanism="repository_instruction_files",
            input_chars=0,
            output_chars=project_chars,
            measure_type="exact",
            ctx={
                "files": [str(record.get("path", "")) for record in records],
                "source_bytes": int(
                    getattr(prompt_metadata, "project_instruction_bytes", 0) or 0
                ),
                "imported_bytes": int(
                    getattr(
                        prompt_metadata,
                        "project_instruction_imported_bytes",
                        0,
                    )
                    or 0
                ),
                "resolved_bytes": int(
                    getattr(
                        prompt_metadata,
                        "project_instruction_resolved_bytes",
                        0,
                    )
                    or 0
                ),
                "truncated": bool(
                    getattr(
                        prompt_metadata, "project_instructions_truncated", False
                    )
                ),
            },
        )

    skills_chars = int(
        getattr(prompt_metadata, "skills_catalog_chars", 0) or 0
    )
    if skills_chars:
        records = tuple(
            getattr(prompt_metadata, "loaded_skills", ()) or ()
        )
        ledger.record(
            bucket="skills_catalog",
            layer="L4_protocol",
            mechanism="agent_skills_metadata",
            input_chars=0,
            output_chars=skills_chars,
            measure_type="exact",
            ctx={
                "skills": [
                    str(record.get("name", ""))
                    for record in records
                    if not bool(record.get("disable_model_invocation", False))
                ],
            },
        )

    # Profile behavioral: probe the denormalize pipeline on a minimal
    # system message marked with "Commandments" so any gated behavioral
    # modules fire. Delta = after-content minus before-content.
    profile = getattr(client, "profile", None)
    if profile is not None and hasattr(profile, "denormalize_messages"):
        try:
            probe = [{"role": "system", "content": "Commandments\n"}]
            before_chars = len(probe[0]["content"])
            after = profile.denormalize_messages([dict(m) for m in probe])
            after_content = after[0].get("content", "") if after else ""
            after_chars = len(after_content)
            if after_chars != before_chars:
                ledger.record(
                    bucket="profile_behavioral",
                    layer="L1_model_quirks",
                    mechanism=f"{profile.name}_behavioral_suffix",
                    input_chars=before_chars,
                    output_chars=after_chars,
                    measure_type="exact",
                    ctx={"profile": profile.name},
                )
        except Exception as e:
            log.debug("Profile-behavioral cost probe skipped: %s", e)
    if profile is not None:
        preamble = str(getattr(profile, "preamble", "") or "")
        if preamble.strip():
            profile_name = str(getattr(profile, "name", "unknown_profile"))
            ledger.record(
                bucket="profile_preamble",
                layer="L1_model_quirks",
                mechanism=f"{profile_name}_capacity_preamble",
                input_chars=0,
                output_chars=len(preamble),
                measure_type="exact",
                ctx={"profile": profile_name},
            )


def _load_bash_transforms(cfg: Config, *, force_load_all: bool = False):
    """Load the bash transform layers respected by Session.

    Each layer has its own enabled flag. Returns a 6-tuple in this order:
      1. output_control       — task-format output control
                                (pytest --tb=short, condense PASSED)
      2. universal_rewrites   — universal rewrites
                                (pip -q, npm --loglevel=error, make -s)
      3. forbidden_rules      — bash_quirks forbidden-rule list
      4. redirect_rules       — compound-aware dedicated-tool redirects
      5. redactions           — secret-redaction patterns applied to
                                tool output
      6. output_parser        — structured test-run digest parser

    Any element can be None if the corresponding layer is disabled or
    misconfigured. A total load failure is logged at warning level and
    a `transforms_load_failed` event is recorded so a comparison can
    spot the silent-degrade case.
    """
    output_control = None
    universal_rewrites = None
    forbidden_rules = None
    redirect_rules = None
    redactions = None
    output_parser = None
    try:
        from ...bash_quirks import (
            load_output_control,
            load_output_parser,
            load_redactions,
            load_universal_rewrites,
        )
        from ...bash_quirks.transforms import load_forbidden_rules
        from ..command_redirect import load_redirect_rules
        if cfg.bash_transforms_universal_enabled or force_load_all:
            universal_rewrites = load_universal_rewrites()
            if universal_rewrites:
                log.info("Loaded %d universal bash rewrites", len(universal_rewrites))
        else:
            log.info("Universal bash rewrites disabled via config")
        if getattr(cfg, "bash_quirks_forbidden_enabled", True):
            forbidden_rules = load_forbidden_rules()
            if forbidden_rules:
                log.info("Loaded %d bash forbidden patterns", len(forbidden_rules))
        else:
            # This flag controls forbidden command rules.
            log.info("Bash forbidden rules disabled via config")
        redirect_rules = load_redirect_rules()
        if redirect_rules:
            log.info("Loaded %d bash redirect rules", len(redirect_rules))
        # Secret redaction always loads; redactions.toml presence is
        # sufficient to enable. No config gate — redacting tokens is
        # always-on safety.
        redactions = load_redactions()
        if redactions:
            log.info("Loaded %d secret-redaction patterns", len(redactions))
        if cfg.bash_transforms_task_format_enabled or force_load_all:
            _analysis_fmt = cfg.analysis_task_format if hasattr(cfg, "analysis_task_format") else None
            # Real runs resolve "auto" -> the detected runner in the driver
            # (resolve_task_format) before we get here. This is the no-repo
            # fallback (direct tool/test calls): degrade to pytest, matching
            # detect_runner's own marker-less default.
            if _analysis_fmt == "auto":
                _analysis_fmt = "pytest"
            if _analysis_fmt:
                from ...language_quirks import FORMATS_DIR
                fmt_path = FORMATS_DIR / f"{_analysis_fmt}.toml"
                output_control = load_output_control(fmt_path)
                if output_control:
                    log.info("Loaded output control: %s (flag=%r, passed=%r)",
                             _analysis_fmt, output_control.failure_only_flag, output_control.passed_marker)
                if cfg.bash_transforms_structured_output_enabled or force_load_all:
                    output_parser = load_output_parser(fmt_path)
                    if output_parser:
                        log.info("Loaded output parser: %s (summary_fields=%d, per_test=%s)",
                                 _analysis_fmt,
                                 len(output_parser.summary_fields),
                                 output_parser.per_test_regex is not None)
                    else:
                        log.info("Structured output enabled but no [output_parser] block in %s.toml", _analysis_fmt)
        else:
            log.info("Task-format output control disabled via config")
    except Exception as e:
        # Surface the silent-degrade case at warn level. This once used
        # log.debug, so a misconfiguration
        # would silently fall back to raw-bash semantics with no signal
        # in standard log streams.
        log.warning("bash_transforms_load_failed: %s", e)
    return (
        output_control, universal_rewrites, forbidden_rules, redirect_rules,
        redactions, output_parser,
    )
