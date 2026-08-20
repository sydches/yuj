"""Helpers for ``__main__.py`` — extracted to keep entry point under
the 500-line file cap.

Contains:
  - ``_harden_process`` — pre-main process hardening (PR_SET_DUMPABLE,
    RLIMIT_CORE, LD_PRELOAD stripping). Called once at module import
    from ``__main__.py``.
  - Hashing + git + repo-relative-path helpers used by run-metadata.
  - Config-layer introspection (``_config_layers``, ``_is_regime_overlay``,
    ``_detect_regime``).
  - Model-param env-var collection (``_declared_model_params_from_env``).
  - Run-metadata builder + session/server-metadata writers
    (``_build_run_metadata``, ``_write_session_json``,
    ``_write_server_metadata``).

All helper names remain underscore-prefixed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import resource
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from .config import dump_config, PROJECT_ROOT
from .server import LlamaClient

log = logging.getLogger(__name__)


def _harden_process() -> None:
    """Pre-main hardening, ported from Codex (process-hardening/src/lib.rs).

    Three cheap, no-regret defenses applied to THIS process (the harness).
    Closes attack vectors that exist today regardless of bwrap correctness:

      1. PR_SET_DUMPABLE = 0 — prevents ptrace-attach to the harness so a
         sandboxed process cannot read the live conversation history,
         tokenizer state, or environment variables out of the parent.

      2. RLIMIT_CORE = 0 — disables core dumps. A core file would contain
         every prompt + completion + tool result the harness has handled
         to date, written under whatever the kernel's core_pattern says.
         Hard==soft==0 means kernel skips the dump entirely on
         SIGSEGV/SIGABRT instead of redirecting it.

      3. Strip LD_PRELOAD / LD_LIBRARY_PATH / LD_AUDIT et al from
         os.environ. Every bash tool call's eventual python/pytest
         subprocess inherits the harness env. A LD_PRELOAD pointing at
         an attacker .so under cwd would get loaded by every dynamically
         linked subprocess and could rewrite test output. Even the
         bwrap-sandboxed bash inherits the harness's env scope. Removing
         here is upstream of every spawn.

    No-op on non-Linux for the prctl call. All failures are best-effort:
    logged at debug, never raised — the harness must start even on a
    hardened host where these syscalls are blocked.

    This follows the Linux process-hardening pattern used by Codex.
    """
    if sys.platform.startswith("linux"):
        try:
            import ctypes
            libc = ctypes.CDLL("libc.so.6", use_errno=True)
            PR_SET_DUMPABLE = 4
            rc = libc.prctl(PR_SET_DUMPABLE, 0, 0, 0, 0)
            if rc != 0:
                log.debug("PR_SET_DUMPABLE failed: errno=%d", ctypes.get_errno())
        except Exception as e:
            log.debug("PR_SET_DUMPABLE skipped: %s", e)

    try:
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except (ValueError, OSError) as e:
        log.debug("RLIMIT_CORE=(0,0) skipped: %s", e)

    _LD_VARS = (
        "LD_PRELOAD", "LD_LIBRARY_PATH", "LD_AUDIT", "LD_DEBUG",
        "LD_DEBUG_OUTPUT", "LD_PROFILE", "LD_PROFILE_OUTPUT",
        "LD_SHOW_AUXV", "LD_DYNAMIC_WEAK", "LD_BIND_NOW", "LD_BIND_NOT",
    )
    _stripped = [k for k in _LD_VARS if k in os.environ]
    for k in _stripped:
        os.environ.pop(k, None)
    if _stripped:
        log.info("process_hardening: stripped %s from harness env", _stripped)


_MODEL_PARAM_ENV = {
    "YUJ_MODEL_FILE": "model_file",
    "YUJ_MODEL_QUANT": "quant",
    "YUJ_MODEL_NAME": "model_name",
    "YUJ_CHAT_TEMPLATE_FILE": "chat_template_file",
    "YUJ_CHAT_TEMPLATE_SHA256": "chat_template_sha256",
    "YUJ_SAMPLING": "sampling",
    "YUJ_TEMPERATURE": "temperature",
    "YUJ_TOP_K": "top_k",
    "YUJ_TOP_P": "top_p",
    "YUJ_MIN_P": "min_p",
    "YUJ_REPEAT_PENALTY": "repeat_penalty",
    "YUJ_PRESENCE_PENALTY": "presence_penalty",
    "YUJ_SEED": "seed",
    "YUJ_LLAMA_SERVER_PID": "llama_server_pid",
    "YUJ_LLAMA_SERVER_BIN": "llama_server_bin",
}


def _git(args_list: list[str]) -> str:
    try:
        return subprocess.check_output(
            ["git"] + args_list,
            cwd=str(PROJECT_ROOT),
            stderr=subprocess.DEVNULL,
            timeout=2.0,
        ).decode().strip()
    except Exception:
        return ""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str | None:
    try:
        return _sha256_bytes(path.read_bytes())
    except OSError:
        return None


def _canonical_json_sha256(data: dict) -> str:
    raw = json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return _sha256_bytes(raw)


def _repo_relative(path: Path) -> str | None:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return None


def _config_layers(config_paths: list[Path]) -> list[dict]:
    layers: list[dict] = []
    for p in config_paths:
        resolved = Path(p).resolve()
        sha = _sha256_file(resolved)
        entry = {
            "path": str(resolved),
            "repo_relative_path": _repo_relative(resolved),
            "sha256": sha,
        }
        layers.append(entry)
    return layers


def _is_regime_overlay(path: Path) -> bool:
    parts = path.parts
    if path.suffix != ".toml":
        return False
    return any(
        parts[i] == "configs" and parts[i + 1] == "regimes"
        for i in range(0, max(0, len(parts) - 1))
    )


def _detect_regime(layers: list[dict]) -> dict | None:
    overlays = [
        layer for layer in layers
        if _is_regime_overlay(Path(layer["path"]))
    ]
    env_name = os.environ.get("YUJ_REGIME_NAME", "").strip()
    if not overlays and not env_name:
        return None

    selected = overlays[-1] if overlays else {}
    detected_name = Path(selected["path"]).stem if selected else ""
    name = env_name or detected_name
    regime: dict = {
        "name": name,
        "name_source": "env" if env_name else "config_path",
        "catalog_version": os.environ.get(
            "YUJ_REGIME_CATALOG_VERSION",
            "regime-catalog-v0" if selected else "",
        ),
        "overlay_path": selected.get("path"),
        "overlay_repo_relative_path": selected.get("repo_relative_path"),
        "overlay_sha256": selected.get("sha256"),
    }
    if overlays:
        regime["overlay_candidates"] = [
            {
                "path": layer["path"],
                "repo_relative_path": layer.get("repo_relative_path"),
                "sha256": layer.get("sha256"),
                "name": Path(layer["path"]).stem,
            }
            for layer in overlays
        ]
    if env_name and detected_name and env_name != detected_name:
        regime["detected_overlay_name"] = detected_name
        regime["name_mismatch"] = True
    return regime


def _declared_model_params_from_env() -> dict:
    declared: dict = {}
    raw_json = os.environ.get("YUJ_MODEL_PARAMS_JSON", "").strip()
    if raw_json:
        try:
            parsed = json.loads(raw_json)
            if isinstance(parsed, dict):
                declared.update(parsed)
            else:
                declared["YUJ_MODEL_PARAMS_JSON_error"] = (
                    f"expected object, got {type(parsed).__name__}"
                )
        except json.JSONDecodeError as e:
            declared["YUJ_MODEL_PARAMS_JSON_error"] = str(e)

    for env_key, field in _MODEL_PARAM_ENV.items():
        value = os.environ.get(env_key)
        if value not in (None, ""):
            declared[field] = value
    if "SEED" in os.environ and "seed" not in declared:
        declared["seed"] = os.environ["SEED"]
    return declared


def _build_run_metadata(
    *,
    run_dir: Path,
    cfg,
    args: argparse.Namespace,
    overrides: dict,
    started_at: str,
    profile_loaded: str | None = None,
    server_metadata_path: Path | None = None,
    server_metadata_sha256: str | None = None,
) -> dict:
    layers = _config_layers(list(args.config or []))
    config_path_hashes = {
        layer["path"]: layer["sha256"]
        for layer in layers
    }
    model_runtime: dict = {
        "wire_model": cfg.model,
        "profile_name": cfg.profile_name or cfg.model,
        "profile_loaded": profile_loaded,
        "base_url": cfg.base_url,
        "context_size": cfg.context_size,
        "context_fill_ratio": cfg.context_fill_ratio,
        "max_tokens": cfg.max_tokens,
        "max_tokens_fraction": cfg.max_tokens_fraction,
        "tokenizer_id": cfg.tokenizer_id,
        "timeout_connect": cfg.timeout_connect,
        "timeout_read": cfg.timeout_read,
    }
    declared = _declared_model_params_from_env()
    if declared:
        model_runtime["declared_server_params"] = declared
    if server_metadata_sha256:
        model_runtime["server_metadata_sha256"] = server_metadata_sha256
    model_runtime_sha256 = _canonical_json_sha256(model_runtime)

    meta: dict = {
        "run_metadata_schema_version": 1,
        "started_at": started_at,
        "session_started_at": started_at,
        "run_dir": str(run_dir),
        "model": cfg.model,
        "context_mode": args.context,
        "system_prompt_path": str(args.system_prompt.resolve())
                              if args.system_prompt is not None else None,
        "config_paths": [layer["path"] for layer in layers],
        "config_layers": layers,
        "config_path_hashes": config_path_hashes,
        "resolved_config_sha256": _canonical_json_sha256(dump_config(cfg)),
        "git_commit": _git(["rev-parse", "HEAD"]),
        "git_branch": _git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "cli_overrides": {k: v for k, v in overrides.items() if v is not None},
        "model_runtime": model_runtime,
        "model_runtime_sha256": model_runtime_sha256,
    }
    regime = _detect_regime(layers)
    if regime is not None:
        meta["regime"] = regime
    if server_metadata_path is not None:
        meta["server_metadata_path"] = str(server_metadata_path)
    if server_metadata_sha256:
        meta["server_metadata_sha256"] = server_metadata_sha256
    return meta


def _write_session_json(run_dir: Path, metadata: dict) -> None:
    (run_dir / "session.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )


def _write_server_metadata(run_dir: Path, client: LlamaClient) -> tuple[Path | None, str | None]:
    metadata = client.query_server_metadata()
    if not metadata:
        return None, None
    metadata["captured_at"] = datetime.now(timezone.utc).isoformat()
    raw = json.dumps(metadata, indent=2, sort_keys=True, default=str) + "\n"
    path = run_dir / "server_meta.json"
    path.write_text(raw)
    return path, _sha256_bytes(raw.encode("utf-8"))
