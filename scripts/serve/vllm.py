#!/usr/bin/env python3
"""Translate a serving overlay TOML to a `vllm serve` argv and exec.

Reads a runtime TOML, emits the argv it would run (so it
shows up in journald + the server log header), then `os.execvp`s vllm.

See docs/serving_overlay.md for the schema. This file is the single
source of truth for vllm flag mapping. Any new vllm-only knob must be
added to FLAG_MAP_VLLM (or [launch] for cross-runtime knobs) here AND
documented in docs/serving_overlay.md.

Usage:
    python3 scripts/serve/vllm.py configs/runtime/<runtime>.toml
"""
from __future__ import annotations

import json
import os
import sys
import tomllib
from pathlib import Path


SUPPORTED_SCHEMA = {1}
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_launch_path(raw: str) -> Path:
    path = Path(os.path.expandvars(os.path.expanduser(raw)))
    return path if path.is_absolute() else PROJECT_ROOT / path


def _bail(msg: str) -> None:
    print(f"[serve/vllm] error: {msg}", file=sys.stderr)
    sys.exit(2)


def _validate_overlay(overlay: dict, path: Path) -> None:
    sv = overlay.get("schema_version")
    if sv not in SUPPORTED_SCHEMA:
        _bail(f"{path}: schema_version {sv!r} not in {SUPPORTED_SCHEMA}")
    launch = overlay.get("launch")
    if not isinstance(launch, dict):
        _bail(f"{path}: missing [launch] block")
    if launch.get("runtime") != "vllm":
        _bail(f"{path}: [launch].runtime is {launch.get('runtime')!r}, not 'vllm'")
    for k in ("model_path", "served_name", "host", "port", "max_model_len"):
        if launch.get(k) is None:
            _bail(f"{path}: [launch].{k} is required")
    sampling = launch.get("sampling") or {}
    # Every sampling key must be explicit — never rely on server defaults.
    required_sampling = {
        "temperature", "top_k", "top_p", "min_p",
        "presence_penalty", "frequency_penalty", "repetition_penalty", "seed",
    }
    missing = required_sampling - set(sampling.keys())
    if missing:
        _bail(f"{path}: [launch.sampling] missing required keys: {sorted(missing)}")
    # A warning lets a user check the file on a machine that does not hold
    # the model. The server still fails at launch time when the path is wrong.
    cp = launch.get("chat_template_path")
    if cp and not _resolve_launch_path(cp).is_file():
        print(f"[serve/vllm] warn: chat_template_path does not exist: {cp}", file=sys.stderr)
    mp = launch.get("model_path")
    if mp and not _resolve_launch_path(mp).exists():
        print(f"[serve/vllm] warn: model_path does not exist: {mp}", file=sys.stderr)


def _build_argv(overlay: dict) -> list[str]:
    launch = overlay["launch"]
    sampling = launch.get("sampling") or {}
    vllm = overlay.get("launch", {}).get("vllm") if isinstance(overlay.get("launch", {}).get("vllm"), dict) else {}
    # Fallback: TOML allows [launch.vllm] as a sub-table; tomllib returns it under launch["vllm"]
    if not vllm:
        vllm = launch.get("vllm") or {}

    argv = ["vllm", "serve", str(_resolve_launch_path(launch["model_path"]))]
    argv += ["--served-model-name", launch["served_name"]]
    argv += ["--host", launch["host"]]
    argv += ["--port", str(launch["port"])]
    argv += ["--max-model-len", str(launch["max_model_len"])]
    if launch.get("max_num_seqs") is not None:
        argv += ["--max-num-seqs", str(launch["max_num_seqs"])]
    if launch.get("cpu_offload_gb") is not None:
        argv += ["--cpu-offload-gb", str(launch["cpu_offload_gb"])]
    if launch.get("gpu_memory_utilization") is not None:
        argv += ["--gpu-memory-utilization", str(launch["gpu_memory_utilization"])]
    if launch.get("chat_template_path"):
        argv += [
            "--chat-template",
            str(_resolve_launch_path(launch["chat_template_path"])),
        ]

    # Sampling — emit as a single override-generation-config JSON.
    # vLLM uses repetition_penalty (matches our overlay key) and seed at top level.
    argv += ["--override-generation-config", json.dumps(sampling)]

    # vLLM-specific knobs
    if vllm.get("reasoning_parser"):
        argv += ["--reasoning-parser", vllm["reasoning_parser"]]
    if vllm.get("tool_call_parser"):
        argv += ["--tool-call-parser", vllm["tool_call_parser"]]
    if vllm.get("enable_auto_tool_choice"):
        argv += ["--enable-auto-tool-choice"]
    if vllm.get("moe_backend"):
        argv += ["--moe-backend", vllm["moe_backend"]]
    if vllm.get("attention_backend"):
        argv += ["--attention-backend", vllm["attention_backend"]]
    if vllm.get("enable_prefix_caching"):
        argv += ["--enable-prefix-caching"]

    return argv


def main() -> int:
    args = sys.argv[1:]
    dry = False
    if args and args[0] == "--print":
        dry = True
        args = args[1:]
    if len(args) != 1:
        _bail("usage: serve/vllm.py [--print] <runtime.toml>")
    path = Path(args[0])
    if not path.is_file():
        _bail(f"runtime file not found: {path}")

    with path.open("rb") as f:
        overlay = tomllib.load(f)
    _validate_overlay(overlay, path)
    argv = _build_argv(overlay)

    # Echo the resolved argv to stdout so the server log header captures it.
    print(f"[serve/vllm] runtime_file={path}", flush=True)
    print(f"[serve/vllm] argv={argv}", flush=True)

    if dry:
        return 0

    # Make the path available to the child server process. A separately
    # started Yuj process does not inherit or read this value.
    os.environ["YUJ_SERVING_OVERLAY"] = str(path.resolve())

    os.execvp(argv[0], argv)


if __name__ == "__main__":
    sys.exit(main())
