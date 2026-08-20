#!/usr/bin/env python3
"""Translate a serving overlay TOML to a `llama-server` argv and exec.

Companion to scripts/serve/vllm.py. Schema is shared (see
docs/serving_overlay.md); flag-name mapping is per-runtime. This file
is the single source of truth for llama-server flag mapping.

Usage:
    python3 scripts/serve/llama_server.py configs/runtime/<runtime>.toml
"""
from __future__ import annotations

import os
import sys
import tomllib
from pathlib import Path


SUPPORTED_SCHEMA = {1}
LLAMA_BIN_DEFAULT = "~/.local/bin/llama-server"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_launch_path(raw: str) -> Path:
    path = Path(os.path.expandvars(os.path.expanduser(raw)))
    return path if path.is_absolute() else PROJECT_ROOT / path


def _bail(msg: str) -> None:
    print(f"[serve/llama_server] error: {msg}", file=sys.stderr)
    sys.exit(2)


def _validate_overlay(overlay: dict, path: Path) -> None:
    sv = overlay.get("schema_version")
    if sv not in SUPPORTED_SCHEMA:
        _bail(f"{path}: schema_version {sv!r} not in {SUPPORTED_SCHEMA}")
    launch = overlay.get("launch")
    if not isinstance(launch, dict):
        _bail(f"{path}: missing [launch] block")
    if launch.get("runtime") != "llama_server":
        _bail(f"{path}: [launch].runtime is {launch.get('runtime')!r}, not 'llama_server'")
    for k in ("model_path", "host", "port", "max_model_len"):
        if launch.get(k) is None:
            _bail(f"{path}: [launch].{k} is required")
    sampling = launch.get("sampling") or {}
    required_sampling = {
        "temperature", "top_k", "top_p", "min_p",
        "presence_penalty", "repetition_penalty", "seed",
    }
    missing = required_sampling - set(sampling.keys())
    if missing:
        _bail(f"{path}: [launch.sampling] missing required keys: {sorted(missing)}")
    # Path existence is a warning, not an error (see serve/vllm.py).
    cp = launch.get("chat_template_path")
    if cp and not _resolve_launch_path(cp).is_file():
        print(f"[serve/llama_server] warn: chat_template_path does not exist: {cp}", file=sys.stderr)
    mp = launch.get("model_path")
    if mp and not _resolve_launch_path(mp).exists():
        print(f"[serve/llama_server] warn: model_path does not exist: {mp}", file=sys.stderr)


def _build_argv(overlay: dict) -> list[str]:
    launch = overlay["launch"]
    sampling = launch.get("sampling") or {}
    llama = launch.get("llama_server") or {}

    bin_path = os.path.expanduser(llama.get("binary") or LLAMA_BIN_DEFAULT)
    argv = [bin_path]
    argv += ["--model", str(_resolve_launch_path(launch["model_path"]))]
    argv += ["--host", str(launch["host"])]
    argv += ["--port", str(launch["port"])]
    argv += ["--ctx-size", str(launch["max_model_len"])]
    if launch.get("max_num_seqs") is not None:
        argv += ["--parallel", str(launch["max_num_seqs"])]
    if launch.get("chat_template_path"):
        argv += [
            "--chat-template-file",
            str(_resolve_launch_path(launch["chat_template_path"])),
        ]

    # Sampling — every key explicit, no falling back to server defaults.
    argv += ["--temp", str(sampling["temperature"])]
    argv += ["--top-k", str(sampling["top_k"])]
    argv += ["--top-p", str(sampling["top_p"])]
    argv += ["--min-p", str(sampling["min_p"])]
    argv += ["--presence-penalty", str(sampling["presence_penalty"])]
    argv += ["--repeat-penalty", str(sampling["repetition_penalty"])]   # name-flip
    argv += ["--seed", str(sampling["seed"])]

    # llama-server-specific knobs. Booleans emit flag (or "on") only when true.
    if (n := llama.get("n_cpu_moe")) is not None:
        argv += ["--n-cpu-moe", str(n)]
    if (m := llama.get("cpu_mask")):
        argv += ["--cpu-mask", str(m)]
    if llama.get("cpu_strict") is not None:
        argv += ["--cpu-strict", str(int(bool(llama["cpu_strict"])))]
    if (t := llama.get("threads")) is not None:
        argv += ["--threads", str(t)]
    if (p := llama.get("prio")) is not None:
        argv += ["--prio", str(p)]
    if llama.get("flash_attn"):
        argv += ["--flash-attn", "on"]
    if (k := llama.get("cache_type_k")):
        argv += ["--cache-type-k", str(k)]
    if (k := llama.get("cache_type_v")):
        argv += ["--cache-type-v", str(k)]
    if llama.get("jinja"):
        argv += ["--jinja"]
    if llama.get("no_context_shift"):
        argv += ["--no-context-shift"]
    if (b := llama.get("batch")) is not None:
        argv += ["-b", str(b)]
    if (u := llama.get("ubatch")) is not None:
        argv += ["-ub", str(u)]
    if llama.get("mmap"):
        argv += ["--mmap"]
    if (n := llama.get("n_predict")) is not None:
        argv += ["-n", str(n)]
    if (g := llama.get("n_gpu_layers")) is not None:
        argv += ["--n-gpu-layers", str(g)]
    # MTP / speculative self-decoding (llama.cpp >= PR #22673). Requires an
    # MTP-converted GGUF (MTP head baked in) and parallel==1 (MTP is
    # incompatible with --parallel > 1 and --mmproj).
    if (st := llama.get("spec_type")):
        argv += ["--spec-type", str(st)]
    if (nm := llama.get("spec_draft_n_max")) is not None:
        argv += ["--spec-draft-n-max", str(nm)]

    return argv


def main() -> int:
    args = sys.argv[1:]
    dry = False
    if args and args[0] == "--print":
        dry = True
        args = args[1:]
    if len(args) != 1:
        _bail("usage: serve/llama_server.py [--print] <runtime.toml>")
    path = Path(args[0])
    if not path.is_file():
        _bail(f"runtime file not found: {path}")

    with path.open("rb") as f:
        overlay = tomllib.load(f)
    _validate_overlay(overlay, path)
    argv = _build_argv(overlay)

    print(f"[serve/llama_server] runtime_file={path}", flush=True)
    print(f"[serve/llama_server] argv={argv}", flush=True)

    if dry:
        return 0

    os.environ["YUJ_SERVING_OVERLAY"] = str(path.resolve())
    os.execvp(argv[0], argv)


if __name__ == "__main__":
    sys.exit(main())
