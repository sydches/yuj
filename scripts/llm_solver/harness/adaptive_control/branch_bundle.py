"""Resume-from-turn branch bundle capture for adaptive control.

Off by default. Called from the adaptive pause point after a live detector
fires and before any TOML intervention is applied. The bundle is the concrete
state needed to resume from that point. If a field cannot be serialized, the
writer reports a blocked status instead of inventing a replay.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import shutil
import subprocess
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BRANCH_BUNDLE_VERSION = "branch_bundle_v1"
SNAPSHOT_METHOD = "copytree_repo_snapshot_v1"

_IGNORE_DIRS = {
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".nox",
}


def _root() -> Path:
    return Path(__file__).resolve().parents[4]


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {k: _jsonable(v) for k, v in dataclasses.asdict(value).items()}
    if isinstance(value, deque):
        return [_jsonable(v) for v in value]
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, set):
        return sorted(_jsonable(v) for v in value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _write_json(path: Path, payload: dict | list) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _tree_hashes(root: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.name != "integrity.json":
            out[p.relative_to(root).as_posix()] = _sha256_file(p)
    return out


def _tree_digest(root: Path) -> str:
    h = hashlib.sha256()
    for rel, digest in sorted(_tree_hashes(root).items()):
        h.update(rel.encode())
        h.update(b"\0")
        h.update(digest.encode())
        h.update(b"\n")
    return h.hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(_root()), "rev-parse", "HEAD"],
            text=True,
            timeout=5,
        ).strip()
    except Exception:
        return ""


def _rel_or_abs(path: Path) -> str:
    try:
        return path.resolve().relative_to(_root()).as_posix()
    except Exception:
        return str(path)


def _config_paths(session) -> tuple[Path, ...]:
    paths = getattr(session, "adaptive_control_baseline_config_paths", ()) or ()
    if not paths:
        paths = getattr(getattr(session, "cfg", None), "adaptive_control_baseline_config_paths", ()) or ()
    return tuple(Path(str(p)).resolve() for p in paths if str(p or "").strip())


def branch_point_id(source_run_id: str, instance_id: str, slot: int,
                    signal_id: str, detector_version: str, scout_policy_id: str) -> str:
    raw = "|".join([
        source_run_id,
        instance_id,
        str(int(slot)),
        signal_id,
        detector_version,
        scout_policy_id,
    ])
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def _ignore(_dir: str, names: list[str]) -> set[str]:
    return {n for n in names if n in _IGNORE_DIRS or n.endswith(".pyc")}


def _copy_repo_snapshot(src: Path, dst: Path) -> str:
    shutil.copytree(src, dst, ignore=_ignore)
    return _tree_digest(dst)


def _prefix_events(session, branch_slot: int) -> list[dict]:
    out = []
    for event in getattr(session, "_trace_events", []) or []:
        turn = event.get("turn_number")
        if turn is None or int(turn) <= int(branch_slot):
            out.append(event)
    return out


def _copy_configs(configs_dir: Path, baseline_paths: tuple[Path, ...]) -> dict[str, str]:
    copied: dict[str, str] = {}
    configs_dir.mkdir(parents=True, exist_ok=True)
    for i, src in enumerate(baseline_paths, start=1):
        if not src.is_file():
            continue
        dst = configs_dir / f"baseline_{i:02d}{src.suffix or '.toml'}"
        shutil.copy2(src, dst)
        copied[str(src)] = dst.relative_to(configs_dir.parent).as_posix()
    return copied


def _write_text_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")


def _source_run_id(session, cfg) -> str:
    return (
        getattr(cfg, "adaptive_control_branch_bundle_source_run_id", "")
        or getattr(cfg, "adaptive_control_source_wave_id", "")
        or getattr(session, "wave_id", "")
        or getattr(session, "attempt_id", "")
        or "unknown_source_run"
    )


def _source_instance_id(session, cfg) -> str:
    return (
        getattr(session, "instance_id", "")
        or getattr(cfg, "adaptive_control_source_instance_id", "")
        or getattr(cfg, "instance_id", "")
    )


def _source_run_dir(session, cfg) -> str:
    configured = str(getattr(cfg, "adaptive_control_source_run_dir", "") or "").strip()
    if configured:
        return configured
    existing = str(getattr(session, "source_run_dir", "") or "").strip()
    if existing:
        return existing
    cwd = str(getattr(session, "cwd", "") or "").strip()
    if cwd:
        return str(Path(cwd).parent)
    return ""


def maybe_capture(session, decision, turn: int, boundary_type: str) -> dict[str, str]:
    cfg = getattr(session, "cfg", None)
    if not getattr(cfg, "adaptive_control_branch_bundle_enabled", False):
        return {"status": "disabled", "path": "", "branch_point_id": "", "reason": ""}
    root_raw = str(getattr(cfg, "adaptive_control_branch_bundle_root", "") or "").strip()
    if not root_raw:
        return {"status": "blocked", "path": "", "branch_point_id": "", "reason": "bundle_root_missing"}
    if not decision or decision.diagnosis_status != "active_confirmed":
        return {"status": "not_applicable", "path": "", "branch_point_id": "", "reason": ""}

    instance_id = _source_instance_id(session, cfg)
    signal_id = decision.active_hurdle_mode or decision.detector_id
    source_run_id = _source_run_id(session, cfg)
    detector_version = str(getattr(cfg, "adaptive_control_detector_version", "") or "")
    scout_policy_id = str(getattr(cfg, "adaptive_control_policy_version", "") or "")
    bpid = branch_point_id(source_run_id, instance_id, int(turn), signal_id,
                           detector_version, scout_policy_id)
    seen = getattr(session, "_adaptive_control_branch_bundle_ids", set())
    limit = max(1, int(getattr(cfg, "adaptive_control_branch_bundle_max_per_attempt", 1) or 1))
    if bpid in seen:
        return {"status": "exists", "path": "", "branch_point_id": bpid, "reason": "already_captured"}
    if len(seen) >= limit:
        return {"status": "blocked", "path": "", "branch_point_id": bpid, "reason": "bundle_cap_reached"}

    bundle = Path(root_raw).expanduser().resolve() / bpid
    repo_dir = Path(getattr(session, "cwd", "")).resolve()
    if repo_dir == bundle or str(bundle).startswith(str(repo_dir) + os.sep):
        return {"status": "blocked", "path": str(bundle), "branch_point_id": bpid,
                "reason": "bundle_root_inside_repo"}
    if (bundle / "integrity.json").is_file():
        seen.add(bpid)
        setattr(session, "_adaptive_control_branch_bundle_ids", seen)
        return {"status": "created", "path": str(bundle), "branch_point_id": bpid,
                "reason": "existing_integrity"}

    tmp = bundle.with_name(f".{bundle.name}.tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    try:
        prefix = _prefix_events(session, int(turn))
        _write_text_jsonl(tmp / "prefix.trace.jsonl", prefix)
        _write_json(tmp / "context_messages.json", session.context.get_messages())
        worktree_digest = _copy_repo_snapshot(repo_dir, tmp / "repo_snapshot")
        _write_json(tmp / "guard_state.json", _jsonable(getattr(session, "_guards", {})))
        _write_json(tmp / "runtime_state.json", {
            "session_number": getattr(session, "_session_number", ""),
            "next_slot_index": int(turn) + 1,
            "trace_path": str(getattr(session, "_trace_path", "") or ""),
            "state_path": str(getattr(session, "_state_path", "") or ""),
            "sink_counter": getattr(session, "_sink_counter", 0),
            "boundary_type": boundary_type,
            "snapshot_method": SNAPSHOT_METHOD,
        })
        _write_json(tmp / "adaptive_state.json", {
            "budget": _jsonable(getattr(session, "_adaptive_control_budget", {})),
            "pending_watch": _jsonable(getattr(session, "_adaptive_control_pending_watch", {})),
            "episode_machine": _jsonable(getattr(session, "_adaptive_control_episode_machine", {})),
        })
        state_path = Path(str(getattr(session, "_state_path", "") or ""))
        if state_path.is_file():
            shutil.copy2(state_path, tmp / "solver_state.json")
        else:
            _write_json(tmp / "solver_state.json", {})
        sunk = repo_dir / ".tool_output"
        if sunk.is_dir():
            shutil.copytree(sunk, tmp / "sunk_outputs", ignore=_ignore)
        else:
            (tmp / "sunk_outputs").mkdir()
        baseline_paths = _config_paths(session)
        copied = _copy_configs(tmp / "configs", baseline_paths)
        baseline_hashes = {str(p): _sha256_file(p) for p in baseline_paths if p.is_file()}
        manifest = {
            "branch_bundle_version": BRANCH_BUNDLE_VERSION,
            "source_run_id": source_run_id,
            "source_run_dir": _source_run_dir(session, cfg),
            "instance_id": instance_id,
            "branch_point_id": bpid,
            "branch_slot": int(turn),
            "online_signal_id": signal_id,
            "detector_version": detector_version,
            "detector_status": decision.detector_status,
            "detector_evidence_refs": ";".join(decision.basis_refs),
            "baseline_config_path": ";".join(str(p) for p in baseline_paths),
            "baseline_config_sha256": ";".join(baseline_hashes.values()),
            "baseline_config_copies": copied,
            "context_mode": type(getattr(session, "context", object())).__name__,
            "model_profile": getattr(cfg, "profile_name", "") or getattr(cfg, "model", ""),
            "harness_commit": _git_commit(),
            "policy_version": scout_policy_id,
            "watch_window_turns": int(getattr(cfg, "adaptive_control_watch_window_turns", 5) or 5),
            "watch_policy_id": getattr(cfg, "adaptive_control_branch_watch_policy_id",
                                       "prefix_rewind_watch_v1"),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "produced_at_detector_fire": True,
            "reconstructed_later": False,
            "future_evidence_used": False,
        }
        _write_json(tmp / "branch_manifest.json", manifest)
        integrity = {
            "branch_bundle_version": BRANCH_BUNDLE_VERSION,
            "worktree_snapshot_digest": worktree_digest,
            "file_sha256": _tree_hashes(tmp),
        }
        _write_json(tmp / "integrity.json", integrity)
        if bundle.exists():
            shutil.rmtree(bundle)
        tmp.rename(bundle)
    except Exception as exc:  # noqa: BLE001 - must not break live runs
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
        return {"status": "blocked", "path": str(bundle), "branch_point_id": bpid,
                "reason": f"bundle_write_failed:{type(exc).__name__}"}

    seen.add(bpid)
    setattr(session, "_adaptive_control_branch_bundle_ids", seen)
    return {"status": "created", "path": _rel_or_abs(bundle), "branch_point_id": bpid, "reason": ""}
