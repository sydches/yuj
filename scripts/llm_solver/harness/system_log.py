"""System log — the harness talking about itself, one JSON event per anomaly.

Writes ``<run_dir>/system_log.jsonl`` (beside the ``savings/`` and
``transcripts/`` run outputs) as one JSON record per harness-level
anomaly. Deliberately a SEPARATE file from ``.trace.jsonl``: the trace
records what the model did; the system log records where the harness's
own accounting went wrong — oversized tool results, char-estimate vs
real-token density blowouts, pre-flight context overflows and how they
were resolved (re-clip vs session end).

Event schema (every event, absent numerics zeroed):

    {ts, type, turn, live_pt, estimate_pt, preflight_pt, density,
     ctx, command_shape, quirk_hit, action, task}

  type          "preflight_overflow" | "density_blowout" | "oversized_result"
  live_pt       real token count (server- or tokenizer-measured side)
  estimate_pt   char-estimate side of the same comparison
  preflight_pt  max(live, estimate) projection at a pre-flight gate (else 0)
  density       real / char-estimate ratio (0.0 = unknown)
  ctx           context window size in tokens
  command_shape binary + flags only, arguments stripped (see command_shape())
  quirk_hit     whether a bash_quirks rewrite touched the producing command
  action        "reclipped" | "session_end" | "none" (observational)

``command_shape`` is content-blind by construction: binaries are
basenamed, flag values after ``=`` are stripped, non-flag arguments are
dropped. No paths, no repo content, no model text ever enters an event.

Mirrors the savings-ledger lifecycle (``harness/savings.py``): a
process-level singleton opened by the driver at task start, no-op when
unopened, so hook sites call ``get_system_log().event(...)`` without
threading a reference through the loop.
"""
from __future__ import annotations

import json
import logging
import shlex
import time
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1

# Provenance map cap — tool_call_id → (command_shape, quirk_hit) entries
# kept per session so a pre-flight event can name the command that
# produced the offending message. Oldest entries are evicted first.
_PROVENANCE_CAP = 512

# Results below this size are never density-checked: tokenizing every
# small result would add per-call latency for no anomaly-detection gain.
_DENSITY_MIN_CHARS = 4000


def command_shape(tool_name: str, arguments: dict | None) -> str:
    """Project a tool call to ``binary + flags only, arguments stripped``.

    Non-bash tools reduce to their tool name. Bash commands are split on
    shell separators (|, ||, &&, ;); per segment the binary is basenamed
    and only ``-``-prefixed tokens survive, with any ``=value`` payload
    stripped. Env-var assignments are skipped. Content-blind by
    construction — no paths, no arguments, no output text.
    """
    if tool_name != "bash":
        return tool_name
    cmd = str((arguments or {}).get("cmd", ""))
    if not cmd:
        return "bash"
    shapes: list[str] = []
    import re
    for segment in re.split(r"\|\||&&|\||;", cmd)[:4]:
        try:
            tokens = shlex.split(segment)
        except ValueError:
            tokens = segment.split()
        binary = ""
        flags: list[str] = []
        for tok in tokens:
            if not binary:
                if "=" in tok and not tok.startswith("-"):
                    continue  # env-var assignment prefix
                binary = tok.rsplit("/", 1)[-1]
                continue
            if tok.startswith("-"):
                flags.append(tok.split("=", 1)[0])
        if binary:
            shapes.append(" ".join([binary, *flags]))
    return " | ".join(shapes) if shapes else "bash"


class _NullSystemLog:
    """No-op system log returned by get_system_log() when none is open."""

    def set_task(self, task: str) -> None:
        pass

    def event(self, *args: Any, **kwargs: Any) -> None:
        pass

    def close(self) -> None:
        pass


class SystemLog:
    """Append-only JSONL log of harness self-observations.

    One file per run at ``<run_dir>/system_log.jsonl``, opened in append
    mode so resumed runs extend the record. A write failure logs at
    debug and never blocks the run.
    """

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self._path, "a")
        self._task = ""

    def set_task(self, task: str) -> None:
        """Stamp subsequent events with the current task name."""
        self._task = str(task)

    def event(self, type: str, *, turn: int,
              live_pt: int = 0, estimate_pt: int = 0, preflight_pt: int = 0,
              density: float = 0.0, ctx: int = 0,
              command_shape: str = "", quirk_hit: bool = False,
              action: str = "none") -> None:
        """Append one anomaly event."""
        record = {
            "ts": round(time.time(), 3),
            "schema_version": SCHEMA_VERSION,
            "type": type,
            "task": self._task,
            "turn": int(turn),
            "live_pt": int(live_pt),
            "estimate_pt": int(estimate_pt),
            "preflight_pt": int(preflight_pt),
            "density": round(float(density), 3),
            "ctx": int(ctx),
            "command_shape": command_shape,
            "quirk_hit": bool(quirk_hit),
            "action": action,
        }
        try:
            self._file.write(json.dumps(record, default=str) + "\n")
            self._file.flush()
        except OSError as e:
            log.debug("System log write failed: %s", e)

    def close(self) -> None:
        """Release the file handle. Idempotent."""
        if self._file is not None:
            try:
                self._file.close()
            except OSError:
                pass
            self._file = None


_system_log: SystemLog | _NullSystemLog = _NullSystemLog()


def open_system_log(path: Path) -> SystemLog:
    """Open a system log at path and register as the process singleton."""
    global _system_log
    close_system_log()
    _system_log = SystemLog(path)
    return _system_log


def close_system_log() -> None:
    """Close the singleton and reset to a no-op."""
    global _system_log
    if isinstance(_system_log, SystemLog):
        _system_log.close()
    _system_log = _NullSystemLog()


def get_system_log() -> SystemLog | _NullSystemLog:
    """Return the current system log (no-op variant when none is open)."""
    return _system_log


def record_result_provenance(session, tc_id: str, tool_name: str,
                             arguments: dict | None, *, quirk_hit: bool) -> str:
    """Remember which command produced tool result ``tc_id``.

    Consulted by the pre-flight overflow path to stamp the offending
    message's ``command_shape`` / ``quirk_hit`` onto its event. Returns
    the computed shape so callers can reuse it.
    """
    shape = command_shape(tool_name, arguments)
    prov = getattr(session, "_result_provenance", None)
    if prov is None:
        prov = {}
        session._result_provenance = prov
    prov[tc_id] = (shape, bool(quirk_hit))
    while len(prov) > _PROVENANCE_CAP:
        prov.pop(next(iter(prov)))
    session._last_result_provenance = (shape, bool(quirk_hit))
    return shape


def provenance_for(session, tc_id: str) -> tuple[str, bool]:
    """Look up (command_shape, quirk_hit) for a tool_call_id.

    Falls back to the most recent tool result's provenance, then to
    ("", False) — a pre-flight event with an unknown producer is still
    worth recording.
    """
    prov = getattr(session, "_result_provenance", None) or {}
    if tc_id and tc_id in prov:
        return prov[tc_id]
    return getattr(session, "_last_result_provenance", ("", False))


def observe_tool_result(session, tc_id: str, tool_name: str,
                        arguments: dict | None, result: str, *,
                        quirk_hit: bool, turn: int) -> None:
    """Post-dispatch observation hook for one tool result.

    Always records provenance. When a local tokenizer is bound and the
    result is large, additionally emits:

      - ``density_blowout`` — real token count exceeds TWICE the chars/4
        estimate (the pre-flight projection would undercount this
        message by >2x), even when everything still fits;
      - ``oversized_result`` — a single result larger than half the
        context window, even when the projection still fits.

    Without a tokenizer the real per-message count is unknowable here;
    the turn-level density check at the pre-flight gate (run_step.py)
    still covers the no-tokenizer arms.
    """
    shape = record_result_provenance(session, tc_id, tool_name, arguments,
                                     quirk_hit=quirk_hit)
    if not isinstance(result, str) or len(result) < _DENSITY_MIN_CHARS:
        return
    tokenizer = getattr(session, "_tokenizer", None)
    if tokenizer is None:
        return
    char_est = max(1, len(result) // 4)
    try:
        real = int(tokenizer.count([{"role": "tool", "content": result}]))
    except Exception:
        return
    ctx_size = int(getattr(session.cfg, "context_size", 0) or 0)
    if real > 2 * char_est:
        get_system_log().event(
            "density_blowout", turn=turn,
            live_pt=real, estimate_pt=char_est,
            density=real / char_est, ctx=ctx_size,
            command_shape=shape, quirk_hit=quirk_hit, action="none",
        )
    if ctx_size > 0 and real > ctx_size // 2:
        get_system_log().event(
            "oversized_result", turn=turn,
            live_pt=real, estimate_pt=char_est,
            density=real / char_est, ctx=ctx_size,
            command_shape=shape, quirk_hit=quirk_hit, action="none",
        )
