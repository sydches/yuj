"""State-addressed memory for concise/yconcise context strategies.

A WorkingSet replaces the char-budgeted rolling deque used by the
earlier strategies with three state-keyed stores:

  * ``files``       — dict[path → FileSlot]. One entry per unique file
                      the model has touched. Content is re-read from
                      disk on projection when the file is inside cwd,
                      so edits made via any channel (tools or bash)
                      are always current. Entries evict by
                      last-access-turn LRU only when the total
                      rendered payload exceeds the char budget.

  * ``gate_latest`` — dict[cmd_sig → GateSlot]. One entry per unique
                      command signature; newest invocation wins. Six
                      pytest reruns collapse to one entry, not six.

  * ``artifacts``   — deque[Artifact]. Non-repeatable one-off outputs
                      (grep/find/structured projections) with LRU
                      char-budget eviction. Only path the old deque
                      retained, kept because one-off tool results have
                      no natural dedup key.

Nothing in the working set is time-addressed. Projection answers
state-addressed questions directly:
  "what is the current content of file X?"   → files[X]
  "what was the last verdict for gate Y?"    → gate_latest[Y]
  "what was the most recent non-repeat run?" → artifacts[-1]

This module is discovery-prefixed with ``_`` so the context-strategy
registry (``__init__.py``) skips it — it is a shared helper, not a
mode.
"""
from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class FileSlot:
    """One file the model has touched.

    ``content`` is the last snapshot we stored; ``render_from_disk``
    re-reads when the file is inside cwd so the projection reflects
    current on-disk state regardless of whether the model edited via
    the ``edit``/``write`` tools or via a bash sed/redirect.
    """
    path: str               # as the model referenced it (relative or absolute)
    content: str            # snapshot; authoritative only when re-read fails
    last_access_turn: int
    epoch: int = 0          # bumped on write/edit; unused by consumers but handy for debug


@dataclass
class GateSlot:
    """One gate verdict. Newest invocation wins per cmd_signature.

    ``repeat_count`` tracks consecutive invocations that returned
    identical content. It resets to 0 on any content change and
    increments when the newest call produces the same bytes as the
    prior stored slot. Consumers surface ``repeat_count >= 3`` as a
    loop signal in the prompt — replacing the old stateful/compound
    dedup escalation that used to rewrite tool-result bodies with
    warning text. State-addressed equivalent: we keep the content
    clean and surface the loop at the trace/state layer.
    """
    cmd_sig: str            # raw signature string (JSON blob for bash tools)
    cmd_display: str        # human-readable command for rendering
    content: str            # the tool result body
    turn: int
    verdict: str            # "OK" / "FAIL" — from classify_outcome
    first_turn: int = 0     # turn of the oldest call in the current repeat streak
    repeat_count: int = 0   # consecutive identical-content repeats; 0 == fresh


@dataclass
class Artifact:
    """One non-file, non-repeatable tool result."""
    tool_name: str
    args_summary: str
    content: str
    turn: int


@dataclass
class WorkingSet:
    """Container for file/gate/artifact stores. All projection is pure.

    The caller owns the char budget and passes it in at projection
    time; the WorkingSet itself has no hidden limits. That way
    compact/concise/yconcise can share the same store while applying
    different budgets.
    """
    cwd: Path
    files: dict[str, FileSlot] = field(default_factory=dict)
    gate_latest: dict[str, GateSlot] = field(default_factory=dict)
    artifacts: "deque[Artifact]" = field(default_factory=deque)

    # ── ingestion ─────────────────────────────────────────────

    def record_read(self, path: str, content: str, turn: int) -> None:
        """Record a file read. Replaces prior snapshot for this path."""
        key = self._canon(path)
        slot = self.files.get(key)
        if slot is None:
            self.files[key] = FileSlot(path=path, content=content, last_access_turn=turn)
        else:
            slot.path = path
            slot.content = content
            slot.last_access_turn = turn

    def record_mutation(self, path: str, turn: int) -> None:
        """Record that the model wrote/edited a file.

        Bumps the epoch and resets content to the on-disk state. If
        we've never seen the path before, create a slot so the next
        projection ships the file even if it wasn't read first.
        """
        key = self._canon(path)
        fresh = self._read_disk(path) or ""
        slot = self.files.get(key)
        if slot is None:
            self.files[key] = FileSlot(path=path, content=fresh,
                                        last_access_turn=turn, epoch=1)
        else:
            slot.content = fresh
            slot.last_access_turn = turn
            slot.epoch += 1

    def record_gate(self, cmd_sig: str, cmd_display: str, content: str,
                    turn: int, verdict: str) -> None:
        """Record a gate outcome. Newest wins per cmd_sig.

        If the newest invocation produced bytes identical to the
        already-stored slot, bump ``repeat_count`` and keep the
        original ``first_turn``. Any content change resets the
        streak. Projection consumers use ``repeat_count >= 3`` as
        the loop-detected threshold.
        """
        prev = self.gate_latest.get(cmd_sig)
        if prev is not None and prev.content == content:
            prev.turn = turn
            prev.verdict = verdict
            prev.cmd_display = cmd_display
            prev.repeat_count += 1
            return
        self.gate_latest[cmd_sig] = GateSlot(
            cmd_sig=cmd_sig, cmd_display=cmd_display,
            content=content, turn=turn, verdict=verdict,
            first_turn=turn, repeat_count=0,
        )

    def loop_warnings(self, threshold: int = 3) -> list[str]:
        """Return one warning line per gate whose repeat_count >= threshold.

        Empty list means no loops detected. Ordered by first_turn so
        the oldest looping command surfaces first.
        """
        hot = [g for g in self.gate_latest.values() if g.repeat_count >= threshold]
        hot.sort(key=lambda g: g.first_turn)
        return [
            f"`{g.cmd_display}` × {g.repeat_count + 1} calls, "
            f"output unchanged since T{g.first_turn} "
            f"— change approach (re-running will not produce new information)"
            for g in hot
        ]

    def record_artifact(self, tool_name: str, args_summary: str,
                        content: str, turn: int) -> None:
        """Record a non-file, non-gate one-off output."""
        self.artifacts.append(Artifact(
            tool_name=tool_name, args_summary=args_summary,
            content=content, turn=turn,
        ))

    def clear_stale_mutations(self) -> None:
        """Placeholder hook for future mtime-based eviction."""
        return

    # ── projection ────────────────────────────────────────────

    def project_files(self, char_budget: int) -> tuple[str, list[str]]:
        """Render the file working set, LRU-dropping over budget.

        Returns ``(rendered, elided_paths)``. Elided paths are still
        reported to the model so it knows it has seen them before.
        Files inside cwd are re-read from disk on every call; files
        outside cwd fall back to the stored snapshot.
        """
        if not self.files:
            return "", []
        # Changed files first, then newest-access first. Concise modes are
        # action-oriented: once a file has been edited it should stay visible
        # ahead of untouched files when budget pressure forces elision.
        ordered = sorted(self.files.values(),
                         key=lambda s: (s.epoch <= 0, -s.last_access_turn))
        kept: list[FileSlot] = []
        chars_used = 0
        elided: list[str] = []
        for slot in ordered:
            body = self._current_body(slot)
            frame_overhead = len(slot.path) + 64
            if chars_used + len(body) + frame_overhead > char_budget and kept:
                elided.append(slot.path)
                continue
            kept.append(slot)
            chars_used += len(body) + frame_overhead
        blocks: list[str] = []
        for slot in kept:
            body = self._current_body(slot)
            lines = body.splitlines() or [""]
            numbered = "\n".join(f"{i+1}: {ln}" for i, ln in enumerate(lines))
            blocks.append(
                f"--- {slot.path} (last read T{slot.last_access_turn}) ---\n"
                f"{numbered}"
            )
        return "\n".join(blocks), elided

    def project_selected_files(self, paths: list[str], char_budget: int) -> tuple[str, list[str]]:
        """Render only the selected file slots, preserving the requested order."""
        if not paths or char_budget <= 0:
            return "", []
        selected: list[FileSlot] = []
        seen: set[str] = set()
        for path in paths:
            key = self._canon(path)
            if key in seen:
                continue
            slot = self.files.get(key)
            if slot is None:
                continue
            seen.add(key)
            selected.append(slot)

        kept: list[FileSlot] = []
        chars_used = 0
        elided: list[str] = []
        for slot in selected:
            body = self._current_body(slot)
            frame_overhead = len(slot.path) + 64
            if chars_used + len(body) + frame_overhead > char_budget and kept:
                elided.append(slot.path)
                continue
            kept.append(slot)
            chars_used += len(body) + frame_overhead

        blocks: list[str] = []
        for slot in kept:
            body = self._current_body(slot)
            lines = body.splitlines() or [""]
            numbered = "\n".join(f"{i+1}: {ln}" for i, ln in enumerate(lines))
            blocks.append(
                f"--- {slot.path} (last read T{slot.last_access_turn}) ---\n"
                f"{numbered}"
            )
        return "\n".join(blocks), elided

    def project_gates(self) -> str:
        """One line per unique cmd_sig with latest verdict.

        A gate with ``repeat_count >= 3`` renders as a turn-range
        with multiplicity so the model sees its own loop:
            T0-15: ls -la seaborn/ → OK ×16 (unchanged — change approach)
        """
        if not self.gate_latest:
            return ""
        rows = sorted(self.gate_latest.values(), key=lambda g: g.turn)
        lines: list[str] = []
        for g in rows:
            if g.repeat_count >= 3:
                lines.append(
                    f"T{g.first_turn}-{g.turn}: {g.cmd_display} → {g.verdict} "
                    f"×{g.repeat_count + 1} (unchanged — change approach)"
                )
            else:
                lines.append(f"T{g.turn}: {g.cmd_display} → {g.verdict}")
        return "\n".join(lines)

    def most_recent_gate(self) -> GateSlot | None:
        """Most recent gate verdict by turn, regardless of OK/FAIL."""
        if not self.gate_latest:
            return None
        return max(self.gate_latest.values(), key=lambda g: g.turn)

    def last_failing_gate(self) -> GateSlot | None:
        """Most recent gate whose verdict starts with FAIL."""
        fails = [g for g in self.gate_latest.values() if g.verdict.startswith("FAIL")]
        if not fails:
            return None
        return max(fails, key=lambda g: g.turn)

    def project_artifacts(self, char_budget: int) -> str:
        """Newest-first render of the artifacts deque, char-bounded."""
        if not self.artifacts:
            return ""
        kept: list[Artifact] = []
        used = 0
        for a in reversed(self.artifacts):
            if used + len(a.content) > char_budget and kept:
                break
            kept.append(a)
            used += len(a.content)
        # Evict everything beyond what we kept so the deque stays bounded.
        while len(self.artifacts) > len(kept):
            self.artifacts.popleft()
        parts = [
            f"[T{a.turn} {a.tool_name}({a.args_summary})]\n{a.content}"
            for a in reversed(kept)
        ]
        return "\n---\n".join(parts)

    def project_last_gate_payload(self, max_chars: int) -> str:
        """Render the most recent gate's full content (truncated).

        The gates section carries only one-line verdicts; this
        companion method gives the model the raw payload for the
        most recent gate so it can diagnose the blocking failure.
        """
        g = self.most_recent_gate()
        if g is None:
            return ""
        body = g.content
        if len(body) > max_chars:
            body = body[:max_chars - 20] + f"\n[... +{len(g.content) - max_chars + 20} chars]"
        return (
            f"{g.cmd_display} → {g.verdict} (T{g.turn})\n{body}"
        )

    # ── session-boundary helpers ──────────────────────────────

    def seed_from_state_trace(self, state_path: Path, turn: int = 0) -> int:
        """Populate ``files`` from write/edit actions in state.json.trace.

        Called on session resume so the model starts session 2+ with
        the files it wrote in session 1 visible. Reads live disk
        content, not the trace's stored result string.
        Returns number of files seeded.
        """
        if not state_path.is_file():
            return 0
        try:
            state = json.loads(state_path.read_text())
        except (json.JSONDecodeError, OSError):
            return 0
        trace = state.get("trace", [])
        if not trace:
            return 0
        import re
        seen: set[str] = set()
        seeded = 0
        for entry in reversed(trace):
            action = entry.get("action", "")
            if not (action.startswith("write(") or action.startswith("edit(")):
                continue
            m = re.search(r"path='([^']+)'", action)
            if not m:
                continue
            fpath = m.group(1)
            key = self._canon(fpath)
            if key in seen or fpath.endswith("state.json"):
                continue
            seen.add(key)
            body = self._read_disk(fpath)
            if body is None:
                continue
            self.files[key] = FileSlot(path=fpath, content=body,
                                        last_access_turn=turn, epoch=1)
            seeded += 1
        return seeded

    # ── internal ──────────────────────────────────────────────

    def _canon(self, path: str) -> str:
        """Canonicalize a path so 'foo.py' and './foo.py' collapse."""
        p = path.lstrip("./").rstrip("/")
        return p or path

    def _read_disk(self, path: str) -> str | None:
        """Read live file content if it is inside cwd. Return None otherwise."""
        stripped = path.lstrip("./")
        target = (self.cwd / stripped)
        try:
            target_res = target.resolve()
            cwd_res = self.cwd.resolve()
            target_res.relative_to(cwd_res)
        except (ValueError, OSError):
            return None
        if not target_res.is_file():
            return None
        try:
            return target_res.read_text(errors="replace")
        except OSError:
            return None

    def _current_body(self, slot: FileSlot) -> str:
        """Return live disk content when available, else stored snapshot."""
        live = self._read_disk(slot.path)
        if live is not None:
            slot.content = live
            return live
        return slot.content
