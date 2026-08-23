"""Replay client: serve recorded assistant turns instead of calling a server.

Spec: docs/replay_mode_spec.md. The verbatim transcript is the source of
truth; nothing else is consulted. `chat()` returns the recorded TurnResult
for the next recorded turn, strictly in order (prefix-only by construction).

Fidelity gate: before serving turn k+1, the tool-result messages the live
loop appended after executing turn k must match what the recording shows in
turn k+1's request payload. A mismatch means a replayed command produced a
different result than the original run — the replay STOPS with a recorded
divergence reason rather than silently continuing from a state that is not
the original.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from .types import ToolCall, TurnResult, Usage

log = logging.getLogger(__name__)

_HEADER_RE = re.compile(r"^=== turn (\d+) (input|output) ===$", re.MULTILINE)

REPLAY_FINISH_REASON_STOP_TURN = "replay_stop_turn"
REPLAY_FINISH_REASON_EXHAUSTED = "replay_recording_exhausted"


class ReplayDivergence(RuntimeError):
    """A replayed tool execution differed from the recording."""


# Volatile-output normalization (docs/replay_mode_spec.md): a NAMED list of
# per-container/per-wall-clock fields that re-execute non-identically on a
# faithful replay. Add an entry only after the fidelity check finds that
# changing value in a real replay. Apply each rule to both sides.
VOLATILE_NORMALIZATION_VERSION = "replay_volatile_norm_v14"
_VOLATILE_PATTERNS = (
    # Docker overlayfs device ID in stat output.
    (re.compile(r"Device: [0-9a-fA-F]+h/\d+d"), "Device: <volatile>"),
    # stat timestamp lines: wall-clock of the replay, not model-relevant
    # state (files created during the run carry replay-time stamps)
    (re.compile(r"^\s*(Access|Modify|Change|Birth): .*$", re.MULTILINE),
     r"\1: <volatile>"),
    # sed -i random temporary name in its failure message. Limit the rule to
    # the error message so ordinary content with the same shape still compares.
    (re.compile(r"(couldn't open temporary file \S*/sed)[A-Za-z0-9]{6}"),
     r"\1<volatile>"),
    # diff/diff -u header times: ---/+++ lines carry each file's time.
    (re.compile(r"^([-+]{3} [^\t\n]+\t).*$", re.MULTILINE), r"\1<volatile>"),
    # Python temporary names: tmp + exactly 8 random [a-z0-9_] characters.
    # Fixed names with this shape are equal on both sides, so the rule changes
    # only names that differ.
    (re.compile(r"\btmp[a-z0-9_]{8}\b"), "tmp<volatile>"),
    # ctypes CDLL repr handle: the dlopen handle and object address vary.
    (re.compile(r"(<CDLL '[^']+', handle )(?:0x)?[0-9a-fA-F]+( at 0x)(?:[0-9a-fA-F]+|XXXX)(>)"),
     r"\1<volatile>\2<volatile>\3"),
    # ls -la "total N": file-system block accounting varies by overlay.
    (re.compile(r"^total \d+$", re.MULTILINE), "total <volatile>"),
    # /proc/*/fd pipe inode IDs are assigned per process.
    (re.compile(r"pipe:\[\d+\]"), "pipe:[<volatile>]"),
    # /proc/self/fd listings put the current process ID in symlink targets.
    (re.compile(r"/proc/\d+/fd\b"), "/proc/<volatile>/fd"),
    # SIGPIPE race: `cmd | head` exits 141 or 0 depending on pipe-close
    # timing. Only 141 and an absent status compare as equal.
    (re.compile(r"\n*\[exit code: 141\]\s*$"), ""),
    # Network retry and timing text may vary under --network=none.
    (re.compile(r"^.*(?:HTTPSConnectionPool|Max retries exceeded|Retrying \(Retry).*$",
                re.MULTILINE), "<volatile:network>"),
    # pytest wall-clock durations
    (re.compile(r"\bin \d+\.\d+s\b"), "in <volatile>s"),
    # process id printed by the task
    (re.compile(r"^(PID:\s*)\d+\s*$", re.MULTILINE), r"\1<volatile>"),
    # containerd snapshot ids inside overlay lowerdir/upperdir mount paths
    (re.compile(r"(/snapshots/)\d+"), r"\1<volatile>"),
    # chown -R errors can arrive in file-system order. Normalize the path.
    (re.compile(r"^chown: .*$", re.MULTILINE), "chown: <volatile>"),
    # git stash-pop messages embed a generated commit hash
    (re.compile(r"(Dropped refs/stash@\{\d+\} \()[0-9a-f]+\)"), r"\1<volatile>)"),
    # test-runner seed, hash-randomization, and wall-clock lines
    (re.compile(r"^random seed:\s+.*$", re.MULTILINE), "random seed: <volatile>"),
    (re.compile(r"^hash randomization:.*$", re.MULTILINE), "hash randomization: <volatile>"),
    (re.compile(r"\bin \d+\.\d+ seconds\b"), "in <volatile> seconds"),
    # ls -la directory sizes depend on the overlay contents. File rows stay
    # strict because file sizes carry task state.
    (re.compile(r"^(d[\w-]{9,10}\s+\d+\s+\S+\s+\S+)\s+\d+(\s+.+)$",
                re.MULTILINE), r"\1 <volatile>\2"),
    # setuptools_scm version hash suffix from runtime Git state
    (re.compile(r"(Sphinx v[\d.]+)\+/[0-9a-f]+"), r"\1+/<volatile>"),
    # setuptools_scm local date suffix from the replay clock
    (re.compile(r"\.d\d{8}\b"), ".d<volatile-date>"),
    # dd timing and throughput; byte counts stay strict
    (re.compile(r"\b\d+(?:\.\d+)?(?:e-?\d+)? s, [\d.]+ [KMGT]?B/s"),
     "<volatile> s, <volatile>"),
    # one-time font-cache banner from a fresh cache
    (re.compile(r"^Matplotlib is building the font cache; this may take a moment\.\n?",
                re.MULTILINE), ""),
    # summary annotations may carry content-length counts that vary with
    # traversal order inside truncated output
    (re.compile(r"\[\.\.\. \d+ chars omitted \.\.\.\]"),
     "[... <volatile> chars omitted ...]"),
    (re.compile(r"^\s*\.\.\. \[×\d+ similar lines\]$", re.MULTILINE),
     "<volatile:similar-lines>"),
    # Sphinx stamps the build date into generated LaTeX.
    (re.compile(r"\\date\{[A-Z][a-z]{2} \d{1,2}, \d{4}\}"),
     r"\\date{<volatile>}"),
    # Test-runner Date: ISO wall-clock stamp.
    (re.compile(r"\bDate: \d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\b"),
     "Date: <volatile>"),
    # stat Size field on directory lines only. Keep file sizes strict.
    (re.compile(r"(Size: )\d+(\s.*directory)"), r"\1<volatile>\2"),
    # os.stat_result carries the container overlay device number.
    (re.compile(r"(st_dev=)\d+"), r"\1<volatile>"),
    # GNU stat Inode is container overlay metadata.
    (re.compile(r"(Inode:\s*)\d+"), r"\1<volatile>"),
    # df overlay disk use varies with the host. Keep the mount path strict.
    (re.compile(r"^(overlay\s+)\d[\w.]*\s+\d[\w.]*\s+\d[\w.]*\s+\d+%(\s+.+)$",
                re.MULTILINE),
     r"\1<volatile> <volatile> <volatile> <volatile>\2"),
    # patch creates a temporary output suffix. Keep the error and target path
    # strict.
    (re.compile(r"(patch: \*\*\*\* Can't create temporary file \S+\.)[A-Za-z0-9]{6,}(\s+: Permission denied)"),
     r"\1<volatile>\2"),
    # Python may exit 120 instead of 1 when stdout flush fails at shutdown.
    # Keep every other exit status strict.
    (re.compile(r"\[exit code: 120\]"), "[exit code: 1]"),
)


_SET_REPR_RE = re.compile(r"\{('[^'{}]*'(?:, '[^'{}]*')+)\}")


def _sort_set_reprs(text: str) -> str:
    """Sort quoted items in set-shaped reprs.

    Python set order can change with ``PYTHONHASHSEED``. Keep item content
    strict while accepting order-only changes.
    """
    def _fix(m):
        items = sorted(s.strip() for s in m.group(1).split(","))
        return "{" + ", ".join(items) + "}"
    return _SET_REPR_RE.sub(_fix, text)


def _collapse_volatile_runs(text: str) -> str:
    """Collapse ADJACENT duplicate lines that contain a <volatile> marker:
    after canonicalization, truncation-dependent repeat counts (chown -R
    error walls under head/tail summary cuts) compare as one line. Lines
    without a volatile marker are never collapsed."""
    out: list[str] = []
    for line in text.splitlines(keepends=False):
        if line == "<volatile:similar-lines>" and out and "<volatile" in out[-1]:
            continue  # harness dedup annotation about an already-volatile wall
        if out and "<volatile" in line and out[-1] == line:
            continue
        out.append(line)
    tail = "\n" if text.endswith("\n") else ""
    return "\n".join(out) + tail


def normalize_volatile(text: str) -> str:
    for pat, repl in _VOLATILE_PATTERNS:
        text = pat.sub(repl, text)
    return _collapse_volatile_runs(_sort_set_reprs(text))


def ordering_only_equal(a: str, b: str) -> bool:
    """Accept identical lines in a different order as a last check.

    The caller logs this choice. Any content difference still fails.
    """
    la, lb = a.splitlines(), b.splitlines()
    return len(la) == len(lb) and sorted(la) == sorted(lb)


def parse_transcript_turns(path: Path) -> dict[tuple[int, str], str]:
    """{(turn_no, 'input'|'output'): body_text} for a verbatim transcript."""
    text = Path(path).read_text()
    headers = list(_HEADER_RE.finditer(text))
    if not headers:
        raise ValueError(f"no turn markers in transcript at {path}")
    bodies: dict[tuple[int, str], str] = {}
    for i, m in enumerate(headers):
        start = m.end()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        bodies[(int(m.group(1)), m.group(2))] = text[start:end].strip()
    return bodies


def _tool_tail(messages: list[dict]) -> list[str]:
    """Contents of the trailing tool-role messages (the results of the most
    recently executed turn). These are the only live-produced texts a replay
    must reproduce; assistant texts come from the recording itself."""
    tail: list[str] = []
    for m in reversed(messages):
        if (m.get("role") or "") == "tool":
            tail.append(str(m.get("content") or ""))
        else:
            break
    tail.reverse()
    return tail


class ReplayClient:
    """Same .chat() contract as the live client; turns come from a recording.

    TURN NUMBERING (canonical = trace): every replay-facing turn number —
    stop_turn, fidelity events, capture slots — is a TRACE turn_number
    (0-based, the doctrine source of truth). Transcript files are 1-based;
    the mapping (transcript_no = trace_turn + 1) is internal to this client
    and appears nowhere else.

    stop_turn >= 0 given: after serving the recorded turn whose TRACE number
    equals stop_turn, the next chat() returns finish_reason=
    REPLAY_FINISH_REASON_STOP_TURN — the loop ends there (or a handover
    layer swaps in the live client before that call).
    """

    is_replay = True

    def __init__(self, transcript_path: str | Path, stop_turn: int = 0,
                 strict_fidelity: bool = True,
                 source_trace_path: str | Path | None = None):
        # A run may leave repo.pre_seg_N.log segments and repo.log. Read
        # every segment in order and number all turns as one conversation.
        tp = Path(transcript_path)
        segments = []
        stem = tp.stem  # e.g. 'repo'
        n = 1
        while (tp.parent / f"{stem}.pre_seg_{n}.log").is_file():
            segments.append(tp.parent / f"{stem}.pre_seg_{n}.log")
            n += 1
        segments.append(tp)
        if len(segments) == 1:
            self._bodies = parse_transcript_turns(tp)
        else:
            bodies: dict = {}
            offset = 0
            for seg in segments:
                seg_bodies = parse_transcript_turns(seg)
                max_no = 0
                for (no, kind), body in seg_bodies.items():
                    bodies[(no + offset, kind)] = body
                    max_no = max(max_no, no)
                offset += max_no
            self._bodies = bodies
            log.info("replay source is segmented: %d segments, %d turns total",
                     len(segments), offset)
        self._turns = sorted({t for t, _k in self._bodies})
        if not self._turns:
            raise ValueError("recording has no turns")
        self._idx = 0
        log.info("replay client: %s turns recorded, volatile list %s",
                 len(self._turns), VOLATILE_NORMALIZATION_VERSION)
        self.stop_turn = int(stop_turn or 0)
        self.strict_fidelity = strict_fidelity
        self.divergence: dict | None = None
        self.served_turns = 0
        # normalization census: how much of the comparison is normalized
        # rather than exact (quantified honesty; logged at replay end)
        self.verified_turns = 0
        self.normalized_turns = 0
        self.normalized_turn_list: list[int] = []
        self.ordering_accepts = 0
        # trace-level fidelity (the spec's gate): recorded tool_call events
        # keyed by turn_number — compare executed command + result summary,
        # not the rendered request (windowing/compaction state-dependent)
        self._trace_events: dict[int, dict] = {}
        self.process_events: list[dict] = []
        self.subagent_events: list[dict] = []
        self.source_trace_path = (
            Path(source_trace_path) if source_trace_path is not None else None
        )
        if self.source_trace_path is not None and self.source_trace_path.is_file():
            for line in self.source_trace_path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if ev.get("event") == "tool_call":
                    self._trace_events[int(ev.get("turn_number", -1) or -1)] = ev
                elif ev.get("event") in {"proc_start", "proc_poll", "proc_kill"}:
                    self.process_events.append(ev)
                elif ev.get("event") == "subagent":
                    self.subagent_events.append(ev)

    # -- helpers -------------------------------------------------------------

    def _recorded_request_tool_tail(self, turn_no: int) -> list[str] | None:
        body = self._bodies.get((turn_no, "input"))
        if not body:
            return None
        try:
            msgs = json.loads(body).get("messages") or []
        except json.JSONDecodeError:
            return None
        return _tool_tail(msgs)

    def _check_fidelity(self, live_messages: list[dict], turn_no: int) -> None:
        """Compare the live trailing tool results against the recording's
        request payload for this turn. Only harness-produced texts compare;
        assistant content is recording-sourced and identical by construction."""
        recorded = self._recorded_request_tool_tail(turn_no)
        if recorded is None:
            return  # nothing to compare against (first turn / parse gap)
        live = _tool_tail(live_messages)
        if live == recorded:
            return
        self.divergence = {
            "turn": turn_no,
            "live_tail_n": len(live),
            "recorded_tail_n": len(recorded),
            "first_mismatch": next(
                (i for i, (a, b) in enumerate(zip(live, recorded)) if a != b),
                min(len(live), len(recorded)),
            ),
        }
        msg = (f"replay divergence at recorded turn {turn_no}: "
               f"live tool results differ from recording "
               f"(first mismatch index {self.divergence['first_mismatch']})")
        if self.strict_fidelity:
            raise ReplayDivergence(msg)
        log.warning("%s (continuing: strict_fidelity=false)", msg)

    def _turn_result(self, turn_no: int) -> TurnResult:
        body = self._bodies.get((turn_no, "output"))
        if body is None:
            raise ReplayDivergence(f"recording has no output for turn {turn_no}")
        resp = json.loads(body)
        choices = resp.get("choices") or []
        if not choices:
            raise ReplayDivergence(f"recorded turn {turn_no} output has no choices")
        msg = choices[0].get("message") or {}
        tool_calls = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function") or {}
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {"_raw": fn.get("arguments") or ""}
            tool_calls.append(ToolCall(
                id=str(tc.get("id") or ""), name=str(fn.get("name") or ""),
                arguments=args))
        usage = resp.get("usage") or {}
        return TurnResult(
            content=msg.get("content"),
            tool_calls=tool_calls,
            finish_reason=str(choices[0].get("finish_reason") or "stop"),
            usage=Usage(prompt_tokens=int(usage.get("prompt_tokens") or 0),
                        completion_tokens=int(usage.get("completion_tokens") or 0)),
        )

    # -- contract ------------------------------------------------------------

    def _log_census(self, why: str) -> None:
        log.info("replay census (%s): verified=%d normalized=%d (%.0f%%) "
                 "ordering_accepts=%d list=%s normalized_turns=%s",
                 why, self.verified_turns, self.normalized_turns,
                 100.0 * self.normalized_turns / max(1, self.verified_turns),
                 self.ordering_accepts, VOLATILE_NORMALIZATION_VERSION,
                 ",".join(map(str, self.normalized_turn_list)) or "-")

    # ── verbatim transcript ─────────────────────────────────────────────
    # The replay writes the turns it serves into the same diary format the
    # live client uses, flushed per turn, so a replayed run leaves ONE
    # complete conversation record no matter where it is killed. At
    # handover the live client continues the same file (append mode).
    _transcript_file = None
    _transcript_path = None
    _transcript_call_n = 0

    def set_transcript(self, path, append: bool = False) -> None:
        self.close_transcript()
        self._transcript_path = path
        self._transcript_call_n = 0
        if path is not None:
            from pathlib import Path as _P
            _P(path).parent.mkdir(parents=True, exist_ok=True)
            self._transcript_file = open(path, "a" if append else "w")

    def close_transcript(self) -> None:
        if self._transcript_file is not None:
            try:
                self._transcript_file.close()
            except Exception:
                pass
            self._transcript_file = None

    def _write_transcript(self, marker: str, body: str) -> None:
        if self._transcript_file is None:
            return
        self._transcript_file.write(f"=== {marker} ===\n")
        self._transcript_file.write(body)
        if not body.endswith("\n"):
            self._transcript_file.write("\n")
        self._transcript_file.flush()

    def chat(self, messages: list[dict], tools: list[dict], turn: int = 0) -> TurnResult:
        if self._idx >= len(self._turns):
            self._log_census("recording_exhausted")
            return TurnResult(content=None, tool_calls=[],
                              finish_reason=REPLAY_FINISH_REASON_EXHAUSTED,
                              usage=Usage(0, 0))
        turn_no = self._turns[self._idx]
        # transcript is 1-based; stop_turn is trace numbering (0-based)
        if self.stop_turn and (turn_no - 1) > self.stop_turn:
            self._log_census("stop_turn")
            return TurnResult(content=None, tool_calls=[],
                              finish_reason=REPLAY_FINISH_REASON_STOP_TURN,
                              usage=Usage(0, 0))
        result = self._turn_result(turn_no)
        if self._transcript_file is not None:
            self._transcript_call_n += 1
            n = self._transcript_call_n
            self._write_transcript(
                f"turn {n:03d} input",
                json.dumps({"messages": messages, "tools": tools}))
            out = self._bodies.get((turn_no, "output"), "")
            self._write_transcript(f"turn {n:03d} output", out)
        self._idx += 1
        self.served_turns += 1
        return result

    def verify_executed_turn(self, live_event: dict) -> None:
        """Trace-level fidelity gate for a just-executed tool_call event.

        New traces compare command identity plus output_sha256. Legacy traces
        without output hashes fall back to result_summary snippets.
        Divergence stops the replay.
        """
        if not self._trace_events:
            return
        turn = int(live_event.get("turn_number", -1) or -1)
        if turn < 0:
            return  # boundary events without turn alignment cannot compare
        rec = self._trace_events.get(turn)
        if rec is None:
            return
        self.verified_turns += 1
        turn_normalized = False
        for field in ("tool_name", "args_summary"):
            live_raw = str(live_event.get(field) or "")
            rec_raw = str(rec.get(field) or "")
            live_v = normalize_volatile(live_raw)
            rec_v = normalize_volatile(rec_raw)
            if not turn_normalized and (live_v != live_raw or rec_v != rec_raw):
                turn_normalized = True
                self.normalized_turns += 1
                self.normalized_turn_list.append(turn)
            if live_v != rec_v:
                mismatch_at = next((i for i, (a, b) in
                                    enumerate(zip(live_v, rec_v)) if a != b),
                                   min(len(live_v), len(rec_v)))
                lo = max(0, mismatch_at - 80)
                self.divergence = {"turn": turn, "field": field,
                                   "live": live_v[lo:mismatch_at + 120],
                                   "recorded": rec_v[lo:mismatch_at + 120]}
                log.error("replay divergence turn %d field=%s\n  live: %r\n  rec:  %r",
                          turn, field, self.divergence["live"],
                          self.divergence["recorded"])
                msg = (f"replay divergence at recorded turn {turn}: "
                       f"{field} differs from recording")
                if self.strict_fidelity:
                    raise ReplayDivergence(msg)
                log.warning("%s (continuing: strict_fidelity=false)", msg)
                return
        if live_event.get("output_sha256") and rec.get("output_sha256"):
            live_hash = str(live_event.get("output_sha256") or "")
            rec_hash = str(rec.get("output_sha256") or "")
            if live_hash != rec_hash:
                self.divergence = {
                    "turn": turn,
                    "field": "output_sha256",
                    "live": live_hash,
                    "recorded": rec_hash,
                }
                log.error(
                    "replay divergence turn %d field=output_sha256\n  live: %s\n  rec:  %s",
                    turn, live_hash, rec_hash,
                )
                msg = (
                    f"replay divergence at recorded turn {turn}: "
                    "output_sha256 differs from recording"
                )
                if self.strict_fidelity:
                    raise ReplayDivergence(msg)
                log.warning("%s (continuing: strict_fidelity=false)", msg)
            return

        field = "result_summary"
        live_raw = str(live_event.get(field) or "")
        rec_raw = str(rec.get(field) or "")
        live_v = normalize_volatile(live_raw)
        rec_v = normalize_volatile(rec_raw)
        if not turn_normalized and (live_v != live_raw or rec_v != rec_raw):
            turn_normalized = True
            self.normalized_turns += 1
            self.normalized_turn_list.append(turn)
        if live_v != rec_v and ordering_only_equal(live_v, rec_v):
            self.ordering_accepts += 1
            log.info("replay turn %d: ordering-only difference accepted "
                     "(identical line multiset)", turn)
            return
        if live_v != rec_v:
            mismatch_at = next((i for i, (a, b) in
                                enumerate(zip(live_v, rec_v)) if a != b),
                               min(len(live_v), len(rec_v)))
            lo = max(0, mismatch_at - 80)
            self.divergence = {"turn": turn, "field": field,
                               "live": live_v[lo:mismatch_at + 120],
                               "recorded": rec_v[lo:mismatch_at + 120]}
            log.error("replay divergence turn %d field=%s\n  live: %r\n  rec:  %r",
                      turn, field, self.divergence["live"],
                      self.divergence["recorded"])
            msg = (f"replay divergence at recorded turn {turn}: "
                   f"{field} differs from recording")
            if self.strict_fidelity:
                raise ReplayDivergence(msg)
            log.warning("%s (continuing: strict_fidelity=false)", msg)
            return

    def build_assistant_message(self, content: str | None,
                                tool_calls: list[ToolCall]) -> dict:
        """History-safe assistant message dict (same shape as the live client)."""
        msg: dict = {"role": "assistant", "content": content}
        if tool_calls:
            msg["tool_calls"] = [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.name,
                              "arguments": json.dumps(tc.arguments)}}
                for tc in tool_calls
            ]
        return msg

    def query_server_context(self):
        """No server in replay; keep the configured context size."""
        return 0


def resolve_replay_source(path: Path) -> tuple[Path, Path | None, str]:
    """Resolve a run dir (or direct transcript path) to
    (transcript, trace_path_or_None, mode)."""
    path = Path(path)
    transcript = path
    trace = None
    if path.is_dir():
        candidates = sorted(path.glob("harness_run/transcripts/*.log")) or \
            sorted(path.glob("transcripts/*.log"))
        if not candidates:
            raise FileNotFoundError(f"no transcript under {path}")
        transcript = candidates[0]
        t = path / "host_task" / ".trace.jsonl"
        trace = t if t.is_file() else None
    mode = ""
    if trace is not None:
        try:
            for line in trace.read_text().splitlines():
                if not line.strip():
                    continue
                d = json.loads(line)
                if d.get("event") == "session_start":
                    mode = str((d.get("context_contract") or {}).get("mode") or "")
                    break
        except (json.JSONDecodeError, OSError):
            mode = ""
    return transcript, trace, mode


def load_replay_provenance(path: Path) -> dict:
    """Read the source model, recorded config paths, and context mode."""
    import hashlib
    path = Path(path)
    # Support a run-root file and the older nested layout.
    sj = path / "session.json"
    if not sj.is_file():
        sj = path / "harness_run" / "session.json"
    if not sj.is_file():
        raise ValueError(
            f"no session.json or harness_run/session.json under {path}")
    s = json.loads(sj.read_text())
    config_paths = list(s.get("config_paths") or [])
    hashes = dict(s.get("config_path_hashes") or {})
    if not config_paths or not s.get("model"):
        raise ValueError("session.json lacks config_paths/model")
    resolved_paths = []
    for p_ in config_paths:
        f = Path(p_)
        if not f.is_file():
            raise ValueError(f"recorded config layer missing on disk: {p_}")
        digest = hashlib.sha256(f.read_bytes()).hexdigest()
        want = hashes.get(p_, "")
        if want and digest != want:
            # Content-addressed substitute: a copy of the RECORDED layer
            # materialized (e.g. from git history) into the recording's
            # replay_layers/, named by the recorded sha. Accepted only
            # when its content hashes to exactly the recorded value —
            # parity is preserved byte-for-byte.
            sub = path / "replay_layers" / f"{want}{f.suffix}"
            if sub.is_file() and hashlib.sha256(
                    sub.read_bytes()).hexdigest() == want:
                resolved_paths.append(str(sub))
                continue
            # When the recorded bytes are missing, an explicit environment
            # flag may accept a substitute whose hash does not match. Log
            # this choice because it breaks byte parity with the recording.
            import logging
            import os as _os
            if _os.environ.get("YUJ_REPLAY_LAYER_SUBSTITUTE_OK") == "1" \
                    and sub.is_file():
                logging.getLogger(__name__).warning(
                    "REPLAY PARITY BREAK (declared): layer %s substituted "
                    "with %s (recorded sha %s unrecoverable)",
                    p_, sub, want[:12])
                resolved_paths.append(str(sub))
                continue
            raise ValueError(
                f"recorded config layer drifted since the recording: {p_} "
                f"(sha now {digest[:12]}, recorded {want[:12]}); "
                f"materialize the recorded bytes at {sub} to replay")
        resolved_paths.append(p_)
    config_paths = resolved_paths
    return {"model": s["model"], "config_paths": config_paths,
            "context_mode": str(s.get("context_mode") or "")}
