"""Keep the useful parts of saved solver state in the next model request.

The ``salience`` mode reads the same ``.solver/state.json`` file as
``compound`` and ``compound_selective``. It selects a smaller part of that
state. It does not read the raw message log as a second source.

Selector policy is task-agnostic:

- keep a recent tail because late turns carry the live plan
- keep sparse older anchors for mutations, failures, source reads, and tests
- reduce trace/evidence/tool-result budgets as the turn count rises
- never inspect task ids, benchmark metadata, expected patches, or old outcomes
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from ..._shared.classification import classify_outcome
from ..bash_write_classification import (
    BASH_LEGACY_MUTATION_RE,
    BASH_LEGACY_PYTHON_WRITE_RE,
)
from .compound_selective_context import CompoundSelectiveContext
from ._metadata import (
    COMPOUND_SELECTIVE_BUDGET_CONFIG_ATTRS,
    COMPOUND_SELECTIVE_CONSTRUCTOR_CONFIG_ATTRS,
    SALIENCE_SECTION_LABELS,
    SALIENCE_SECTION_ORDER,
    ContextModeMetadata,
)

class SalienceContext(CompoundSelectiveContext):
    """Compound-selective renderer with stronger recency/salience pressure."""

    _MUTATION_PREFIXES = ("edit(", "write(", "str_replace(", "create(")
    _BASH_MUTATION_RE = BASH_LEGACY_MUTATION_RE
    _BASH_PYTHON_WRITE_RE = BASH_LEGACY_PYTHON_WRITE_RE
    _BASH_READ_ONLY_RE = re.compile(
        r"^(?:cd\s+\S+\s+&&\s+|env\s+[^;&|]+\s+)*"
        r"(?:cat|sed\s+-n|grep|rg|find|ls|head|tail|wc)\b"
    )
    _BASH_VERIFICATION_RE = re.compile(
        r"\b(?:pytest|py\.test|python3?\s+-m\s+pytest|"
        r"unittest|python3?\s+-c|python3?\s+-\s*<<)\b"
    )
    _MUTATION_INTENT_RE = re.compile(
        r"\b(?:apply|change|edit|fix|implement|modify|patch|replace|rewrite|write)\b",
        re.IGNORECASE,
    )
    _DIFF_ACTION_RE = re.compile(r"(?:^|[;&|]\s*)git\s+diff(?:\s|$)")
    _FILE_TOKEN_RE = re.compile(
        r"(?<![\w/.-])(?:/testbed/)?[A-Za-z0-9_./+-]+\."
        r"(?:py|rst|txt|md|toml|cfg|ini|yaml|yml)\b"
    )
    _PY_OPEN_PATH_RE = re.compile(
        r"(?:open|Path)\(\s*['\"]([^'\"]+)['\"]"
    )
    _PERMISSION_PATH_RE = re.compile(
        r"Permission denied:\s*['\"]([^'\"]+)['\"]"
    )
    _DIFF_HEADER_RE = re.compile(r"^diff --git a/(.+?) b/(.+)$")
    _DIFF_STAT_RE = re.compile(r"^\s*([^|\n]+?)\s+\|\s+\d+")
    _CANDIDATE_INTENT_RE = re.compile(
        r"\b(?:change|edit|fix|implement|modify|patch|replace|rewrite|"
        r"apply|ready to apply|the fix is|fix is to|use .+ instead of)\b",
        re.IGNORECASE,
    )
    _EXPLORATION_INTENT_RE = re.compile(
        r"\b(?:understand|explore|examine|look at|read|check|first inspect|"
        r"first look|find where|search for|current state)\b",
        re.IGNORECASE,
    )
    _CONCRETE_EDIT_RE = re.compile(
        r"\b(?:change .+ to|change .+ from .+ to|replace .+ with|use .+ instead of|"
        r"the fix is|fix is to|ready to apply|apply the (?:edit|fix|patch)|"
        r"implement the fix|modify .+ to|patch .+)\b",
        re.IGNORECASE,
    )
    _NON_EDIT_PROGRESS_RE = re.compile(
        r"\b(?:run(?:ning)? (?:the )?(?:test|tests|pytest)|try running|"
        r"verify|verification|git diff|current state|check the current|"
        r"correct python environment|python environment|probe|reproducer)\b",
        re.IGNORECASE,
    )
    _APPLIED_EDIT_RE = re.compile(
        r"\b(?:already (?:made|applied|implemented|changed)|"
        r"(?:is|has|have|had|was|were) already "
        r"(?:made|applied|implemented|changed)|"
        r"(?:is|has|have|had|was|were) already been "
        r"(?:made|applied|implemented|changed)|"
        r"successfully (?:made|applied|implemented|changed)|"
        r"(?:change|changes|edit|edits|fix|patch) "
        r"(?:is|has|have|had|was|were) already "
        r"(?:made|applied|implemented|changed)|"
        r"marked as SUCCESS|need to verify|verify the (?:change|patch|fix)|"
        r"tests? pass)\b",
        re.IGNORECASE,
    )
    _ENV_BLOCKER_RE = re.compile(
        r"\b(?:ModuleNotFoundError|No matching distribution|Could not find a version|"
        r"network|package index|pip install|ImportError|"
        r"AttributeError:\s+module ['\"]collections['\"] has no attribute|"
        r"collections\.(?:Mapping|MutableSet|Sequence))\b",
        re.IGNORECASE,
    )

    _TARGET_TOKENS = 46_000
    _MIN_TRACE_LINES = 8
    _MIN_TOOL_CHARS = 2_500
    _PRESSURE_MIN_TURN = 16
    _READ_LOOP_WINDOW = 16
    _READ_LOOP_REPEAT_CAP = 2

    @staticmethod
    def _entry_step(item) -> int | None:
        if not isinstance(item, dict):
            return None
        step = item.get("step")
        return step if isinstance(step, int) else None

    @classmethod
    def _is_mutation_item(cls, item) -> bool:
        if isinstance(item, dict) and item.get("plan_artifact") is True:
            return False
        if isinstance(item, dict) and item.get("source_write_like") is True:
            return True
        action = cls._item_action_text(item)
        if action.startswith(cls._MUTATION_PREFIXES):
            return True
        cmd = cls._extract_action_cmd(item)
        return bool(
            cmd
            and (
                cls._BASH_MUTATION_RE.search(cmd)
                or cls._BASH_PYTHON_WRITE_RE.search(cmd)
            )
        )

    @classmethod
    def _mutation_failed(cls, item) -> bool:
        if not isinstance(item, dict):
            return False
        if item.get("gate_blocked") is True:
            return True
        return classify_outcome(str(item.get("result") or "")) == "FAIL"

    @classmethod
    def _is_successful_mutation_item(cls, item) -> bool:
        return cls._is_mutation_item(item) and not cls._mutation_failed(item)

    @classmethod
    def _is_failed_mutation_item(cls, item) -> bool:
        return cls._is_mutation_item(item) and cls._mutation_failed(item)

    @classmethod
    def _is_read_only_item(cls, item) -> bool:
        action = cls._item_action_text(item)
        if action.startswith(("read(", "grep(", "glob(", "list_files(")):
            return True
        cmd = cls._extract_action_cmd(item).strip()
        return bool(cmd and cls._BASH_READ_ONLY_RE.search(cmd))

    @classmethod
    def _is_verification_item(cls, item) -> bool:
        if cls._anchor_bucket(item) == "test":
            return True
        cmd = cls._extract_action_cmd(item)
        return bool(cmd and cls._BASH_VERIFICATION_RE.search(cmd))

    @classmethod
    def _is_diff_item(cls, item) -> bool:
        cmd = cls._extract_action_cmd(item)
        return bool(cmd and cls._DIFF_ACTION_RE.search(cmd))

    @classmethod
    def _is_failing_trace_item(cls, item, failing_steps: set[int]) -> bool:
        if not isinstance(item, dict):
            return False
        step = cls._entry_step(item)
        return item.get("gate_blocked") is True or (
            step is not None and step in failing_steps
        )

    @staticmethod
    def _recent_trace_count(trace_len: int, limit: int) -> int:
        if trace_len <= limit:
            return trace_len
        recent_floor = min(4, limit)
        return min(trace_len, max(recent_floor, (limit * 2 + 2) // 3))

    def _pressure_trace_limit(self) -> int:
        base = self._selective_trace_lines
        return max(base, min(36, max(24, base * 2)))

    def _pressure_unresolved_evidence_limit(self) -> int:
        base = self._selective_unresolved_evidence_lines
        return max(base, 8)

    def _pressure_resolved_evidence_limit(self) -> int:
        base = self._selective_resolved_evidence_lines
        return max(base, 4)

    @classmethod
    def _recent_loop_stats(cls, trace: list) -> tuple[int, int, int]:
        recent_window = min(cls._READ_LOOP_WINDOW, len(trace))
        recent_items = trace[-recent_window:] if recent_window else []
        recent_read_only = sum(
            1 for item in recent_items if cls._is_read_only_item(item)
        )
        recent_mutations = sum(
            1 for item in recent_items if cls._is_successful_mutation_item(item)
        )
        return recent_window, recent_read_only, recent_mutations

    @classmethod
    def _has_read_only_loop(cls, trace: list) -> bool:
        recent_window, recent_read_only, recent_mutations = cls._recent_loop_stats(trace)
        return (
            recent_window >= 8
            and recent_read_only >= max(8, recent_window - 2)
            and recent_mutations == 0
        )

    def _pressure_tool_chars(self, trace: list | None = None) -> int:
        base = self._selective_recent_tool_results_chars
        base = min(base, 6_000)
        if self._turn_count >= 160:
            base = min(base, 4_000)
        elif self._turn_count >= 120:
            base = min(base, 6_000)
        elif self._turn_count >= 80:
            base = min(base, 9_000)
        elif self._turn_count >= 40:
            base = min(base, 12_000)
        if trace and self._has_read_only_loop(trace):
            base = min(base, 3_500)
        return max(self._MIN_TOOL_CHARS, base)

    def _trace_repeat_cap(self) -> int:
        if self._selective_trace_action_repeat_cap > 0:
            return self._selective_trace_action_repeat_cap
        return self._READ_LOOP_REPEAT_CAP

    def _select_recent_salience_trace(self, trace: list, limit: int) -> list:
        if limit <= 0 or not trace:
            return []
        if not self._has_read_only_loop(trace):
            return trace[-limit:]
        candidates = trace[-limit:]
        return self._select_tail_with_repeat_cap(
            candidates,
            limit=limit,
            repeat_cap=self._READ_LOOP_REPEAT_CAP,
            key_fn=self._item_action_key,
        )

    def _select_salient_older_trace(
        self,
        older_trace: list,
        *,
        limit: int,
        failing_steps: set[int],
        exclude_read_keys: set[str] | None = None,
    ) -> list:
        if limit <= 0 or not older_trace:
            return []

        selected: list[tuple[int, object]] = []
        selected_indices: set[int] = set()
        seen_counts: dict[str, int] = {}
        cap = self._trace_repeat_cap() if self._trace_repeat_cap() > 0 else None

        def maybe_add(idx: int, item: object) -> bool:
            if idx in selected_indices or len(selected) >= limit:
                return False
            key = self._item_action_key(item)
            if (
                exclude_read_keys
                and key in exclude_read_keys
                and self._is_read_only_item(item)
            ):
                return False
            count = seen_counts.get(key, 0)
            if cap is not None and count >= cap:
                return False
            seen_counts[key] = count + 1
            selected.append((idx, item))
            selected_indices.add(idx)
            return True

        def add_newest_where(predicate) -> None:
            for idx in range(len(older_trace) - 1, -1, -1):
                if len(selected) >= limit:
                    break
                item = older_trace[idx]
                if predicate(item):
                    maybe_add(idx, item)

        def add_oldest_where(predicate, budget: int) -> None:
            added = 0
            for idx, item in enumerate(older_trace):
                if len(selected) >= limit or added >= budget:
                    break
                if predicate(item) and maybe_add(idx, item):
                    added += 1

        add_newest_where(
            lambda item: self._is_failing_trace_item(item, failing_steps)
        )
        add_newest_where(self._is_mutation_item)

        source_budget = min(self._selective_trace_source_anchor_lines, limit)
        add_oldest_where(lambda item: self._anchor_bucket(item) == "source", source_budget)

        test_budget = min(
            self._selective_trace_test_anchor_lines,
            max(0, limit - len(selected)),
        )
        add_oldest_where(lambda item: self._anchor_bucket(item) == "test", test_budget)

        generic_budget = min(
            self._selective_trace_anchor_lines,
            max(0, limit - len(selected)),
        )
        add_oldest_where(lambda item: True, generic_budget)

        add_newest_where(lambda item: True)

        selected.sort(key=lambda pair: pair[0])
        return [item for _, item in selected[:limit]]

    def _select_salience_trace(
        self,
        trace: list,
        *,
        failing_steps: set[int],
        trace_limit: int | None = None,
    ) -> tuple[list, list]:
        limit = trace_limit if trace_limit is not None else self._pressure_trace_limit()
        if limit <= 0 or not trace:
            return [], []
        if len(trace) <= limit:
            return [], trace

        recent_count = self._recent_trace_count(len(trace), limit)
        recent_trace = self._select_recent_salience_trace(trace, recent_count)
        older_budget = max(0, limit - len(recent_trace))
        exclude_read_keys = (
            {self._item_action_key(item) for item in recent_trace}
            if self._has_read_only_loop(trace)
            else set()
        )
        older_trace = self._select_salient_older_trace(
            trace[:-recent_count],
            limit=older_budget,
            failing_steps=failing_steps,
            exclude_read_keys=exclude_read_keys,
        )
        return older_trace, recent_trace

    def _format_salience_trace(
        self,
        trace: list,
        evidence: list,
        *,
        trace_limit: int | None = None,
    ) -> str:
        failing_steps = set()
        for item in evidence:
            step = self._entry_step(item)
            if step is not None and self._is_failing(item):
                failing_steps.add(step)
        older_trace, recent_trace = self._select_salience_trace(
            trace,
            failing_steps=failing_steps,
            trace_limit=trace_limit,
        )

        parts: list[str] = []
        if older_trace:
            rendered = self._format_trace(older_trace, len(older_trace))
            if rendered:
                parts.append("-- salient older --\n" + rendered)
        if recent_trace:
            rendered = self._format_trace(recent_trace, len(recent_trace))
            if rendered:
                parts.append("-- recent --\n" + rendered)
        return "\n".join(parts)

    def _split_evidence_pressure(
        self,
        evidence_list,
        *,
        unresolved_limit: int,
        resolved_limit: int,
    ) -> tuple[list[str], list[str]]:
        fails: list[str] = []
        passes: list = []
        for item in evidence_list:
            if self._is_failing(item):
                fails.append(self._render_evidence_item(item))
            else:
                passes.append(item)
        selected_passes = self._select_anchored_tail(
            passes,
            limit=resolved_limit,
            repeat_cap=self._selective_resolved_action_repeat_cap,
            anchor_lines=min(self._selective_resolved_anchor_lines, resolved_limit),
            source_anchor_lines=min(
                self._selective_resolved_source_anchor_lines, resolved_limit
            ),
            test_anchor_lines=min(
                self._selective_resolved_test_anchor_lines, resolved_limit
            ),
            key_fn=self._item_action_key,
        )
        return (
            fails[-unresolved_limit:],
            [self._render_resolved_evidence_item(item) for item in selected_passes],
        )

    def _format_tool_results_budget(self, char_budget: int) -> str:
        original = self._recent_tool_results_chars
        self._recent_tool_results_chars = max(0, char_budget)
        try:
            return self._format_tool_results()
        finally:
            self._recent_tool_results_chars = original

    def _format_tool_results_pressure(self, trace: list, char_budget: int) -> str:
        repeated = self._repeated_read_summaries(trace, min_count=8)
        if repeated and self._has_read_only_loop(trace):
            rendered = "; ".join(repeated[:3])
            return (
                "=== Tool results suppressed ===\n"
                "Recent raw read/search output is omitted because the same read/search "
                f"actions are looping: {rendered}. Use the trace/evidence already shown; "
                "the next tool call should not repeat those reads."
            )
        return self._format_tool_results_budget(char_budget)

    @classmethod
    def _latest_mutation_intent(cls, trace: list) -> tuple[int, str] | None:
        for idx in range(len(trace) - 1, -1, -1):
            item = trace[idx]
            if not isinstance(item, dict):
                continue
            if cls._is_successful_mutation_item(item):
                continue
            reasoning = (item.get("reasoning") or "").strip()
            if not cls._is_candidate_edit_reasoning(reasoning):
                continue
            compact = " ".join(reasoning.split())
            if len(compact) > 180:
                compact = compact[:177] + "..."
            return idx, compact
        return None

    @classmethod
    def _is_candidate_edit_reasoning(cls, reasoning: str) -> bool:
        reasoning = (reasoning or "").strip()
        if not reasoning or not cls._CANDIDATE_INTENT_RE.search(reasoning):
            return False
        if cls._APPLIED_EDIT_RE.search(reasoning):
            return False
        concrete = cls._CONCRETE_EDIT_RE.search(reasoning) is not None
        if cls._NON_EDIT_PROGRESS_RE.search(reasoning) and not concrete:
            return False
        if cls._EXPLORATION_INTENT_RE.search(reasoning) and not concrete:
            return False
        return True

    @classmethod
    def _repeated_read_summaries(cls, trace: list, *, min_count: int = 3) -> list[str]:
        counts = Counter(
            cls._item_action_text(item)
            for item in trace
            if cls._is_read_only_item(item)
        )
        summaries = []
        for action, count in counts.most_common(3):
            if count < min_count:
                continue
            if len(action) > 120:
                action = action[:117] + "..."
            summaries.append(f"{action} [{count}x]")
        return summaries

    @classmethod
    def _repeated_inspection_summaries(
        cls, trace: list, *, min_count: int = 3
    ) -> list[str]:
        counts = Counter(
            cls._item_action_text(item)
            for item in trace
            if cls._is_read_only_item(item) or cls._is_diff_item(item)
        )
        summaries = []
        for action, count in counts.most_common(3):
            if count < min_count:
                continue
            if len(action) > 120:
                action = action[:117] + "..."
            summaries.append(f"{action} [{count}x]")
        return summaries

    @staticmethod
    def _normalize_target_path(path: str) -> str:
        path = path.strip().strip("'\"")
        if path.startswith("/testbed/"):
            path = path[len("/testbed/"):]
        if path.startswith("./"):
            path = path[2:]
        return path

    @classmethod
    def _mutation_target_paths(cls, item) -> list[str]:
        if not cls._is_mutation_item(item):
            return []
        explicit_paths = (
            list(item.get("source_write_paths") or [])
            if isinstance(item, dict)
            else []
        )
        cmd = cls._extract_action_cmd(item)
        paths = [
            cls._normalize_target_path(str(path))
            for path in explicit_paths
        ]
        if not cmd:
            return paths
        paths.extend(
            cls._normalize_target_path(match.group(0))
            for match in cls._FILE_TOKEN_RE.finditer(cmd)
        )
        paths.extend(
            cls._normalize_target_path(match.group(1))
            for match in cls._PY_OPEN_PATH_RE.finditer(cmd)
        )
        seen: set[str] = set()
        unique = []
        for path in paths:
            if path and path not in seen:
                seen.add(path)
                unique.append(path)
        return unique

    @classmethod
    def _mutated_paths(cls, trace: list) -> list[str]:
        seen: set[str] = set()
        paths: list[str] = []
        for item in trace:
            if not cls._is_successful_mutation_item(item):
                continue
            for path in cls._mutation_target_paths(item):
                if path not in seen:
                    seen.add(path)
                    paths.append(path)
        return paths

    @classmethod
    def _failed_mutation_paths(cls, item) -> list[str]:
        if not cls._is_failed_mutation_item(item):
            return []
        paths = cls._mutation_target_paths(item)
        result = str(item.get("result") or "") if isinstance(item, dict) else ""
        for match in cls._PERMISSION_PATH_RE.finditer(result):
            path = cls._normalize_target_path(match.group(1))
            if path and path not in paths:
                paths.append(path)
        for match in cls._FILE_TOKEN_RE.finditer(result):
            path = cls._normalize_target_path(match.group(0))
            if (
                path
                and path not in paths
                and not path.startswith("/")
                and not path.startswith("opt/")
            ):
                paths.append(path)
        return paths

    @classmethod
    def _latest_diff_paths(cls, trace: list) -> list[str]:
        for item in reversed(trace):
            if not isinstance(item, dict) or not cls._is_diff_item(item):
                continue
            result = str(item.get("result") or "")
            paths: list[str] = []
            seen: set[str] = set()
            for line in result.splitlines():
                header_match = cls._DIFF_HEADER_RE.match(line)
                if header_match:
                    path = cls._normalize_target_path(header_match.group(2))
                else:
                    stat_match = cls._DIFF_STAT_RE.match(line)
                    path = cls._normalize_target_path(stat_match.group(1)) if stat_match else ""
                if path and path not in seen:
                    seen.add(path)
                    paths.append(path)
            if paths:
                return paths
        return []

    @staticmethod
    def _compact_text(text: str, limit: int = 360) -> str:
        compact = " ".join(str(text or "").split())
        if len(compact) > limit:
            return compact[: max(0, limit - 3)] + "..."
        return compact

    @classmethod
    def _read_target_paths(cls, item) -> list[str]:
        if not cls._is_read_only_item(item):
            return []
        paths: list[str] = []
        explicit_path = cls._extract_action_path(item)
        if explicit_path:
            paths.append(cls._normalize_target_path(explicit_path))
        action = cls._item_action_text(item)
        cmd = cls._extract_action_cmd(item)
        for source in (cmd, action):
            for match in cls._FILE_TOKEN_RE.finditer(source):
                path = cls._normalize_target_path(match.group(0))
                if path and cls._is_concrete_source_path(path):
                    paths.append(path)
        seen: set[str] = set()
        unique: list[str] = []
        for path in paths:
            if path and path not in seen:
                seen.add(path)
                unique.append(path)
        return unique

    @classmethod
    def _candidate_reasoning_item(cls, trace: list) -> tuple[int, object, str] | None:
        for idx in range(len(trace) - 1, -1, -1):
            item = trace[idx]
            if not isinstance(item, dict):
                continue
            if cls._is_successful_mutation_item(item):
                continue
            reasoning = str(item.get("reasoning") or "").strip()
            if not cls._is_candidate_edit_reasoning(reasoning):
                continue
            mutation_after = any(
                cls._is_successful_mutation_item(next_item)
                for next_item in trace[idx + 1:]
            )
            if mutation_after:
                continue
            return idx, item, reasoning
        return None

    def _candidate_edit_details(self, trace: list) -> dict[str, object] | None:
        candidate = self._candidate_reasoning_item(trace)
        if candidate is None:
            return None
        idx, item, reasoning = candidate
        target_paths: list[str] = []
        for scan_item in reversed(trace[max(0, idx - 16): idx + 1]):
            for path in self._read_target_paths(scan_item):
                if path not in target_paths:
                    target_paths.append(path)
            if len(target_paths) >= 4:
                break
        for path in self._mutation_target_paths(item):
            if path not in target_paths:
                target_paths.insert(0, path)

        latest_read = ""
        for scan_item in reversed(trace[: idx + 1]):
            if self._read_target_paths(scan_item):
                latest_read = (
                    f"step {self._entry_step(scan_item)} "
                    f"{self._compact_text(self._item_action_text(scan_item), 180)}"
                )
                result = (
                    scan_item.get("result", "") if isinstance(scan_item, dict) else ""
                )
                if result:
                    latest_read += f" -> {self._compact_text(result, 300)}"
                break

        return {
            "trace_index": idx,
            "item": item,
            "reasoning": reasoning,
            "target_paths": target_paths,
            "latest_read": latest_read,
        }

    @classmethod
    def _post_mutation_stall_details(
        cls,
        trace: list,
        *,
        last_mutation_idx: int,
        last_test_idx: int,
    ) -> dict[str, object] | None:
        after_progress = trace[max(last_mutation_idx, last_test_idx) + 1:]
        if not after_progress:
            return None

        since_mutation = len(trace) - last_mutation_idx - 1
        since_verification = len(trace) - last_test_idx - 1
        repeated = cls._repeated_inspection_summaries(after_progress, min_count=3)
        diff_count = sum(1 for item in after_progress if cls._is_diff_item(item))

        recent = after_progress[-cls._READ_LOOP_WINDOW:]
        inspection_count = sum(
            1
            for item in recent
            if cls._is_read_only_item(item) or cls._is_diff_item(item)
        )
        progress_count = sum(
            1
            for item in recent
            if cls._is_successful_mutation_item(item) or cls._is_verification_item(item)
        )
        inspection_tail = (
            len(recent) >= 8
            and inspection_count >= max(8, len(recent) - 2)
            and progress_count == 0
        )
        stale_after_probe = since_verification >= 20 and (
            bool(repeated) or diff_count >= 2 or inspection_count >= 8
        )
        stale_after_patch = since_mutation >= 40 and (bool(repeated) or diff_count >= 3)
        if not (inspection_tail or stale_after_probe or stale_after_patch):
            return None

        return {
            "since_mutation": since_mutation,
            "since_verification": since_verification,
            "repeated": repeated,
            "diff_count": diff_count,
            "inspection_tail": inspection_tail,
        }

    @classmethod
    def _post_mutation_revision_details(
        cls, trace: list, *, last_mutation_idx: int
    ) -> dict[str, object] | None:
        after_mutation = trace[last_mutation_idx + 1:]
        if not after_mutation:
            return None

        tests = [
            item for item in after_mutation
            if cls._is_verification_item(item)
        ]
        if len(tests) < 3:
            return None

        repeated = cls._repeated_inspection_summaries(after_mutation, min_count=3)
        verification_counts = Counter(
            cls._item_action_text(item)
            for item in tests
        )
        repeated_verifications = []
        for action, count in verification_counts.most_common(3):
            if count < 3:
                continue
            if len(action) > 120:
                action = action[:117] + "..."
            repeated_verifications.append(f"{action} [{count}x]")

        latest_failure = ""
        env_blocker = ""
        for item in reversed(after_mutation):
            if not isinstance(item, dict):
                continue
            result = str(item.get("result") or "")
            if not env_blocker and cls._ENV_BLOCKER_RE.search(result):
                env_blocker = cls._compact_text(result, 500)
            if not latest_failure and classify_outcome(result) == "FAIL":
                latest_failure = cls._compact_text(result, 600)
            if latest_failure and env_blocker:
                break

        should_render = (
            len(tests) >= 5
            or bool(repeated_verifications)
            or (len(tests) >= 3 and bool(latest_failure))
        )
        if not should_render:
            return None
        return {
            "verification_count": len(tests),
            "latest_verification_step": cls._entry_step(tests[-1]),
            "repeated": repeated,
            "repeated_verifications": repeated_verifications,
            "latest_failure": latest_failure,
            "env_blocker": env_blocker,
        }

    def _format_next_action_contract(self, trace: list, evidence: list) -> str:
        if not trace:
            return ""
        mutation_indices = [
            idx for idx, item in enumerate(trace)
            if self._is_successful_mutation_item(item)
        ]
        failed_mutation_indices = [
            idx for idx, item in enumerate(trace)
            if self._is_failed_mutation_item(item)
        ]
        test_indices = [
            idx for idx, item in enumerate(trace)
            if self._is_verification_item(item)
        ]
        mutated_paths = self._mutated_paths(trace)
        diff_paths = self._latest_diff_paths(trace)
        read_loop = self._has_read_only_loop(trace)

        if failed_mutation_indices and (
            not mutation_indices or failed_mutation_indices[-1] > mutation_indices[-1]
        ):
            last_failed = failed_mutation_indices[-1]
            item = trace[last_failed]
            target_paths = self._failed_mutation_paths(item) or diff_paths
            lines = [
                "Status: last source write attempt failed; no source mutation is recorded.",
                f"- failed write step: {self._entry_step(item)}",
            ]
            if target_paths:
                lines.append(f"- target path: {', '.join(target_paths[:4])}")
            result = item.get("result", "") if isinstance(item, dict) else ""
            if result:
                lines.append(
                    f"- failure: {self._compact_text(result, 700)}"
                )
            lines.extend([
                "- next bash command should retry the source edit, not run verification and not reread the same region.",
                "- source files may reject direct Path.write_text(); prefer sed -i/perl -0pi for one-line replacements, or write a temporary sibling file and os.replace it.",
                "- example shape:",
                "  cd /testbed && python - <<'PY'",
                "  from pathlib import Path",
                "  import os",
                f"  path = Path({(target_paths[0] if target_paths else 'relative/source.py')!r})",
                "  text = path.read_text()",
                "  old = \"\"\"paste the exact old snippet shown in the evidence\"\"\"",
                "  new = \"\"\"paste the corrected snippet\"\"\"",
                "  updated = text.replace(old, new, 1)",
                "  if updated == text:",
                "      raise SystemExit('target snippet not found')",
                "  tmp = path.with_name(path.name + '.tmp-write')",
                "  tmp.write_text(updated)",
                "  os.replace(tmp, path)",
                "  PY",
            ])
            return "\n".join(lines)

        if mutation_indices:
            last_mutation = mutation_indices[-1]
            tests_after = [idx for idx in test_indices if idx > last_mutation]
            target_paths = mutated_paths or diff_paths
            if tests_after:
                revision = self._post_mutation_revision_details(
                    trace, last_mutation_idx=last_mutation
                )
                if revision is not None:
                    lines = [
                        "Status: source mutation already has repeated verification/probe results; more reruns are not progress.",
                        f"- last mutation step: {self._entry_step(trace[last_mutation])}",
                        f"- latest verification/probe step: {revision.get('latest_verification_step')}",
                        f"- verification/probe count after mutation: {revision.get('verification_count')}",
                    ]
                    if target_paths:
                        lines.append(f"- patch paths: {', '.join(target_paths[:6])}")
                    repeated_verifications = list(
                        revision.get("repeated_verifications") or []
                    )
                    if repeated_verifications:
                        lines.append(
                            "- repeated verification/probe commands: "
                            + "; ".join(str(item) for item in repeated_verifications[:3])
                            + "."
                        )
                    repeated = list(revision.get("repeated") or [])
                    if repeated:
                        lines.append(
                            "- repeated post-mutation inspections: "
                            + "; ".join(str(item) for item in repeated[:3])
                            + "."
                        )
                    latest_failure = str(revision.get("latest_failure") or "")
                    if latest_failure:
                        lines.append(f"- latest failing result: {latest_failure}")
                    if revision.get("env_blocker"):
                        lines.append(
                            "- verification blocker looks environmental; do not patch "
                            "unrelated compatibility/import files only to make local "
                            "probes run."
                        )
                    if len(target_paths) > 1:
                        lines.append(
                            "- patch hygiene: keep only task-relevant source edits; "
                            "remove setup/interpreter compatibility edits made only to run probes."
                        )
                    lines.extend([
                        "- next action must revise the source patch from the concrete failure, "
                        "remove setup-only edits, or call done if the current patch is intended.",
                        "- do not rerun the same verification/probe or reread the same diff without a new edit.",
                    ])
                    return "\n".join(lines)
                last_test = tests_after[-1]
                stalled = self._post_mutation_stall_details(
                    trace,
                    last_mutation_idx=last_mutation,
                    last_test_idx=last_test,
                )
                if stalled is None:
                    return ""
                lines = [
                    "Status: source mutation has been verified or probed, but the trajectory is stalled after the patch.",
                    f"- last mutation step: {self._entry_step(trace[last_mutation])}",
                    f"- latest verification/probe step: {self._entry_step(trace[last_test])}",
                ]
                if target_paths:
                    lines.append(f"- patch paths: {', '.join(target_paths[:6])}")
                since_verification = stalled.get("since_verification")
                if isinstance(since_verification, int):
                    lines.append(
                        f"- {since_verification} step(s) since the latest verification/probe."
                    )
                repeated = list(stalled.get("repeated") or [])
                if repeated:
                    lines.append(
                        "- repeated post-mutation inspections: "
                        + "; ".join(str(item) for item in repeated[:3])
                        + "."
                    )
                if len(target_paths) > 1:
                    lines.append(
                        "- patch hygiene: keep only task-relevant source edits; "
                        "remove setup/interpreter compatibility edits made only to run probes."
                    )
                lines.extend([
                    "- next action must be one of: run one targeted verification/reproducer "
                    "for the current diff; revise the patch from a concrete failing result; "
                    "remove unrelated compatibility/setup edits and verify; or call done "
                    "if the current source patch is intended.",
                    "- do not keep reading git diff or source regions without a new failing fact.",
                ])
                return "\n".join(lines)
            lines = [
                "Status: source mutation exists and needs verification.",
                f"- last mutation step: {self._entry_step(trace[last_mutation])}",
            ]
            if target_paths:
                lines.append(f"- patch paths: {', '.join(target_paths[:6])}")
            lines.append(
                "- next bash command should run one targeted verification or a small reproducer for the changed behavior."
            )
            lines.append(
                "- if verification is blocked by environment/setup, treat that as a limitation; do not patch unrelated compatibility/import files to make local probes run."
            )
            lines.append(
                "- after one concrete verification result, revise the source patch or call done instead of repeating the same probe."
            )
            return "\n".join(lines)

        candidate = self._candidate_edit_details(trace)
        if candidate is not None:
            item = candidate["item"]
            reasoning = str(candidate["reasoning"])
            target_paths = list(candidate["target_paths"])
            latest_read = str(candidate["latest_read"])
            target = target_paths[0] if target_paths else "relative/source.py"
            lines = [
                "Status: pending source mutation; no source write has been recorded after the latest edit plan.",
                f"- edit intent step {self._entry_step(item)}: {self._compact_text(reasoning, 700)}",
            ]
            if target_paths:
                lines.append(f"- target path: {', '.join(target_paths[:4])}")
            if latest_read:
                lines.append(f"- evidence already available: {latest_read}")
            if read_loop:
                lines.append("- repeated reads of the same source area are closed.")
            lines.extend([
                "- next bash command must write the target source file.",
                "- do not use cat, sed -n, grep, rg, find, ls, head, tail, or wc as the next command.",
                "- use sed -i/perl -0pi for a one-line replacement, or a temp-file replace script for multiline edits:",
                "  cd /testbed && python - <<'PY'",
                "  from pathlib import Path",
                "  import os",
                f"  path = Path({target!r})",
                "  text = path.read_text()",
                "  old = \"\"\"paste the exact old snippet shown in the evidence\"\"\"",
                "  new = \"\"\"paste the corrected snippet\"\"\"",
                "  updated = text.replace(old, new, 1)",
                "  if updated == text:",
                "      raise SystemExit('target snippet not found')",
                "  tmp = path.with_name(path.name + '.tmp-write')",
                "  tmp.write_text(updated)",
                "  os.replace(tmp, path)",
                "  PY",
                "- after mutation, run targeted verification or a small reproducer before done.",
            ])
            return "\n".join(lines)

        if read_loop:
            repeated = self._repeated_read_summaries(trace, min_count=4)
            lines = [
                "Status: hard read/search loop before any recorded source mutation.",
            ]
            if repeated:
                lines.append("- closed read targets: " + "; ".join(repeated) + ".")
            lines.append(
                "- next bash command must not repeat those reads."
            )
            lines.append(
                "- next move should make a minimal source edit; only read again if it names a different unresolved symbol and will directly support that edit."
            )
            return "\n".join(lines)

        gate = self._last_blocking_gate_entry(evidence)
        if gate:
            return (
                "Status: harness gate is blocking the current trajectory.\n"
                f"- latest gate: {self._compact_text(gate, 500)}\n"
                "- next command must satisfy the gate instead of repeating the blocked action."
            )
        return ""

    def _last_blocking_gate_entry(self, evidence: list) -> str | None:
        for item in reversed(evidence):
            if isinstance(item, dict) and item.get("gate_blocked") is True:
                return self._render_evidence_item(item)
        return None

    def _format_salience_pressure(self, trace: list) -> str:
        if self._turn_count < self._PRESSURE_MIN_TURN or not trace:
            return ""

        actions = [self._item_action_text(item) for item in trace]
        if not actions:
            return ""

        mutation_indices = [
            idx for idx, item in enumerate(trace)
            if self._is_successful_mutation_item(item)
        ]
        failed_mutation_indices = [
            idx for idx, item in enumerate(trace)
            if self._is_failed_mutation_item(item)
        ]
        test_indices = [
            idx for idx, item in enumerate(trace)
            if self._is_verification_item(item)
        ]
        recent_window, recent_read_only, recent_mutations = self._recent_loop_stats(
            trace
        )

        repeated = 1
        latest = actions[-1]
        for action in reversed(actions[:-1]):
            if action != latest:
                break
            repeated += 1

        lines: list[str] = []
        if not mutation_indices:
            lines.append(
                f"No write/edit/create-like action has succeeded across {len(actions)} steps."
            )
            lines.append(
                "Calling done now would submit an empty patch; do not call done to describe a pending edit."
            )
        else:
            since_mutation = len(actions) - mutation_indices[-1] - 1
            if since_mutation >= 20:
                lines.append(
                    f"{since_mutation} steps since the last write/edit/create-like action."
                )

        if failed_mutation_indices:
            failed_item = trace[failed_mutation_indices[-1]]
            lines.append(
                "Latest write/edit-like command failed: "
                + self._compact_text(self._item_action_text(failed_item), 240)
            )

        if not test_indices:
            lines.append(
                f"No test/verification-like command is recorded across {len(actions)} steps."
            )
        else:
            since_test = len(actions) - test_indices[-1] - 1
            if since_test >= 20:
                lines.append(
                    f"{since_test} steps since the last test/verification-like command."
                )

        if repeated >= 3:
            lines.append(
                f"Newest action repeated {repeated} consecutive times: {latest}"
            )

        if (
            recent_window >= 8
            and recent_read_only >= max(8, recent_window - 2)
            and recent_mutations == 0
        ):
            lines.append(
                f"Recent actions are read/search-only ({recent_read_only}/{recent_window})."
            )

        read_repeats = self._repeated_read_summaries(trace)
        if read_repeats:
            lines.append(
                "Repeated read/search actions already seen: "
                + "; ".join(read_repeats)
                + "."
            )
        if read_repeats and recent_mutations == 0:
            lines.append(
                "Treat the repeated read/search targets as closed unless a new command names a different unresolved symbol needed for the next edit."
            )

        latest_intent = self._latest_mutation_intent(trace)
        if latest_intent is not None:
            intent_idx, intent = latest_intent
            mutation_after_intent = any(
                self._is_successful_mutation_item(item)
                for item in trace[intent_idx + 1:]
            )
            mutation_before_intent = any(
                self._is_successful_mutation_item(item)
                for item in trace[:intent_idx]
            )
            intent_reasoning = (
                str(trace[intent_idx].get("reasoning") or "")
                if isinstance(trace[intent_idx], dict)
                else ""
            )
            concrete_post_mutation_revision = (
                self._CONCRETE_EDIT_RE.search(intent_reasoning) is not None
                and self._NON_EDIT_PROGRESS_RE.search(intent_reasoning) is None
            )
            if (
                not mutation_after_intent
                and (
                    not mutation_before_intent
                    or concrete_post_mutation_revision
                )
            ):
                lines.append(
                    "Latest model intent says it is ready to edit: "
                    f"{intent}"
                )

        mutated_paths = self._mutated_paths(trace)
        if mutated_paths:
            shown = ", ".join(mutated_paths[:6])
            extra = "" if len(mutated_paths) <= 6 else f", +{len(mutated_paths) - 6} more"
            lines.append(f"Mutation targets recorded: {shown}{extra}.")
            last_mutation = mutation_indices[-1] if mutation_indices else -1
            tests_after_mutation = [
                idx for idx in test_indices if idx > last_mutation
            ]
            if not tests_after_mutation:
                lines.append(
                    "There are mutations without a later verification-like command; verify the patch or name the blocker instead of continuing broad exploration."
                )
            if len(mutated_paths) > 1:
                lines.append(
                    "Patch hygiene: keep only task-relevant source edits; remove setup/interpreter compatibility edits made only to run local probes."
                )

        diff_paths = self._latest_diff_paths(trace)
        if diff_paths:
            shown = ", ".join(diff_paths[:6])
            extra = "" if len(diff_paths) <= 6 else f", +{len(diff_paths) - 6} more"
            lines.append(f"Latest diff paths visible in trace: {shown}{extra}.")

        if not lines:
            return ""
        mutation_verified = False
        if mutation_indices and test_indices:
            last_mutation = mutation_indices[-1]
            mutation_verified = any(idx > last_mutation for idx in test_indices)
        if mutation_verified:
            lines.append(
                "Immediate next tool call should revise from a concrete verification result, "
                "run one targeted verification, call done if the patch is intended, "
                "or name a concrete blocker; avoid another broad cat/sed/grep/read "
                "of the same area."
            )
        else:
            lines.append(
                "Immediate next tool call should mutate a source file, run targeted "
                "verification after a mutation, or name a concrete blocker; avoid another "
                "broad cat/sed/grep/read of the same area."
            )
        lines.append(
            "If only bash/done tools are available, edit through bash with a short python/perl/sed script; do not request an unavailable patch tool."
        )
        return "\n".join(lines)

    def _build_parts(
        self,
        *,
        state_text: str,
        trace: list,
        evidence: list,
        trace_limit: int,
        unresolved_limit: int,
        resolved_limit: int,
        tool_chars: int,
    ) -> list[str]:
        parts = [f"Task: {self._original_prompt}"]

        action_contract = self._format_next_action_contract(trace, evidence)
        if action_contract:
            parts.append(f"=== Next Action Contract ===\n{action_contract}")

        if state_text:
            parts.append(f"=== State ===\n{state_text}")

        gate = self._last_blocking_gate_entry(evidence)
        if gate:
            parts.append(f"=== Gate (blocking) ===\n{gate}")

        pressure = self._format_salience_pressure(trace)
        if pressure:
            parts.append(f"=== Salience Pressure ===\n{pressure}")

        trace_rendered = self._format_salience_trace(
            trace,
            evidence,
            trace_limit=trace_limit,
        )
        if trace_rendered:
            parts.append(f"=== Trace ===\n{trace_rendered}")

        fails, passes = self._split_evidence_pressure(
            evidence,
            unresolved_limit=unresolved_limit,
            resolved_limit=resolved_limit,
        )
        if fails or passes:
            ev_lines = []
            if fails:
                ev_lines.append("-- unresolved --")
                ev_lines.extend(fails)
            if passes:
                ev_lines.append("-- resolved --")
                ev_lines.extend(passes)
            parts.append("=== Evidence ===\n" + "\n".join(ev_lines))

        tool_results = self._format_tool_results_pressure(trace, tool_chars)
        if tool_results:
            parts.append(tool_results)

        if self._suffix:
            parts.append(self._suffix)
        return parts

    def _messages_from_parts(self, parts: list[str]) -> list[dict]:
        return [
            {"role": "system", "content": self._system_content},
            {"role": "user", "content": "\n\n".join(parts)},
        ]

    def _bounded_projection(
        self,
        *,
        state_text: str,
        trace: list,
        evidence: list,
    ) -> list[dict]:
        trace_limit = self._pressure_trace_limit()
        unresolved_limit = self._pressure_unresolved_evidence_limit()
        resolved_limit = self._pressure_resolved_evidence_limit()
        tool_chars = self._pressure_tool_chars(trace)

        while True:
            messages = self._messages_from_parts(self._build_parts(
                state_text=state_text,
                trace=trace,
                evidence=evidence,
                trace_limit=trace_limit,
                unresolved_limit=unresolved_limit,
                resolved_limit=resolved_limit,
                tool_chars=tool_chars,
            ))
            if self._token_estimator(messages) <= self._TARGET_TOKENS:
                return messages
            next_trace = max(self._MIN_TRACE_LINES, trace_limit - 4)
            next_unresolved = max(3, unresolved_limit - 2)
            next_resolved = max(0, resolved_limit - 2)
            next_tool = max(self._MIN_TOOL_CHARS, int(tool_chars * 0.65))
            if (
                next_trace == trace_limit
                and next_unresolved == unresolved_limit
                and next_resolved == resolved_limit
                and next_tool == tool_chars
            ):
                return messages
            trace_limit = next_trace
            unresolved_limit = next_unresolved
            resolved_limit = next_resolved
            tool_chars = next_tool

    def _build_from_solver(self, solver_dir: Path) -> list[dict]:
        files = self._get_solver_files(solver_dir)

        if self._raw_state_cache is None:
            state_path = solver_dir / "state.json"
            try:
                self._raw_state_cache = json.loads(state_path.read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                self._raw_state_cache = {}
        raw_state = self._raw_state_cache
        raw_trace = raw_state.get("trace", []) if isinstance(raw_state, dict) else []
        raw_evidence = raw_state.get("evidence", []) if isinstance(raw_state, dict) else []

        return self._bounded_projection(
            state_text=files["state"],
            trace=raw_trace,
            evidence=raw_evidence,
        )


CONTEXT_MODE = "salience"
CONTEXT_CLASS = SalienceContext
CONTEXT_METADATA = ContextModeMetadata(
    cli_order=11,
    message_shape="two-message salience v1 projection after min_turns_before_context",
    state_source=".solver/state.json",
    source_type="trace_state",
    normal_prompt_sources=(
        ".solver/state.json",
        "in_memory_recent_tool_results",
        "live_workspace_files_from_state_trace_on_session_resume",
    ),
    section_order=SALIENCE_SECTION_ORDER,
    section_labels=SALIENCE_SECTION_LABELS,
    file_freshness="snapshot",
    injection_support="buried_in_projection",
    state_ignored_when_context_ignore_state=True,
    budget_config_attrs=COMPOUND_SELECTIVE_BUDGET_CONFIG_ATTRS,
    constructor_config_attrs=COMPOUND_SELECTIVE_CONSTRUCTOR_CONFIG_ATTRS,
)
