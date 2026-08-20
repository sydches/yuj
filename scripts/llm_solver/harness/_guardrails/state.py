"""Guardrail state types — extracted from guardrails.py."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class Action(Enum):
    PASS = "pass"
    WARN = "warn"
    BLOCK = "block"
    END = "end"


@dataclass(frozen=True)
class Decision:
    action: Action
    text: str = ""
    reason: str = ""

    @classmethod
    def pass_(cls) -> "Decision":
        return cls(Action.PASS)

    @classmethod
    def warn(cls, text: str, reason: str = "") -> "Decision":
        return cls(Action.WARN, text=text, reason=reason)

    @classmethod
    def block(cls, text: str, reason: str = "") -> "Decision":
        return cls(Action.BLOCK, text=text, reason=reason)

    @classmethod
    def end(cls, reason: str) -> "Decision":
        return cls(Action.END, reason=reason)


PASS = Decision.pass_()


@dataclass(frozen=True)
class GuardrailSpec:
    """First-class registry metadata for one guardrail callable."""

    name: str
    phase: str


GUARDRAIL_SPECS: tuple[GuardrailSpec, ...] = (
    GuardrailSpec("intent_gate", "turn_pre_dispatch"),
    GuardrailSpec("duplicate_guard", "turn_pre_dispatch"),
    GuardrailSpec("loop_detect", "turn_pre_dispatch"),
    GuardrailSpec("done_guard", "tool_pre_dispatch"),
    GuardrailSpec("mutation_repeat_guard", "tool_pre_dispatch"),
    GuardrailSpec("contract_gate", "tool_pre_dispatch"),
    GuardrailSpec("pre_mutation_gate", "tool_pre_dispatch"),
    GuardrailSpec("rumination_gate", "tool_pre_dispatch"),
    GuardrailSpec("error_ladder", "tool_post_dispatch"),
    GuardrailSpec("test_read_ladder", "tool_post_dispatch"),
    GuardrailSpec("rumination_ladder", "tool_post_dispatch"),
    GuardrailSpec("mark_bash_verified", "observers"),
    GuardrailSpec("observe_test_file_read", "observers"),
    GuardrailSpec("observe_contract_state", "observers"),
)


def guardrail_order_for_phase(phase: str) -> tuple[str, ...]:
    """Return guardrail names for a phase in run-loop order."""
    return tuple(spec.name for spec in GUARDRAIL_SPECS if spec.phase == phase)


TURN_PRE_DISPATCH_ORDER = guardrail_order_for_phase("turn_pre_dispatch")
TOOL_PRE_DISPATCH_ORDER = guardrail_order_for_phase("tool_pre_dispatch")
TOOL_POST_DISPATCH_ORDER = guardrail_order_for_phase("tool_post_dispatch")
OBSERVER_ORDER = guardrail_order_for_phase("observers")


@dataclass(frozen=True)
class GuardrailRegistry:
    """Composable guardrail callables grouped by run-loop phase."""

    turn_pre_dispatch: dict[str, Callable[..., Decision]]
    tool_pre_dispatch: dict[str, Callable[..., Decision]]
    tool_post_dispatch: dict[str, Callable[..., Decision]]
    observers: dict[str, Callable[..., None]]


# ─── Shared state ─────────────────────────────────────────────────────────

@dataclass
class GuardrailState:
    """All per-session state owned by guardrails.

    Session holds one of these and passes it to each guardrail's
    evaluate() call. Moving this out of Session keeps the ~15
    thrash-control fields together (they belong together) and makes the
    turn loop's responsibility clear: it orchestrates, the guardrails
    own their own state.
    """
    # default_factory uses maxlen=1 (not 0) so a direct GuardrailState()
    # construction in tests still produces a usable deque. The real
    # deque is replaced wholesale by init_guardrail_state(cfg) with
    # maxlen=cfg.duplicate_abort.
    recent_calls: deque = field(default_factory=lambda: deque(maxlen=1))
    consecutive_errors: dict[str, int] = field(default_factory=dict)
    same_class_error_signature: str = ""
    same_class_error_count: int = 0
    intent_block_count: int = 0
    intent_first_block_turn: int | None = None
    consecutive_intent_rejections: int = 0
    non_write_calls_since_write: int = 0
    same_target_key: str = ""
    same_target_display: str = ""
    same_target_count: int = 0
    same_target_nudge_emitted: bool = False
    rumination_gate: bool = False
    rumination_gate_grace: int = 0
    rumination_nudge_emitted: bool = False
    gate_block_count: int = 0
    has_mutated: bool = False
    verified_since_mutation: bool = False
    # Total done blocks in the session. Used by the done-loop failsafe:
    # after N blocks the session ends regardless of cause (parity flake,
    # novel verify-path drift, model genuinely misreading the gate). Bounds
    # the wasted-pt cost of any rejected-done loop to N round-trips.
    done_blocked_count: int = 0
    # Running count of successful write/edit operations. Used by the
    # regression observer to decide whether "PASSED→FAILED" on a test
    # had an intervening mutation (flaky-vs-caused-by-edit disambiguation).
    mutation_count: int = 0

    # Derived thresholds (computed at session init, read during the loop).
    # `rumination_nudge_threshold` is the PRE-first-mutation value.
    # `rumination_nudge_threshold_post_mutation` is used once has_mutated=True.
    # Both default to the same computed value if no asymmetric config is set.
    rumination_nudge_threshold: int = 0
    rumination_nudge_threshold_post_mutation: int = 0
    rumination_arm_threshold: int = 0

    # Pretest parity (done_guard ground truth — filled at session 1 start
    # by the harness parsing pretest output through the task format's
    # [output_parser]. Empty sets = pretest not parseable; done_guard
    # falls back to heuristic preconditions in that case).
    pretest_failing_tests: set[str] = field(default_factory=set)
    pretest_passing_tests: set[str] = field(default_factory=set)
    latest_test_parsed: dict[str, str] = field(default_factory=dict)
    green_parity_streak: int = 0
    test_file_reads: set[str] = field(default_factory=set)
    test_runs_without_test_read: int = 0
    last_test_target: str = ""
    test_read_nudge_target: str = ""

    # Regression observability (independent of pretest-parity mode).
    # Holds the prior test run's parsed verdicts and the mutation count
    # observed alongside it. Session compares incoming parsed verdicts
    # against prev_test_parsed to detect PASSED→FAILED transitions
    # with at least one intervening mutation.
    prev_test_parsed: dict[str, str] = field(default_factory=dict)
    mutation_count_at_prev_test: int = 0
    commit_pending: bool = False
    commit_source_path: str = ""
    commit_violation_count: int = 0
    commit_turns_since_arm: int = 0
    contract_block_sig: str = ""
    contract_block_count: int = 0
    recovery_mode_active: bool = False
    recovery_reason: str = ""
    recovery_target: str = ""
    recovery_turns_since_arm: int = 0
    verify_repeat_sig: str = ""
    verify_repeat_count: int = 0
    mutation_count_at_last_verify: int = 0
    mutation_repeat_sig: str = ""
    mutation_repeat_target: str = ""
    mutation_repeat_count: int = 0
    mutation_repeat_block_sig: str = ""
    mutation_repeat_block_count: int = 0
    # loop_detect guardrail (tighter than duplicate_guard, with a
    # recovery-inject step before hard abort). Tracks the current
    # consecutive-identical streak plus a one-shot flag that arms END
    # on the very next repeat after the WARN has been emitted.
    loop_detect_last_sig: tuple = ()
    loop_detect_streak: int = 0
    loop_detect_warned: bool = False


def init_guardrail_state(cfg: Any) -> GuardrailState:
    """Build a GuardrailState seeded from cfg at session start."""
    # Nudge threshold: absolute (if > 0) overrides percentage-of-max_turns.
    # Still clamped to the min-threshold floor so trivial max_turns values
    # don't collapse the nudge below usefulness.
    if cfg.rumination_nudge_threshold_abs > 0:
        nudge = max(cfg.rumination_min_threshold, cfg.rumination_nudge_threshold_abs)
    else:
        nudge = max(
            cfg.rumination_min_threshold,
            int(cfg.max_turns * cfg.rumination_nudge_threshold / 100),
        )
    # Post-mutation nudge threshold: separate knob. When > 0, overrides the
    # pre-mutation value for the rumination ladder's post-has_mutated checks.
    # When 0, post = pre (symmetric default).
    if cfg.rumination_nudge_threshold_abs_post_mutation > 0:
        nudge_post = max(cfg.rumination_min_threshold,
                         cfg.rumination_nudge_threshold_abs_post_mutation)
    else:
        nudge_post = nudge
        # Activation threshold: an absolute value overrides the percentage.
    # Absolute form decouples the gate from max_turns so reducing the
    # turn budget doesn't tighten the gate proportionally.
    if cfg.rumination_gate_arm_threshold_abs > 0:
        arm = max(nudge, cfg.rumination_gate_arm_threshold_abs)
    else:
        arm_pct = cfg.rumination_gate_arm_threshold or cfg.rumination_nudge_threshold
        arm = max(nudge, int(cfg.max_turns * arm_pct / 100))
    # Deque max length must tolerate duplicate_abort=0 (guardrail disabled).
    # maxlen=0 would make the deque never retain anything; treat 0 as "no
    # abort" by giving the deque a nominal length of 1 so it still appends
    # (the guardrail function will short-circuit on its enabled flag).
    deque_len = max(1, cfg.duplicate_abort)
    return GuardrailState(
        recent_calls=deque(maxlen=deque_len),
        rumination_nudge_threshold=nudge,
        rumination_nudge_threshold_post_mutation=nudge_post,
        rumination_arm_threshold=arm,
    )


# ─── Turn-level pre-dispatch guardrails ──────────────────────────────────
# These look at the whole turn (all tool_calls together) before any
# tool executes. Called once per turn.
