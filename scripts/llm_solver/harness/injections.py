"""Keyword-triggered injection subsystem.

Borrowed in spirit from OpenHands' micro-agents
(``openhands/microagent/microagent.py``). User-authored markdown
files in ``.harness/injections/*.md`` declare fragments that inject
into the conversation either always-on (once at session start) or
when a keyword appears in a user / tool-result message.

Every injection is visible in the conversation with a
``<injected-fragment source="{name}">`` wrapper so the agent knows
the content came from the harness, not the user or the model. Each
fire records an ``injection`` event on the savings ledger.

File format (TOML frontmatter + markdown body, split by ``+++``):

    +++
    name = "pytest-hint"
    trigger = "keyword"
    keywords = ["pytest", "py.test"]
    fire_once = true
    +++

    pytest's -q flag reduces per-test output; --tb=short truncates
    tracebacks.

Schema (enforced at load time; missing required key raises loudly):

    name       str               — ledger mechanism + source attribute
    trigger    "always"|"keyword" — firing mode
    keywords   list[str]          — required when trigger == "keyword"
    fire_once  bool               — default true; when false, every
                                    matching turn re-fires

Fragment content must apply to any task. Do not put a task ID in a
fragment. The loader checks task IDs, and reviewers check the rest.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .._shared.toml_compat import tomllib


_FRONTMATTER_FENCE = "+++"

# Reject task IDs that use the ``<org>__<repo>`` marker. The pattern allows
# ordinary project and framework names because general tool hints may use them.
_TASK_ID_PATTERN = re.compile(r"\b[A-Za-z][\w.-]*__[A-Za-z][\w.-]*\b")


@dataclass(frozen=True)
class Injection:
    """One parsed injection record loaded from a markdown file."""
    name: str
    trigger: str            # "always" or "keyword"
    keywords: tuple[str, ...]
    fire_once: bool
    body: str
    source_path: str        # for debugging / trace only

    def format_block(self) -> str:
        """Wrap the body in a <injected-fragment> envelope."""
        return (
            f'<injected-fragment source="{self.name}">\n'
            f'{self.body.rstrip()}\n'
            f'</injected-fragment>'
        )


def parse_injection(text: str, *, source_path: str) -> Injection:
    """Parse one markdown file's contents into an Injection.

    Raises ValueError on malformed frontmatter or missing required
    keys. No silent fallbacks — the loader surfaces config errors
    per the no-bullshit policy.
    """
    parts = text.split(_FRONTMATTER_FENCE)
    if len(parts) < 3:
        raise ValueError(
            f"{source_path}: missing {_FRONTMATTER_FENCE!r} frontmatter fences"
        )
    # parts[0] is the empty prefix before the opening fence.
    frontmatter_src = parts[1].strip()
    body = _FRONTMATTER_FENCE.join(parts[2:]).strip()
    try:
        fm = tomllib.loads(frontmatter_src)
    except Exception as e:
        raise ValueError(f"{source_path}: invalid TOML frontmatter: {e}")
    for key in ("name", "trigger"):
        if key not in fm:
            raise ValueError(f"{source_path}: missing required key {key!r}")
    name = str(fm["name"])
    trigger = str(fm["trigger"])
    if trigger not in ("always", "keyword"):
        raise ValueError(
            f"{source_path}: invalid trigger {trigger!r} "
            f"(expected 'always' or 'keyword')"
        )
    keywords_list = fm.get("keywords", [])
    if trigger == "keyword" and not keywords_list:
        raise ValueError(
            f"{source_path}: trigger='keyword' requires non-empty keywords list"
        )
    keywords = tuple(str(k) for k in keywords_list)
    fire_once = bool(fm.get("fire_once", True))
    _assert_task_agnostic(body, keywords, source_path=source_path)
    return Injection(
        name=name, trigger=trigger, keywords=keywords,
        fire_once=fire_once, body=body, source_path=source_path,
    )


def _assert_task_agnostic(
    body: str, keywords: tuple[str, ...], *, source_path: str,
) -> None:
    """Reject content that names a specific task ID.

    Check the fragment body and each keyword for the ``<org>__<repo>``
    marker. Run this check while loading so invalid text does not reach
    the conversation.
    """
    body_hit = _TASK_ID_PATTERN.search(body)
    if body_hit:
        raise ValueError(
            f"{source_path}: injection body contains task-id pattern "
            f"{body_hit.group()!r} (LEAKAGE_RULES: content must be "
            f"task-agnostic — no <org>__<repo> markers)"
        )
    for kw in keywords:
        kw_hit = _TASK_ID_PATTERN.search(kw)
        if kw_hit:
            raise ValueError(
                f"{source_path}: keyword {kw!r} contains task-id pattern "
                f"(LEAKAGE_RULES: keywords must be task-agnostic)"
            )


def load_injections(dir_path: Path) -> list[Injection]:
    """Load every ``*.md`` in ``dir_path`` as an Injection list.

    Missing directory returns an empty list — running without any
    declared injections is a first-class configuration. Files that
    fail to parse raise ValueError (loud, no silent skip).

    Fire order is **alphabetical filename order** (`sorted(*.md)`).
    Multiple injections firing on the same turn append to the user
    message log in that order; rename files to control sequencing.
    """
    if not dir_path.is_dir():
        return []
    injections: list[Injection] = []
    for path in sorted(dir_path.glob("*.md")):
        text = path.read_text()
        injections.append(parse_injection(text, source_path=str(path)))
    return injections


@dataclass
class InjectionState:
    """Per-session injection firing state.

    fired_names is a set of Injection.name values that have already
    injected this session (used to enforce fire_once).
    """
    fired_names: set[str] = field(default_factory=set)


def match(injection: Injection, text: str) -> bool:
    """Return True when ``injection`` should fire against ``text``.

    Always-on injections match on empty text too (used for session-
    start always-on fires). Keyword match is case-insensitive
    substring.
    """
    if injection.trigger == "always":
        return True
    lower = text.lower()
    return any(k.lower() in lower for k in injection.keywords)


def fire_candidates(
    injections: Iterable[Injection],
    *,
    text: str,
    state: InjectionState,
) -> list[Injection]:
    """Return the injections that should fire for ``text``, updating
    ``state.fired_names`` to reflect the fire_once contract.

    Called by the harness immediately before sending the next API
    request; the resulting Injection list becomes a list of
    ``<injected-fragment>`` blocks appended to the outbound context.
    """
    fired: list[Injection] = []
    for inj in injections:
        if inj.fire_once and inj.name in state.fired_names:
            continue
        if not match(inj, text):
            continue
        fired.append(inj)
        if inj.fire_once:
            state.fired_names.add(inj.name)
    return fired


def record_fire(name: str, *, body_chars: int, match_mode: str) -> None:
    """Ledger helper — emit an injection event.

    bucket = "injection"; mechanism = the injection's name so the
    aggregator groups by (bucket, name).
    """
    from .savings import get_ledger
    get_ledger().record(
        bucket="injection",
        layer="harness",
        mechanism=name,
        input_chars=0,
        output_chars=int(body_chars),
        measure_type="exact",
        ctx={"match_mode": match_mode},
    )


# ── system-reminder envelope helper ──────────────────────────────────
#
# Three call-sites already wrap harness-injected directives in
# `<system-reminder>...</system-reminder>` (config.toml::
# read_truncated_reminder, read_empty_reminder, loop_detect_recovery).
# This helper formalises the shape so future call sites use the same
# wrapper consistently.
#
# The envelope establishes a stable boundary the model can be told about
# in its system prompt: anything inside <system-reminder> is the
# harness, NOT the user. That distinction matters because some of the
# reminders (truncation warnings, recovery nudges) are conditional on
# behaviour the model itself produced, and the model's decision rule
# for "did the user say X?" should not be triggered by those.
#
# Composable: any guardrail / transform call attach_reminder(text)
# instead of constructing the markup string itself, so a future
# rename / format change is a one-edit operation.

_SYSTEM_REMINDER_OPEN = "<system-reminder>"
_SYSTEM_REMINDER_CLOSE = "</system-reminder>"


def attach_reminder(text: str) -> str:
    """Wrap *text* in the canonical <system-reminder>…</system-reminder>.

    Idempotent — already-wrapped text passes through unchanged so
    callers that read templates from config (which already include the
    markup) don't double-wrap. Whitespace at the boundaries is
    preserved by the caller's choice; this helper does not strip.
    """
    s = text or ""
    if s.startswith(_SYSTEM_REMINDER_OPEN) and s.endswith(_SYSTEM_REMINDER_CLOSE):
        return s
    return f"{_SYSTEM_REMINDER_OPEN}{s}{_SYSTEM_REMINDER_CLOSE}"
