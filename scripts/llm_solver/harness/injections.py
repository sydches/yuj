"""Conditional injection-rule subsystem.

Borrowed in spirit from OpenHands' micro-agents
(``openhands/microagent/microagent.py``). User-authored markdown
files in ``.harness/injections/*.md`` declare fragments that inject
into the conversation either always-on (once at session start), when
a keyword appears in a user / tool-result message, or when a model
tool targets a matching project-relative path.

Every injection is visible in the conversation with a
``<injected-fragment source="{name}">`` wrapper so the agent knows
the content came from the harness, not the user or the model. Each
fire records an ``injection`` event on the savings ledger.

File format (TOML frontmatter + markdown body, split by ``+++``):

    +++
    name = "pytest-hint"
    trigger = "keyword"
    keywords = ["pytest", "py.test"]
    paths = ["tests/**/*.py"]
    repeat = false
    +++

    pytest's -q flag reduces per-test output; --tb=short truncates
    tracebacks.

Schema (enforced at load time; missing required key raises loudly):

    name       str               — ledger mechanism + source attribute
    trigger    "always"|"keyword"|"path" — optional legacy firing mode
    keywords   list[str]          — non-empty strings; keyword trigger
    paths      list[str]          — project-relative path globs
    repeat     bool               — optional per-rule repeat override
    fire_once  bool               — legacy inverse of repeat

Fragment content must apply to any task. Do not put a task ID in a
fragment. The loader checks task IDs, and reviewers check the rest.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from pathlib import PurePosixPath
from typing import Iterable, Mapping, Sequence

from .._shared.toml_compat import tomllib
from .prompt_imports import DEFAULT_IMPORT_MAX_DEPTH, process_imports


_FRONTMATTER_FENCE = "+++"
log = logging.getLogger(__name__)

# Reject task IDs that use the ``<org>__<repo>`` marker. The pattern allows
# ordinary project and framework names because general tool hints may use them.
_TASK_ID_PATTERN = re.compile(r"\b[A-Za-z][\w.-]*__[A-Za-z][\w.-]*\b")


@dataclass(frozen=True)
class Injection:
    """One parsed injection record loaded from a markdown file."""
    name: str
    trigger: str            # "always", "keyword", or "path"
    keywords: tuple[str, ...]
    fire_once: bool
    body: str
    source_path: str        # for debugging / trace only
    paths: tuple[str, ...] = ()
    repeat: bool | None = None

    def repeats_for(self, trigger: str, *, path_rule_repeat: bool = False) -> bool:
        """Return whether this rule re-fires for one trigger kind."""
        if self.repeat is not None:
            return self.repeat
        if trigger == "path":
            if not self.fire_once:
                return True
            return path_rule_repeat
        return not self.fire_once

    def format_block(self, *, trigger: str = "", path: str = "") -> str:
        """Wrap the body in a <injected-fragment> envelope."""
        if trigger == "path":
            opening = (
                f'<injected-fragment rule="{_xml_attr(self.name)}" '
                f'trigger="path" path="{_xml_attr(path)}" '
                f'source="{_xml_attr(self.name)}">'
            )
        else:
            # Preserve the original keyword/always envelope for existing
            # consumers. Conditional path rules add explicit rule/path fields.
            opening = f'<injected-fragment source="{_xml_attr(self.name)}">'
        return (
            f'{opening}\n'
            f'{self.body.rstrip()}\n'
            f'</injected-fragment>'
        )


def _xml_attr(value: str) -> str:
    """Escape a rule name or path for one XML attribute."""
    return (
        value.replace("&", "&amp;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _string_list(
    frontmatter: Mapping[str, object], key: str, *, source_path: str,
) -> tuple[str, ...]:
    """Read one strict TOML string array without scalar coercion."""
    value = frontmatter.get(key, [])
    if not isinstance(value, list):
        raise ValueError(f"{source_path}: {key!r} must be an array of strings")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError(
            f"{source_path}: {key!r} entries must be non-empty strings"
        )
    return tuple(item.strip() for item in value)


def _normalize_path_pattern(pattern: str, *, source_path: str) -> str:
    """Validate one project-relative POSIX glob and return its stable form."""
    normalized = pattern
    while normalized.startswith("./"):
        normalized = normalized[2:]
    posix = PurePosixPath(normalized)
    if (
        not normalized
        or normalized in {".", ".."}
        or posix.is_absolute()
        or ".." in posix.parts
        or normalized.endswith("/")
        or "\x00" in normalized
        or "\n" in normalized
    ):
        raise ValueError(
            f"{source_path}: path glob {pattern!r} must name files within "
            "the project root"
        )
    stable = posix.as_posix()
    _compiled_path_glob(stable)
    return stable


@lru_cache(maxsize=1024)
def _compiled_path_glob(pattern: str):
    """Compile the repository's slash-aware ``*``/``**`` glob syntax."""
    from .sandbox.ignore_policy import _glob_regex
    return _glob_regex(pattern)


def parse_injection(text: str, *, source_path: str) -> Injection:
    """Parse one markdown file's contents into an Injection.

    Raises ValueError on malformed frontmatter or missing required
    keys. No silent fallbacks — the loader surfaces config errors
    per the no-bullshit policy.
    """
    parts = text.split(_FRONTMATTER_FENCE, 2)
    if len(parts) < 3:
        raise ValueError(
            f"{source_path}: missing {_FRONTMATTER_FENCE!r} frontmatter fences"
        )
    if parts[0].strip():
        raise ValueError(
            f"{source_path}: frontmatter must start with "
            f"{_FRONTMATTER_FENCE!r}"
        )
    # parts[0] is the empty prefix before the opening fence.
    frontmatter_src = parts[1].strip()
    body = _FRONTMATTER_FENCE.join(parts[2:]).strip()
    try:
        fm = tomllib.loads(frontmatter_src)
    except Exception as e:
        raise ValueError(f"{source_path}: invalid TOML frontmatter: {e}")
    if "name" not in fm:
        raise ValueError(f"{source_path}: missing required key 'name'")
    name = fm["name"]
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"{source_path}: 'name' must be a non-empty string")
    name = name.strip()

    keywords = _string_list(fm, "keywords", source_path=source_path)
    raw_paths = _string_list(fm, "paths", source_path=source_path)
    paths = tuple(
        _normalize_path_pattern(pattern, source_path=source_path)
        for pattern in raw_paths
    )

    raw_trigger = fm.get("trigger")
    if raw_trigger is not None and not isinstance(raw_trigger, str):
        raise ValueError(f"{source_path}: 'trigger' must be a string")
    trigger = raw_trigger or (
        "keyword" if keywords else "path" if paths else "always"
    )
    if trigger not in ("always", "keyword", "path"):
        raise ValueError(
            f"{source_path}: invalid trigger {trigger!r} "
            f"(expected 'always', 'keyword', or 'path')"
        )
    if trigger == "keyword" and not keywords:
        raise ValueError(
            f"{source_path}: trigger='keyword' requires non-empty keywords list"
        )
    if trigger == "path" and not paths:
        raise ValueError(
            f"{source_path}: trigger='path' requires non-empty paths list"
        )
    if trigger == "always" and (keywords or paths):
        raise ValueError(
            f"{source_path}: trigger='always' cannot declare paths or keywords"
        )

    if "repeat" in fm and not isinstance(fm["repeat"], bool):
        raise ValueError(f"{source_path}: 'repeat' must be true or false")
    if "fire_once" in fm and not isinstance(fm["fire_once"], bool):
        raise ValueError(f"{source_path}: 'fire_once' must be true or false")
    if "repeat" in fm and "fire_once" in fm:
        raise ValueError(
            f"{source_path}: declare 'repeat' or legacy 'fire_once', not both"
        )
    repeat = fm.get("repeat")
    if repeat is not None:
        fire_once = not repeat
    else:
        fire_once = fm.get("fire_once", True)
        if "fire_once" in fm:
            repeat = not fire_once
    _assert_task_agnostic(body, keywords, source_path=source_path)
    return Injection(
        name=name, trigger=trigger, keywords=keywords,
        fire_once=fire_once, body=body, source_path=source_path,
        paths=paths, repeat=repeat,
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


@dataclass(frozen=True, slots=True)
class LoadedInjections:
    """Resolved fragments plus safe session-start import provenance."""

    injections: tuple[Injection, ...]
    prompt_import_tree: tuple[dict[str, object], ...]


def _safe_source(path: Path, roots: Sequence[Path]) -> str:
    resolved = path.resolve(strict=False)
    for root in roots:
        try:
            return resolved.relative_to(root.resolve(strict=False)).as_posix()
        except ValueError:
            continue
    return f"<outside-allowed-dirs>/{path.name}"


def load_injections_with_metadata(
    dir_path: Path,
    *,
    imports_enabled: bool = False,
    imports_max_depth: int = DEFAULT_IMPORT_MAX_DEPTH,
    allowed_dirs: Sequence[Path] | None = None,
    unreadable_paths: Sequence[str] = (),
) -> LoadedInjections:
    """Load fragments once and retain safe import metadata for the trace."""
    if not dir_path.is_dir():
        return LoadedInjections((), ())
    roots = tuple(allowed_dirs or (dir_path,))
    injections: list[Injection] = []
    import_tree: list[dict[str, object]] = []
    for path in sorted(dir_path.glob("*.md")):
        raw = path.read_bytes()
        text = raw.decode("utf-8-sig", errors="replace")
        source = _safe_source(path, roots)
        tree: list[dict[str, object]] = []
        imported_bytes = 0
        if imports_enabled:
            processed = process_imports(
                text,
                path.parent,
                roots,
                max_depth=imports_max_depth,
                source_path=path,
                unreadable_paths=unreadable_paths,
            )
            text = processed.content
            tree = processed.trace_tree()
            imported_bytes = processed.imported_bytes
        injection = parse_injection(text, source_path=source)
        injections.append(injection)
        armed_triggers = []
        if injection.paths:
            armed_triggers.append("path")
        if injection.keywords:
            armed_triggers.append("keyword")
        if not armed_triggers:
            armed_triggers.append("always")
        log.info(
            "injection armed: rule=%s triggers=%s source=%s",
            injection.name,
            ",".join(armed_triggers),
            source,
        )
        import_tree.append(
            {
                "owner": "injection",
                "source": source,
                "source_bytes": len(raw),
                "imported_bytes": imported_bytes,
                "imports": tree,
            }
        )
    return LoadedInjections(tuple(injections), tuple(import_tree))


def load_injections(
    dir_path: Path,
    *,
    imports_enabled: bool = False,
    imports_max_depth: int = DEFAULT_IMPORT_MAX_DEPTH,
    allowed_dirs: Sequence[Path] | None = None,
    unreadable_paths: Sequence[str] = (),
) -> list[Injection]:
    """Load every ``*.md`` in ``dir_path`` as an Injection list.

    Missing directory returns an empty list — running without any
    declared injections is a first-class configuration. Files that
    fail to parse raise ValueError (loud, no silent skip).

    Fire order is **alphabetical filename order** (`sorted(*.md)`).
    Multiple injections firing on the same turn append to the user
    message log in that order; rename files to control sequencing.
    """
    loaded = load_injections_with_metadata(
        dir_path,
        imports_enabled=imports_enabled,
        imports_max_depth=imports_max_depth,
        allowed_dirs=allowed_dirs,
        unreadable_paths=unreadable_paths,
    )
    return list(loaded.injections)


@dataclass
class InjectionState:
    """Per-session injection firing state.

    fired_names is a set of Injection.name values that have already
    injected this session (used to enforce fire_once).
    """
    fired_names: set[str] = field(default_factory=set)


@dataclass(frozen=True, slots=True)
class PathInjectionFire:
    """One path-triggered rule fire and its safe canonical target path."""

    injection: Injection
    path: str


@dataclass(frozen=True, slots=True)
class _PathTarget:
    """Canonical trace label plus lexical/resolved glob candidates."""

    path: str
    candidates: tuple[str, ...]


def _normalize_tool_target(raw_path: str, *, cwd: str) -> _PathTarget | None:
    """Resolve one model-supplied path exactly like the file tools do."""
    if not isinstance(raw_path, str) or not raw_path or "\x00" in raw_path:
        return None
    from ._tools._common import _resolve
    try:
        resolved = _resolve(cwd, raw_path)
        cwd_resolved = Path(cwd).resolve()
        canonical = resolved.relative_to(cwd_resolved).as_posix()
    except (OSError, ValueError):
        return None
    if canonical in {"", "."}:
        return None

    # Match both the spelling the model used and the symlink-resolved target.
    # The event path is always canonical, so neither absolute host paths nor a
    # symlink alias enter the durable trace by accident.
    lexical = canonical
    try:
        cwd_lexical = Path(os.path.abspath(cwd))
        supplied = Path(raw_path)
        supplied_abs = Path(os.path.abspath(
            supplied if supplied.is_absolute() else cwd_lexical / supplied
        ))
        lexical = supplied_abs.relative_to(cwd_lexical).as_posix()
    except (OSError, ValueError):
        pass
    candidates = tuple(dict.fromkeys((lexical, canonical)))
    return _PathTarget(path=canonical, candidates=candidates)


def path_targets_for_tool(
    tool_name: str,
    arguments: Mapping[str, object],
    *,
    cwd: str,
    applied_operations: Sequence[Sequence[object]] = (),
    bash_rewritten: bool = False,
) -> tuple[_PathTarget, ...]:
    """Return deterministic file targets eligible for path-rule matching.

    Typed file tools use their ``path`` argument. ``apply_patch`` uses only
    operations that the transactional patch handler reported as applied.
    Bash is eligible only when the existing fail-closed classifier proves the
    original command reads one explicit file.
    """
    raw_paths: list[str] = []
    if tool_name in {"read", "edit", "write"}:
        value = arguments.get("path")
        if isinstance(value, str):
            raw_paths.append(value)
    elif tool_name == "apply_patch":
        for operation in applied_operations:
            if len(operation) >= 2 and isinstance(operation[1], str):
                raw_paths.append(operation[1])
    elif tool_name == "bash" and not bash_rewritten:
        from .stale_guard import classify_single_file_read
        command = arguments.get("cmd")
        shell_read = (
            classify_single_file_read(command)
            if isinstance(command, str) else None
        )
        if shell_read is not None:
            raw_paths.append(shell_read.path)

    targets: list[_PathTarget] = []
    seen: set[str] = set()
    for raw_path in raw_paths:
        target = _normalize_tool_target(raw_path, cwd=cwd)
        if target is None or target.path in seen:
            continue
        seen.add(target.path)
        targets.append(target)
    return tuple(targets)


def fire_path_candidates(
    injections: Iterable[Injection],
    *,
    tool_name: str,
    arguments: Mapping[str, object],
    cwd: str,
    state: InjectionState,
    path_rule_repeat: bool = False,
    applied_operations: Sequence[Sequence[object]] = (),
    bash_rewritten: bool = False,
) -> list[PathInjectionFire]:
    """Claim matching path rules in stable rule order.

    One rule fires at most once for one tool result even when a multi-file
    patch contains several matching paths. Its first matching canonical path
    becomes the visible fragment and trace label.
    """
    targets = path_targets_for_tool(
        tool_name,
        arguments,
        cwd=cwd,
        applied_operations=applied_operations,
        bash_rewritten=bash_rewritten,
    )
    if not targets:
        return []
    fired: list[PathInjectionFire] = []
    for injection in injections:
        if not injection.paths:
            continue
        repeats = injection.repeats_for(
            "path", path_rule_repeat=path_rule_repeat,
        )
        if not repeats and injection.name in state.fired_names:
            continue
        matching_target = next((
            target
            for target in targets
            if any(
                _compiled_path_glob(pattern).fullmatch(candidate) is not None
                for pattern in injection.paths
                for candidate in target.candidates
            )
        ), None)
        if matching_target is None:
            continue
        fired.append(PathInjectionFire(injection, matching_target.path))
        if not repeats:
            state.fired_names.add(injection.name)
    return fired


def match(injection: Injection, text: str) -> bool:
    """Return True when ``injection`` should fire against ``text``.

    Always-on injections match on empty text too (used for session-
    start always-on fires). Keyword match is case-insensitive
    substring.
    """
    if not injection.keywords and not injection.paths:
        return True
    if not injection.keywords:
        return False
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
        trigger = "keyword" if inj.keywords else "always"
        repeats = inj.repeats_for(trigger)
        if not repeats and inj.name in state.fired_names:
            continue
        if not match(inj, text):
            continue
        fired.append(inj)
        if not repeats:
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
