"""Pattern-based prompt-injection scanning for untrusted model inputs.

The scanner owns detection and rendering only.  ``tools.dispatch`` decides
when arguments and results cross the model/tool boundary, while the outer
driver applies the same registry to imported instruction files.  Findings
never retain or trace matched text.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import re
from typing import Iterable, Iterator, Mapping
from uuid import uuid4

from .._shared.toml_compat import tomllib


SECURITY_PATTERN_SCHEMA_VERSION = 1
SECURITY_FINDING_VERSION = "1"
SECURITY_SCAN_MODES = frozenset({"off", "flag", "block"})
SECURITY_SCAN_STAGES = frozenset({"args", "result"})
_IDENTIFIER_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_SECURITY_BLOCK_STAGE_RE = re.compile(
    r'^<tool_result\b[^>]*\berror_kind="security_block"'
    r'[^>]*\bsecurity_stage="(?P<stage>args|result)"'
)


class SecurityPatternError(ValueError):
    """A security registry or security knob is invalid."""


class SecurityScanBlocked(RuntimeError):
    """Imported instruction content was blocked before a model call."""


@dataclass(frozen=True, slots=True)
class SecurityPattern:
    """One compiled registry rule."""

    rule: str
    finding_class: str
    stages: frozenset[str]
    expression: str
    compiled: re.Pattern[str]


@dataclass(frozen=True, slots=True)
class SecurityFinding:
    """One value-free detection result."""

    id: str
    rule: str
    stage: str
    action: str

    def trace_fields(self) -> dict[str, str]:
        """Return the complete public ``security_finding`` payload."""
        return {
            "id": self.id,
            "rule": self.rule,
            "stage": self.stage,
            "action": self.action,
        }

    def marker(self) -> str:
        """Return the model-visible, value-free finding marker."""
        return (
            f'<security-finding id="{_xml_attr(self.id)}" '
            f'rule="{_xml_attr(self.rule)}" stage="{self.stage}" '
            f'action="{self.action}" v="{SECURITY_FINDING_VERSION}"/>'
        )


@dataclass(frozen=True, slots=True)
class SecurityScanOutcome:
    """Findings and the aggregate decision for one scanned boundary."""

    findings: tuple[SecurityFinding, ...] = ()

    @property
    def blocked(self) -> bool:
        return any(finding.action == "block" for finding in self.findings)


@dataclass(frozen=True, slots=True)
class SecurityPatternRegistry:
    """Immutable ordered pattern registry."""

    source: Path
    patterns: tuple[SecurityPattern, ...]

    def matching_rules(self, text: str, *, stage: str) -> tuple[SecurityPattern, ...]:
        """Return each matching rule once, in registry order."""
        if stage not in SECURITY_SCAN_STAGES:
            raise ValueError(f"unknown security scan stage: {stage!r}")
        return tuple(
            pattern
            for pattern in self.patterns
            if stage in pattern.stages and pattern.compiled.search(text) is not None
        )


@dataclass(frozen=True, slots=True)
class SecurityScanner:
    """Apply one immutable registry under the resolved scan policy."""

    mode: str
    block_classes: frozenset[str]
    registry: SecurityPatternRegistry | None

    @classmethod
    def from_config(cls, cfg) -> "SecurityScanner":
        mode = str(getattr(cfg, "security_scan_mode", "flag"))
        block_classes = frozenset(
            getattr(cfg, "security_block_classes", ()) or ()
        )
        if mode == "off":
            return cls(mode=mode, block_classes=block_classes, registry=None)
        path = resolve_patterns_path(
            str(
                getattr(
                    cfg,
                    "security_patterns_file",
                    "security/patterns.toml",
                )
            )
        )
        return cls(
            mode=mode,
            block_classes=block_classes,
            registry=load_pattern_registry(path),
        )

    def scan_arguments(self, arguments: Mapping[str, object]) -> SecurityScanOutcome:
        """Scan string values in a tool argument object."""
        return self._scan_many(_iter_string_values(arguments), stage="args")

    def scan_text(self, text: object, *, stage: str = "result") -> SecurityScanOutcome:
        """Scan one text boundary."""
        return self._scan_many((str(text),), stage=stage)

    def _scan_many(
        self,
        texts: Iterable[str],
        *,
        stage: str,
    ) -> SecurityScanOutcome:
        if self.mode == "off" or self.registry is None:
            return SecurityScanOutcome()
        matched: dict[str, SecurityPattern] = {}
        for text in texts:
            for pattern in self.registry.matching_rules(text, stage=stage):
                matched.setdefault(pattern.rule, pattern)
        findings = tuple(
            SecurityFinding(
                id=f"SEC-{uuid4()}",
                rule=pattern.rule,
                stage=stage,
                action=(
                    "block"
                    if self.mode == "block"
                    and pattern.finding_class in self.block_classes
                    else "flag"
                ),
            )
            for pattern in matched.values()
        )
        return SecurityScanOutcome(findings=findings)


def resolve_patterns_path(path: str) -> Path:
    """Resolve a registry path relative to the checked-in config owner."""
    if not isinstance(path, str) or not path.strip():
        raise SecurityPatternError(
            "config error: security.patterns_file must be a non-empty string."
        )
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        # Keep relative registry paths under the same root that owns the
        # active config.toml, including an explicit YUJ_CONFIG root.
        from ..config import resolve_project_path

        candidate = resolve_project_path(candidate)
    return candidate.resolve()


@lru_cache(maxsize=32)
def load_pattern_registry(path: str | Path) -> SecurityPatternRegistry:
    """Load and compile one TOML pattern registry."""
    source = Path(path)
    try:
        with source.open("rb") as handle:
            data = tomllib.load(handle)
    except OSError as exc:
        raise SecurityPatternError(
            f"security pattern registry is not readable: {source}: {exc}"
        ) from exc

    allowed_top = {"schema_version", "pattern"}
    unknown_top = set(data) - allowed_top
    if unknown_top:
        raise SecurityPatternError(
            f"{source}: unknown top-level key(s): {sorted(unknown_top)}"
        )
    if data.get("schema_version") != SECURITY_PATTERN_SCHEMA_VERSION:
        raise SecurityPatternError(
            f"{source}: schema_version must be "
            f"{SECURITY_PATTERN_SCHEMA_VERSION}."
        )
    raw_patterns = data.get("pattern")
    if not isinstance(raw_patterns, list) or not raw_patterns:
        raise SecurityPatternError(
            f"{source}: [[pattern]] must contain at least one rule."
        )

    patterns: list[SecurityPattern] = []
    seen_rules: set[str] = set()
    allowed_pattern = {"rule", "class", "stages", "regex"}
    for index, raw in enumerate(raw_patterns, start=1):
        label = f"{source}: pattern {index}"
        if not isinstance(raw, dict):
            raise SecurityPatternError(f"{label} must be a TOML table.")
        unknown = set(raw) - allowed_pattern
        if unknown:
            raise SecurityPatternError(
                f"{label} has unknown key(s): {sorted(unknown)}"
            )
        rule = raw.get("rule")
        finding_class = raw.get("class")
        stages = raw.get("stages")
        expression = raw.get("regex")
        if not isinstance(rule, str) or _IDENTIFIER_RE.fullmatch(rule) is None:
            raise SecurityPatternError(
                f"{label}.rule must be a lowercase snake_case identifier."
            )
        if rule in seen_rules:
            raise SecurityPatternError(f"{source}: duplicate rule {rule!r}.")
        if (
            not isinstance(finding_class, str)
            or _IDENTIFIER_RE.fullmatch(finding_class) is None
        ):
            raise SecurityPatternError(
                f"{label}.class must be a lowercase snake_case identifier."
            )
        if (
            not isinstance(stages, list)
            or not stages
            or any(not isinstance(stage, str) for stage in stages)
        ):
            raise SecurityPatternError(
                f"{label}.stages must be a non-empty array of strings."
            )
        stage_set = frozenset(stages)
        if not stage_set <= SECURITY_SCAN_STAGES:
            raise SecurityPatternError(
                f"{label}.stages may contain only 'args' and 'result'."
            )
        if not isinstance(expression, str) or not expression:
            raise SecurityPatternError(f"{label}.regex must be a non-empty string.")
        try:
            compiled = re.compile(expression)
        except re.error as exc:
            raise SecurityPatternError(f"{label}.regex is invalid: {exc}") from exc
        if compiled.search("") is not None:
            raise SecurityPatternError(f"{label}.regex must not match empty text.")
        seen_rules.add(rule)
        patterns.append(
            SecurityPattern(
                rule=rule,
                finding_class=finding_class,
                stages=stage_set,
                expression=expression,
                compiled=compiled,
            )
        )
    return SecurityPatternRegistry(source=source, patterns=tuple(patterns))


def validate_security_settings(
    mode: object,
    patterns_file: object,
    block_classes: object,
) -> None:
    """Validate public security knobs and the active registry."""
    if not isinstance(mode, str) or mode not in SECURITY_SCAN_MODES:
        raise SecurityPatternError(
            "config error: security.scan_mode must be 'off', 'flag', or "
            f"'block', got {mode!r}."
        )
    if (
        not isinstance(block_classes, (list, tuple))
        or any(not isinstance(item, str) for item in block_classes)
    ):
        raise SecurityPatternError(
            "config error: security.block_classes must be an array of strings."
        )
    for finding_class in block_classes:
        if _IDENTIFIER_RE.fullmatch(finding_class) is None:
            raise SecurityPatternError(
                "config error: security.block_classes entries must be "
                "lowercase snake_case identifiers."
            )
    path = resolve_patterns_path(patterns_file)  # type: ignore[arg-type]
    if mode != "off":
        registry = load_pattern_registry(path)
        if mode == "block":
            registry_classes = {
                pattern.finding_class for pattern in registry.patterns
            }
            unknown_classes = set(block_classes) - registry_classes
            if unknown_classes:
                raise SecurityPatternError(
                    "config error: security.block_classes names class(es) "
                    f"absent from the active registry: {sorted(unknown_classes)}"
                )


def emit_findings(findings: Iterable[SecurityFinding], sink) -> None:
    """Best-effort trace emission through the caller-owned event sink."""
    if sink is None:
        return
    for finding in findings:
        sink({"event": "security_finding", **finding.trace_fields()})


def render_security_block(
    tool_name: str,
    outcome: SecurityScanOutcome,
) -> str:
    """Render a value-free error envelope for one blocked boundary."""
    if not outcome.blocked:
        raise ValueError("a non-blocking scan outcome has no block envelope")
    stage = next(
        finding.stage
        for finding in outcome.findings
        if finding.action == "block"
    )
    markers = "\n".join(finding.marker() for finding in outcome.findings)
    boundary = "arguments before execution" if stage == "args" else "result"
    return (
        f'<tool_result tool_name="{_xml_attr(tool_name)}" status="error" '
        f'error_kind="security_block" security_stage="{stage}" v="1">\n'
        f"{markers}\n"
        f"ERROR: security scan blocked the tool {boundary}.\n"
        "</tool_result>"
    )


def prepend_finding_markers(
    text: str,
    findings: Iterable[SecurityFinding],
) -> str:
    """Prepend value-free markers to imported instruction content."""
    markers = "\n".join(finding.marker() for finding in findings)
    if not markers:
        return text
    return f"{markers}\n{text}"


def security_block_stage(result: object) -> str | None:
    """Return the stage for a security block envelope, if present."""
    match = _SECURITY_BLOCK_STAGE_RE.match(str(result or ""))
    return match.group("stage") if match is not None else None


def _iter_string_values(value: object) -> Iterator[str]:
    """Yield argument string values without scanning JSON field names."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for child in value.values():
            yield from _iter_string_values(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _iter_string_values(child)


def _xml_attr(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace('"', "&quot;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


__all__ = [
    "SecurityFinding",
    "SecurityPatternError",
    "SecurityPatternRegistry",
    "SecurityScanBlocked",
    "SecurityScanOutcome",
    "SecurityScanner",
    "emit_findings",
    "load_pattern_registry",
    "prepend_finding_markers",
    "render_security_block",
    "resolve_patterns_path",
    "security_block_stage",
    "validate_security_settings",
]
