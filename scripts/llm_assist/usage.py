"""Read-only aggregation and rendering for assistant session usage."""
from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from typing import Iterable


class UsageEvidenceError(ValueError):
    """Persisted session usage evidence is corrupt or contradictory."""


@dataclass(frozen=True)
class CostEvidence:
    amount: Decimal
    currency: str


@dataclass(frozen=True)
class QuotaEvidence:
    remaining: Decimal
    limit: Decimal
    unit: str
    scope: str


@dataclass(frozen=True)
class SessionUsage:
    segments: int
    input_tokens: int | None
    output_tokens: int | None
    cached_tokens: int | None
    cache_ratio: Fraction | None
    cost: CostEvidence | None
    quota: QuotaEvidence | None


@dataclass(frozen=True)
class _SegmentUsage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    cached_tokens: int | None = None
    cost: CostEvidence | None = None
    quota: QuotaEvidence | None = None


def aggregate_session_usage(trace_paths: Iterable[Path]) -> SessionUsage:
    """Aggregate immutable segment facts from each distinct trace path."""
    segment_numbers: set[int] = set()
    facts: dict[int, _SegmentUsage] = {}

    for path in _distinct_paths(trace_paths):
        for line_number, event in _read_events(path):
            event_type = event.get("event")
            if event_type in {
                "session_start",
                "session_end",
                "session_usage",
                "turn",
            } and "session_number" in event:
                segment_numbers.add(
                    _segment_number(event.get("session_number"), path, line_number)
                )
            if event_type != "session_usage":
                continue
            segment = _segment_number(
                event.get("session_number"), path, line_number
            )
            if event.get("scope") != "all_model_responses":
                raise UsageEvidenceError(
                    f"{path}:{line_number}: session_usage has incompatible scope"
                )
            fact = _parse_segment(event, path, line_number)
            previous = facts.get(segment)
            if previous is not None and previous != fact:
                raise UsageEvidenceError(
                    f"conflicting session_usage facts for segment {segment}"
                )
            facts[segment] = fact
            segment_numbers.add(segment)

    expected_numbers = (
        range(1, max(segment_numbers) + 1) if segment_numbers else range(0)
    )
    ordered = [facts.get(number, _SegmentUsage()) for number in expected_numbers]
    input_tokens = _sum_known(ordered, "input_tokens")
    output_tokens = _sum_known(ordered, "output_tokens")
    cached_tokens = _sum_known(ordered, "cached_tokens")
    cache_ratio = (
        Fraction(cached_tokens, input_tokens)
        if cached_tokens is not None
        and input_tokens is not None
        and input_tokens > 0
        else None
    )
    return SessionUsage(
        segments=len(ordered),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_tokens=cached_tokens,
        cache_ratio=cache_ratio,
        cost=_aggregate_cost(ordered),
        quota=_latest_compatible_quota(ordered),
    )


def render_session_usage(usage: SessionUsage) -> list[str]:
    """Render one stable plain-terminal report from a typed aggregate."""
    return [
        f"segments: {usage.segments}",
        f"input_tokens: {_known(usage.input_tokens)}",
        f"output_tokens: {_known(usage.output_tokens)}",
        f"cached_tokens: {_known(usage.cached_tokens)}",
        "cache_ratio: " + (
            _percent_text(usage.cache_ratio)
            if usage.cache_ratio is not None
            else "unknown"
        ),
        "cost: " + (
            f"{_decimal_text(usage.cost.amount)} {usage.cost.currency}"
            if usage.cost is not None
            else "unknown"
        ),
        "quota: " + (
            f"{_decimal_text(usage.quota.remaining)}/"
            f"{_decimal_text(usage.quota.limit)} {usage.quota.unit} "
            f"remaining ({usage.quota.scope})"
            if usage.quota is not None
            else "unknown"
        ),
    ]


def _distinct_paths(paths: Iterable[Path]) -> list[Path]:
    distinct: dict[Path, Path] = {}
    for raw_path in paths:
        path = Path(raw_path)
        distinct.setdefault(path.resolve(strict=False), path)
    return list(distinct.values())


def _read_events(path: Path):
    if not path.is_file():
        return
    try:
        with path.open(encoding="utf-8") as trace_file:
            for line_number, raw_line in enumerate(trace_file, start=1):
                if not raw_line.strip():
                    continue
                try:
                    event = json.loads(raw_line)
                except json.JSONDecodeError as exc:
                    raise UsageEvidenceError(
                        f"{path}:{line_number}: invalid JSON"
                    ) from exc
                if not isinstance(event, dict):
                    raise UsageEvidenceError(
                        f"{path}:{line_number}: trace event must be an object"
                    )
                yield line_number, event
    except UnicodeDecodeError as exc:
        raise UsageEvidenceError(f"{path}: trace is not UTF-8") from exc


def _parse_segment(event: dict, path: Path, line_number: int) -> _SegmentUsage:
    input_tokens = _optional_count(
        event.get("input_tokens"), "input_tokens", path, line_number
    )
    output_tokens = _optional_count(
        event.get("output_tokens"), "output_tokens", path, line_number
    )
    cached_tokens = _optional_count(
        event.get("cached_tokens"), "cached_tokens", path, line_number
    )
    if (
        input_tokens is not None
        and cached_tokens is not None
        and cached_tokens > input_tokens
    ):
        raise UsageEvidenceError(
            f"{path}:{line_number}: cached_tokens cannot exceed input_tokens"
        )
    return _SegmentUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_tokens=cached_tokens,
        cost=_parse_cost(event.get("cost"), path, line_number),
        quota=_parse_quota(event.get("quota"), path, line_number),
    )


def _segment_number(value: object, path: Path, line_number: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise UsageEvidenceError(
            f"{path}:{line_number}: session_number must be a positive integer"
        )
    return value


def _optional_count(
    value: object, field: str, path: Path, line_number: int
) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise UsageEvidenceError(
            f"{path}:{line_number}: {field} must be a non-negative integer or null"
        )
    return value


def _parse_cost(value: object, path: Path, line_number: int) -> CostEvidence | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise UsageEvidenceError(
            f"{path}:{line_number}: cost must be an object or null"
        )
    currency = _label(value.get("currency"), "cost currency", path, line_number)
    amount = _decimal(value.get("amount"), "cost amount", path, line_number)
    return CostEvidence(amount=amount, currency=currency)


def _parse_quota(value: object, path: Path, line_number: int) -> QuotaEvidence | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise UsageEvidenceError(
            f"{path}:{line_number}: quota must be an object or null"
        )
    remaining = _decimal(value.get("remaining"), "quota remaining", path, line_number)
    limit = _decimal(value.get("limit"), "quota limit", path, line_number)
    if remaining > limit:
        raise UsageEvidenceError(
            f"{path}:{line_number}: quota remaining cannot exceed quota limit"
        )
    return QuotaEvidence(
        remaining=remaining,
        limit=limit,
        unit=_label(value.get("unit"), "quota unit", path, line_number),
        scope=_label(value.get("scope"), "quota scope", path, line_number),
    )


def _decimal(value: object, field: str, path: Path, line_number: int) -> Decimal:
    if isinstance(value, bool) or isinstance(value, float) or not isinstance(
        value, (str, int)
    ):
        raise UsageEvidenceError(
            f"{path}:{line_number}: {field} must be an exact decimal string or integer"
        )
    try:
        number = Decimal(value)
    except (InvalidOperation, ValueError) as exc:
        raise UsageEvidenceError(f"{path}:{line_number}: {field} is invalid") from exc
    if not number.is_finite():
        raise UsageEvidenceError(f"{path}:{line_number}: {field} must be finite")
    if number < 0:
        raise UsageEvidenceError(f"{path}:{line_number}: {field} must be non-negative")
    return number


def _label(value: object, field: str, path: Path, line_number: int) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise UsageEvidenceError(
            f"{path}:{line_number}: {field} must be a non-empty exact string"
        )
    if not value.isprintable() or any(character in value for character in "\r\n"):
        raise UsageEvidenceError(f"{path}:{line_number}: {field} must be one line")
    return value


def _sum_known(segments: list[_SegmentUsage], field: str) -> int | None:
    if not segments:
        return None
    values = [getattr(segment, field) for segment in segments]
    if any(value is None for value in values):
        return None
    return sum(value for value in values if value is not None)


def _aggregate_cost(segments: list[_SegmentUsage]) -> CostEvidence | None:
    if not segments or any(segment.cost is None for segment in segments):
        return None
    costs = [segment.cost for segment in segments if segment.cost is not None]
    currencies = {cost.currency for cost in costs}
    if len(currencies) != 1:
        return None
    return CostEvidence(
        amount=_exact_decimal_sum([cost.amount for cost in costs]),
        currency=costs[0].currency,
    )


def _latest_compatible_quota(segments: list[_SegmentUsage]) -> QuotaEvidence | None:
    if not segments or any(segment.quota is None for segment in segments):
        return None
    quotas = [segment.quota for segment in segments if segment.quota is not None]
    semantics = {(quota.scope, quota.unit) for quota in quotas}
    return quotas[-1] if len(semantics) == 1 else None


def _known(value: int | None) -> str:
    return str(value) if value is not None else "unknown"


def _decimal_text(value: Decimal) -> str:
    return format(value, "f")


def _exact_decimal_sum(values: list[Decimal]) -> Decimal:
    """Add finite decimals without depending on the active decimal context."""
    exponent = min(value.as_tuple().exponent for value in values)
    total = 0
    for value in values:
        sign, digits, value_exponent = value.as_tuple()
        coefficient = int("".join(str(digit) for digit in digits) or "0")
        if sign:
            coefficient = -coefficient
        total += coefficient * 10 ** (value_exponent - exponent)
    sign = int(total < 0)
    digits = tuple(int(digit) for digit in str(abs(total))) or (0,)
    return Decimal((sign, digits, exponent))


def _percent_text(value: Fraction) -> str:
    """Format a ratio as a percentage with exact half-even rounding."""
    hundredths, remainder = divmod(value.numerator * 10_000, value.denominator)
    doubled = remainder * 2
    if doubled > value.denominator or (
        doubled == value.denominator and hundredths % 2
    ):
        hundredths += 1
    whole, fraction = divmod(hundredths, 100)
    return f"{whole}.{fraction:02d}%"


__all__ = [
    "CostEvidence",
    "QuotaEvidence",
    "SessionUsage",
    "UsageEvidenceError",
    "aggregate_session_usage",
    "render_session_usage",
]
