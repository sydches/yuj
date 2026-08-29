"""Language quirks — per-runner/per-language semantics in declarative TOML.

Each ``<runner>.toml`` describes one test runner (pytest, cargo, jest, go,
ctest, …): invocation patterns, verdict markers, output-control flags.
Consumed by bash quirks (for output condensation) and by analysis
detectors (for format-conditional tagging).

Adding a new language/runner = adding a TOML file here. No code change in
harness, analysis, or any other layer.
"""
from __future__ import annotations

import functools
import logging
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .._shared.paths import package_data_path

log = logging.getLogger(__name__)

FORMATS_DIR = package_data_path(__package__)


@dataclass(frozen=True)
class RunnerDescriptor:
    """First-class metadata for one run_tests-capable runner descriptor."""

    name: str
    path: Path
    detection_priority: int
    detect_files: tuple[str, ...]


@dataclass(frozen=True)
class RunTestsQuirk:
    """First-class command metadata for the run_tests tool."""

    runner: str
    env_activate_prefix: str = ""
    base_cmd: str = ""
    detect_files: tuple[str, ...] = ()
    arg_path_style: str = "ignored"
    arg_k_template: str = ""
    arg_last_failed: str = ""
    # Multilingual exit-code -> status mapping. Empty by default; when empty,
    # run_tests.py
    # falls back to the hardcoded pytest _PYTEST_STATUS map so pytest's
    # behavior is unaffected whether or not pytest.toml carries an explicit
    # table. Non-pytest runners (cargo/go/jest/ctest) declare their own
    # table via [run_tests.status_map] in their TOML so a runner-specific
    # exit code (e.g. cargo's 101) doesn't get labelled with pytest
    # vocabulary (e.g. "usage_error").
    status_map: dict[int, str] = field(default_factory=dict)
    # Status name for exit codes not present in status_map, used only
    # when status_map is non-empty (i.e. only for runners that opted in).
    # Empty string means "fall back to error_<code>" (matches the pytest
    # legacy behavior for unmapped codes).
    status_default: str = ""
    advice: dict[str, str] = field(default_factory=dict)
    extra_fields: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    def to_legacy_dict(self) -> dict:
        """Return the historical dict shape expected by older callers."""
        legacy = dict(self.extra_fields)
        legacy.update(
            {
                "env_activate_prefix": self.env_activate_prefix,
                "base_cmd": self.base_cmd,
                "detect_files": list(self.detect_files),
                "arg_path_style": self.arg_path_style,
                "arg_k_template": self.arg_k_template,
                "arg_last_failed": self.arg_last_failed,
                "status_map": dict(self.status_map),
                "status_default": self.status_default,
                "advice": dict(self.advice),
                "_runner": self.runner,
            }
        )
        return legacy


# Cache the per-runner TOML parse. Keys are the resolved runner name, so cardinality is
# bounded and the cache never grows.
@functools.lru_cache(maxsize=8)
def _load_runner_quirk_dict(runner: str) -> dict:
    toml_path = FORMATS_DIR / f"{runner}.toml"
    with toml_path.open("rb") as f:
        return tomllib.load(f)


def _run_tests_path(runner: str) -> Path:
    return FORMATS_DIR / f"{runner}.toml"


def _detect_files_from_dict(runner: str, run_tests: dict) -> tuple[str, ...]:
    detect_files = run_tests.get("detect_files")
    if not isinstance(detect_files, list) or not all(
        isinstance(marker, str) for marker in detect_files
    ):
        raise ValueError(
            f"{_run_tests_path(runner)} [run_tests].detect_files "
            "must be a list of strings"
        )
    return tuple(detect_files)


def _descriptor_from_dict(runner: str, cfg: dict) -> RunnerDescriptor | None:
    """Build a detection descriptor for TOML files with [run_tests]."""
    run_tests = cfg.get("run_tests")
    if not isinstance(run_tests, dict):
        return None
    name = str(cfg.get("name") or runner)
    if name != runner:
        raise ValueError(
            f"{_run_tests_path(runner)} has name={name!r}; expected {runner!r}"
        )
    priority = run_tests.get("detection_priority")
    if not isinstance(priority, int):
        raise ValueError(
            f"{_run_tests_path(runner)} [run_tests] missing integer "
            "detection_priority"
        )
    return RunnerDescriptor(
        name=name,
        path=_run_tests_path(runner),
        detection_priority=priority,
        detect_files=_detect_files_from_dict(runner, run_tests),
    )


def _run_tests_quirk_from_dict(runner: str, run_tests: dict) -> RunTestsQuirk:
    """Build command metadata from one TOML [run_tests] table."""
    string_defaults = {
        "env_activate_prefix": "",
        "base_cmd": "",
        "arg_path_style": "ignored",
        "arg_k_template": "",
        "arg_last_failed": "",
    }
    values: dict[str, str] = {}
    for key, default in string_defaults.items():
        value = run_tests.get(key, default)
        if not isinstance(value, str):
            raise ValueError(
                f"{_run_tests_path(runner)} [run_tests].{key} must be a string"
            )
        values[key] = value

    status_map = _status_map_from_dict(runner, run_tests)
    status_default = run_tests.get("status_default", "")
    if not isinstance(status_default, str):
        raise ValueError(
            f"{_run_tests_path(runner)} [run_tests].status_default must be a string"
        )
    advice = _advice_from_dict(runner, run_tests)

    known = set(string_defaults) | {
        "advice", "detect_files", "detection_priority", "status_map",
        "status_default",
    }
    extra_fields = {key: value for key, value in run_tests.items() if key not in known}

    return RunTestsQuirk(
        runner=runner,
        env_activate_prefix=values["env_activate_prefix"],
        base_cmd=values["base_cmd"],
        detect_files=_detect_files_from_dict(runner, run_tests),
        arg_path_style=values["arg_path_style"],
        arg_k_template=values["arg_k_template"],
        arg_last_failed=values["arg_last_failed"],
        status_map=status_map,
        status_default=status_default,
        advice=advice,
        extra_fields=extra_fields,
    )


def _advice_from_dict(runner: str, run_tests: dict) -> dict[str, str]:
    """Parse optional model-visible recovery advice for one runner."""
    raw = run_tests.get("advice", {})
    if not isinstance(raw, dict):
        raise ValueError(
            f"{_run_tests_path(runner)} [run_tests].advice must be a table"
        )
    advice: dict[str, str] = {}
    for key, value in raw.items():
        if not isinstance(value, str):
            raise ValueError(
                f"{_run_tests_path(runner)} [run_tests].advice.{key} "
                "must be a string"
            )
        advice[str(key)] = value
    return advice


def _status_map_from_dict(runner: str, run_tests: dict) -> dict[int, str]:
    """Parse the optional ``[run_tests.status_map]`` table.

    TOML keys are always strings (bare-integer keys like ``101 = "failed"``
    parse as the string ``"101"``), so this converts each key to an int
    exit code and validates the value is a status-name string. Absent
    entirely -> empty dict, which signals run_tests.py to fall back to
    the hardcoded pytest ``_PYTEST_STATUS`` map (byte-identical legacy
    behavior for runners that don't declare their own table).
    """
    raw = run_tests.get("status_map", {})
    if not isinstance(raw, dict):
        raise ValueError(
            f"{_run_tests_path(runner)} [run_tests].status_map must be a table"
        )
    status_map: dict[int, str] = {}
    for key, value in raw.items():
        try:
            code = int(key)
        except (TypeError, ValueError):
            raise ValueError(
                f"{_run_tests_path(runner)} [run_tests].status_map key "
                f"{key!r} must be an integer exit code"
            ) from None
        if not isinstance(value, str):
            raise ValueError(
                f"{_run_tests_path(runner)} [run_tests].status_map[{key!r}] "
                "must be a string"
            )
        status_map[code] = value
    return status_map


@functools.lru_cache(maxsize=1)
def list_run_test_runner_descriptors() -> tuple[RunnerDescriptor, ...]:
    """Return run_tests runner descriptors in detection order."""
    descriptors: list[RunnerDescriptor] = []
    for toml_path in sorted(FORMATS_DIR.glob("*.toml")):
        descriptor = _descriptor_from_dict(
            toml_path.stem,
            _load_runner_quirk_dict(toml_path.stem),
        )
        if descriptor is not None:
            descriptors.append(descriptor)

    seen_priorities: dict[int, str] = {}
    for descriptor in descriptors:
        existing = seen_priorities.get(descriptor.detection_priority)
        if existing is not None:
            raise ValueError(
                "Duplicate language runner detection_priority "
                f"{descriptor.detection_priority}: {existing}, {descriptor.name}"
            )
        seen_priorities[descriptor.detection_priority] = descriptor.name

    return tuple(
        sorted(descriptors, key=lambda item: (item.detection_priority, item.name))
    )


_DETECTION_ORDER = tuple(
    descriptor.name for descriptor in list_run_test_runner_descriptors()
)


@functools.lru_cache(maxsize=1)
def all_verification_patterns() -> tuple[str, ...]:
    """Union of every runner's ``verification_patterns``.

    The single source of truth for "is this bash call a test/verification
    command?" across all languages. Any shared regex that previously
    hard-coded one language's runners (e.g. ``_shell_patterns.TEST_COMMAND_RE``)
    derives from this so it can never drift from the per-runner TOMLs.
    """
    pats: list[str] = []
    for toml_path in sorted(FORMATS_DIR.glob("*.toml")):
        d = _load_runner_quirk_dict(toml_path.stem)
        for p in d.get("verification_patterns", []) or []:
            if p not in pats:
                pats.append(p)
    return tuple(pats)


def detect_runner(cwd: str | Path) -> str:
    """Return the runner name (e.g. 'pytest', 'cargo') for the given cwd.

    Inspects ``cwd`` for each descriptor's ``[run_tests].detect_files``
    in ``detection_priority`` order. First match wins. Falls back to
    ``"pytest"`` when nothing matches because it is the most common Python
    test runner.
    """
    cwd_path = Path(cwd)
    for descriptor in list_run_test_runner_descriptors():
        for marker in descriptor.detect_files:
            if (cwd_path / marker).exists():
                return descriptor.name
    # Log the fallback so the operator can see when run_tests is
    # dispatched to pytest because
    # no language marker matched (vs because pytest was the explicit
    # detection winner).
    log.info("detect_runner: no language marker found in cwd %s; falling back to pytest", cwd)
    return "pytest"


def load_run_tests_quirk_object(cwd: str | Path) -> RunTestsQuirk:
    """Return command metadata for the run_tests runner detected for ``cwd``."""
    return load_run_tests_quirk_for_runner(detect_runner(cwd))


def load_run_tests_quirk_for_runner(runner: str) -> RunTestsQuirk:
    """Return command metadata for one named runner descriptor."""
    cfg = _load_runner_quirk_dict(runner)
    run_tests = cfg.get("run_tests", {})
    if not isinstance(run_tests, dict):
        raise ValueError(f"{_run_tests_path(runner)} missing [run_tests] table")
    return _run_tests_quirk_from_dict(runner, run_tests)


def load_run_tests_quirk(cwd: str | Path) -> dict:
    """Return the legacy [run_tests] dict of the runner detected for ``cwd``.

    The returned dict carries an extra ``_runner`` key naming the
    matched runner (useful for callers that want to log which runner
    they used). Missing optional keys default to empty/sensible values
    rather than raising, so a partially-populated TOML is still usable.
    """
    return load_run_tests_quirk_object(cwd).to_legacy_dict()


__all__ = [
    "FORMATS_DIR",
    "RunnerDescriptor",
    "RunTestsQuirk",
    "detect_runner",
    "list_run_test_runner_descriptors",
    "load_run_tests_quirk",
    "load_run_tests_quirk_for_runner",
    "load_run_tests_quirk_object",
]
