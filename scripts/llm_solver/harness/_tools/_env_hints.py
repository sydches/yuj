"""Python environment failure detectors for bash command output."""
from __future__ import annotations

import re


_MISSING_MODULE_RE = re.compile(
    r"ModuleNotFoundError: No module named "
    r"['\"]([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)['\"]"
)


def _missing_python_module(out: str, exit_code: int | None) -> str | None:
    if exit_code == 0:
        return None
    match = _MISSING_MODULE_RE.search(out)
    return match.group(1) if match is not None else None


def _python_install_failure(cmd: str, out: str, exit_code: int | None) -> bool:
    if exit_code == 0:
        return False
    lowered_cmd = cmd.lower()
    if not (
        "pip install" in lowered_cmd
        or "conda install" in lowered_cmd
        or "python -m pip" in lowered_cmd
    ):
        return False
    lowered_out = out.lower()
    return (
        "temporary failure in name resolution" in lowered_out
        or "no matching distribution found" in lowered_out
        or "could not find a version that satisfies the requirement" in lowered_out
        or "failed to establish a new connection" in lowered_out
    )
