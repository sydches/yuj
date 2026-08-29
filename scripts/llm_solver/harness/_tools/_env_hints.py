"""Python environment failure detectors for bash command output."""
from __future__ import annotations

import re


_MISSING_MODULE_RE = re.compile(r"ModuleNotFoundError: No module named ['\"]([^'\"]+)['\"]")

def _python_env_missing(out: str, exit_code: int | None) -> bool:
    if exit_code == 0:
        return False
    return _MISSING_MODULE_RE.search(out) is not None


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
