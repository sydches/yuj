"""Generic sealed-environment hints for bash command output."""
from __future__ import annotations

import re


_MISSING_MODULE_RE = re.compile(r"ModuleNotFoundError: No module named ['\"]([^'\"]+)['\"]")

_PYTHON_ENV_MISSING_HINT = (
    "\n[HARNESS: this Python interpreter cannot import a required module. "
    "In SWE-bench images, try the prebuilt testbed environment first, e.g. "
    "`source /opt/miniconda3/bin/activate && conda activate testbed && "
    "<your command>`. Network installs are usually unavailable in this "
    "sandbox.]"
)

_SEALED_INSTALL_FAILURE_HINT = (
    "\n[HARNESS: package installation failed in the sealed environment. "
    "Do not spend more turns trying network installs unless the package is "
    "already available locally. Prefer the prebuilt testbed environment: "
    "`source /opt/miniconda3/bin/activate && conda activate testbed && "
    "<your command>`.]"
)


def _python_env_missing(out: str, exit_code: int | None) -> bool:
    if exit_code == 0:
        return False
    return _MISSING_MODULE_RE.search(out) is not None


def _sealed_install_failure(cmd: str, out: str, exit_code: int | None) -> bool:
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
