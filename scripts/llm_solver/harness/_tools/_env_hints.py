"""Python request shapes and output hints, not environment diagnoses."""
from __future__ import annotations

import re
import shlex


def _python_request_kind(command: str) -> str:
    """Describe one literal requested interface, not the executable that ran.

    Wrappers and compound shell commands stay unresolved. A matching binary
    spelling cannot establish its identity or the cause of its output.
    """
    from ..command_redirect import strip_leading_assignments
    from ..shell_verification import _literal_simple_command

    if not _literal_simple_command(command):
        return ""
    try:
        argv = shlex.split(strip_leading_assignments(command))
    except ValueError:
        return ""
    if not argv:
        return ""
    binary = argv[0].rsplit("/", 1)[-1]
    args = argv[1:]
    if re.fullmatch(r"python(?:\d+(?:\.\d+)*)?", binary):
        while args and args[0] in {"-u", "-B", "-I", "-E", "-s", "-S", "-O", "-OO"}:
            args = args[1:]
        if args[:3] == ["-m", "pip", "install"]:
            return "install"
        return "pytest" if args[:2] == ["-m", "pytest"] else "python"
    if binary == "pytest":
        return "pytest"
    if (re.fullmatch(r"pip(?:\d+(?:\.\d+)*)?", binary) or binary == "conda") and args[:1] == ["install"]:
        return "install"
    return ""


_MISSING_MODULE_RE = re.compile(
    r"ModuleNotFoundError: No module named "
    r"['\"]([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)['\"]"
)


def _missing_python_module(out: str, exit_code: int | None) -> str | None:
    if exit_code in (0, None):
        return None
    match = _MISSING_MODULE_RE.search(out)
    return match.group(1) if match is not None else None


def _python_install_failure(cmd: str, out: str, exit_code: int | None) -> bool:
    if exit_code in (0, None):
        return False
    if _python_request_kind(cmd) != "install":
        return False
    lowered_out = out.lower()
    return (
        "temporary failure in name resolution" in lowered_out
        or "no matching distribution found" in lowered_out
        or "could not find a version that satisfies the requirement" in lowered_out
        or "failed to establish a new connection" in lowered_out
    )
