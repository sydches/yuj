"""pytest exit-code semantics and output detectors."""
import re

_PYTEST_COMMAND_NOT_FOUND_RE = re.compile(
    r"(?:^|:\s)(?:python(?:\d+(?:\.\d+)*)?|pytest):\s+"
    r"(?:command\s+)?not found\b",
    re.IGNORECASE | re.MULTILINE,
)


def _pytest_path_missing(out: str, exit_code: int | None) -> bool:
    """True iff pytest just refused because the target path is absent.

    pytest exit code 4 is "usage error". The combination of code 4 plus
    the verbatim ``ERROR: file or directory not found:`` and ``no tests
    ran`` strings is pytest's own signal that the test path doesn't
    exist. Triggering off pytest's own output is leakage-clean — no
    F2P-set knowledge required.
    """
    if exit_code != 4:
        return False
    return ("ERROR: file or directory not found:" in out
            and "no tests ran" in out)


def _pytest_binary_missing(out: str, exit_code: int | None) -> bool:
    """True iff `python -m pytest` couldn't even start.

    Distinct from path-missing: this fires when the shell or python
    interpreter rejects the invocation itself. Two shapes:
      * exit 127 naming missing ``python`` or ``pytest`` → no test
        runner on PATH
      * any exit + ``No module named pytest`` → wrong python (no pytest
        installed in that interpreter)
    Both indicate the canonical fix is "use the task's own python".
    """
    if "No module named pytest" in out:
        return True
    return (
        exit_code == 127
        and _PYTEST_COMMAND_NOT_FOUND_RE.search(out) is not None
    )
