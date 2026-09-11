"""pytest exit-code semantics and output detectors."""
import re

_PYTEST_COMMAND_NOT_FOUND_RE = re.compile(
    r"(?:^|:\s)(?:python(?:\d+(?:\.\d+)*)?|pytest):\s+"
    r"(?:command\s+)?not found\b",
    re.IGNORECASE | re.MULTILINE,
)


def _pytest_path_missing(out: str, exit_code: int | None) -> bool:
    """Match a pytest-shaped missing-path message with usage-error status.

    Copied output can match; this does not prove which path is absent.
    """
    if exit_code != 4:
        return False
    return ("ERROR: file or directory not found:" in out
            and "no tests ran" in out)


def _pytest_binary_missing(out: str, exit_code: int | None) -> bool:
    """Match runner lookup text from a known failed result.

    This pattern is not native evidence of executable or module availability.
    """
    if exit_code in (0, None):
        return False
    if "No module named pytest" in out:
        return True
    return (
        exit_code == 127
        and _PYTEST_COMMAND_NOT_FOUND_RE.search(out) is not None
    )
