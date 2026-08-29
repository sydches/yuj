"""pytest exit-code semantics + hint blocks shared by bash and run_tests."""
import re

# Pytest exit-code semantics. Source: docs.pytest.org/en/stable/reference/exit-codes.html
#   0 = all collected tests passed
#   1 = some tests failed
#   2 = test execution interrupted by user / collection error
#   3 = internal error happened while executing tests
#   4 = pytest invocation error (e.g. bad command line)
#   5 = no tests were collected
_PYTEST_STATUS = {
    0: "passed",
    1: "failed",
    2: "collection_error",
    3: "internal_error",
    4: "usage_error",
    5: "no_tests_collected",
}


_PYTEST_COMMAND_NOT_FOUND_RE = re.compile(
    r"(?:^|:\s)(?:python(?:\d+(?:\.\d+)*)?|pytest):\s+"
    r"(?:command\s+)?not found\b",
    re.IGNORECASE | re.MULTILINE,
)


_PYTEST_PATH_MISSING_HINT = (
    "\n[HARNESS: pytest reports the test path does not exist. "
    "Either fix the path, or write the test file before running tests again.]"
)

_PYTEST_BINARY_MISSING_HINT = (
    "\n[HARNESS: python -m pytest could not start. The interpreter on "
    "PATH may not be the one with pytest installed. Try the SWE-bench "
    "testbed activation path first, e.g. "
    "`source /opt/miniconda3/bin/activate && conda activate testbed && "
    "pytest ...`; if that hook is absent, try the task env directly, e.g. "
    "`/opt/miniconda3/envs/testbed/bin/python -m pytest ...`. Then "
    "switch back to run_tests once pytest is reachable.]"
)

_PYTEST_LF_CACHE_EMPTY_HINT = (
    "\n[HARNESS: last_failed=true requires a prior run to populate the "
    "lastfailed cache; pytest collected no tests because the cache is "
    "empty (no previously-failing tests on record). Run run_tests once "
    "without last_failed to populate the cache, then re-issue with "
    "last_failed=true. If you intended to verify against the full "
    "suite, drop last_failed.]"
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
