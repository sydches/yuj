"""Summary parsing must preserve the supplied output's token and line boundaries."""
import re

import pytest

from scripts.llm_solver.bash_quirks._output import OutputParser, parse_structured


@pytest.mark.parametrize("pattern,output,expected", [
    (r"(\d+) passed", "12345 passed" + "x" * 3990, {"passed": 12345}),
    (r"^(\d+) passed", "prefix 12345 passed" + "x" * 3990, None),
    (r"^(\d+) passed", "12 passed\n" + "x" * 5000 + "\n3 passed", {"passed": 3}),
    (r"^(\d+) passed", "12 passed\n" + "x" * 5000, {"passed": 12}),
], ids=["split-count", "false-line-start", "last-summary", "early-summary"])
def test_summary_preserves_original_boundaries(pattern, output, expected):
    parser = OutputParser(
        summary_fields={"passed": re.compile(pattern, re.MULTILINE)},
        per_test_regex=None,
    )
    assert parse_structured(output, parser)["summary"] == expected


def test_failure_line_preserves_full_identity_verdict_and_diagnostic():
    test_id = "case_" + "x" * 600
    parser = OutputParser(
        summary_fields={},
        per_test_regex=re.compile(r"^(?P<verdict>FAILED) (?P<test_id>\S+)", re.MULTILINE),
    )
    parsed = parse_structured(f"FAILED {test_id} diagnostic detail", parser)
    assert parsed["tests"] == {test_id: "FAILED"}
    detail, = parsed["failure_details"]
    assert detail == f"FAILED {test_id} diagnostic detail"
