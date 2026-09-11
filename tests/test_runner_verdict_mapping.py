"""Runner declarations, not a global vocabulary, interpret verdict words."""
from pathlib import Path

import pytest

from scripts.llm_solver.bash_quirks import load_output_parser, parse_structured


@pytest.mark.parametrize("status", ["PASSED", "FAILED"])
def test_same_word_follows_the_selected_descriptor(tmp_path, status):
    descriptor = tmp_path / "runner.toml"
    descriptor.write_text(
        "[output_parser.per_test]\n"
        "regex = '^(?P<verdict>ready)\\s+(?P<test_id>\\S+)$'\n"
        f"verdict_map = {{ ready = '{status}' }}\n"
    )
    parsed = parse_structured("ready sample", load_output_parser(descriptor))
    assert parsed["tests"] == {"sample": status}


@pytest.mark.parametrize("runner,text,expected", [
    ("pytest", "PASSED sample\nFAILED other", {"sample": "PASSED", "other": "FAILED"}),
    ("go", "--- PASS: sample\n--- FAIL: other", {"sample": "PASSED", "other": "FAILED"}),
    ("cargo", "test sample ... ok\ntest other ... ignored",
     {"sample": "PASSED", "other": "SKIPPED"}),
    ("jest", "✓ sample\n✕ other", {"sample": "PASSED", "other": "FAILED"}),
])
def test_shipped_descriptors_preserve_canonical_results(runner, text, expected):
    root = Path(__file__).resolve().parents[1]
    descriptor = root / "scripts/llm_solver/language_quirks" / f"{runner}.toml"
    assert parse_structured(text, load_output_parser(descriptor))["tests"] == expected
