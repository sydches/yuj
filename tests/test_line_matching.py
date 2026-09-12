"""Patch matching preserves whole lines, scopes and overlapping windows."""
import random

from scripts.llm_solver.harness.line_matching import matching_line_starts


def test_line_windows_match_reference_for_empty_unicode_and_repeated_lines():
    rng = random.Random(20260912)
    alphabet = ['', 'a', 'aa', 'é', '\r', '\x00']
    cases = [([], ['']), (['a', 'a', 'a'], ['a', 'a']),
             (['prefix-a', 'a-suffix', 'a'], ['a']),
             (['', '', ''], ['', ''])]
    for _ in range(1000):
        lines = rng.choices(alphabet, k=rng.randrange(25))
        needle = rng.choices(alphabet, k=rng.randrange(6))
        cases.append((lines, needle))
    for lines, needle in cases:
        for scope in range(len(lines) + 1):
            expected = [i for i in range(scope, len(lines) - len(needle) + 1)
                        if lines[i:i + len(needle)] == needle]
            assert list(matching_line_starts(lines, needle, scope)) == expected
