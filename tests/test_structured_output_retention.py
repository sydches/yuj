"""Retain supplied diagnostics when structured projection cannot save them."""
from contextlib import contextmanager
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from _config_helpers import make_config
from scripts.llm_solver.bash_quirks import OutputControl, OutputParser
from scripts.llm_solver.harness._loop.state_projection import project_and_sink, sink_to_disk


@pytest.fixture
def projection(tmp_path, monkeypatch):
    parser = OutputParser(
        summary_fields={"failed": re.compile(r"(\d+) failed")},
        per_test_regex=re.compile(r"^(?P<verdict>FAILED) (?P<test_id>\S+)", re.MULTILINE),
    )
    control = OutputControl("", "PASSED", "FAILED", (re.compile(r"check"),), parser)
    ledger = SimpleNamespace(record_transform=Mock())
    monkeypatch.setattr("scripts.llm_solver.harness.savings.get_ledger", lambda: ledger)
    session = SimpleNamespace(
        cfg=make_config(bash_transforms_structured_output_enabled=True,
                        bash_transforms_sink_threshold_chars=0,
                        done_require_pretest_parity=False),
        cwd=tmp_path, _session_number=1, _sink_counter=0,
        output_parser=parser, output_control=control,
        _guards=SimpleNamespace(prev_test_parsed=None, mutation_count=0,
                                mutation_count_at_prev_test=0),
    )
    raw = "FAILED case_one " + "x" * 600 + " REQUIRED_DIAGNOSTIC\n1 failed\n"
    return session, raw, ledger


@pytest.mark.parametrize("failure", ["mkdir", "write"])
@pytest.mark.parametrize("sink_threshold", [0, 1])
def test_storage_failure_preserves_supplied_result(
    projection, monkeypatch, failure, sink_threshold,
):
    from dataclasses import replace
    session, raw, ledger = projection
    session.cfg = replace(session.cfg, bash_transforms_sink_threshold_chars=sink_threshold)
    sink_dir = session.cwd / ".tool_output"
    if failure == "mkdir":
        sink_dir.write_text("existing task file")
    else:
        open_path = Path.open

        def reject_write(data):
            raise OSError("fixture storage failure")

        @contextmanager
        def reject_sink(path, *args, **kwargs):
            with open_path(path, *args, **kwargs) as stream:
                yield SimpleNamespace(write=reject_write) if path.parent == sink_dir else stream

        monkeypatch.setattr(Path, "open", reject_sink)
    result = project_and_sink(session, "bash", "check", raw, 3)
    assert result == raw
    assert session._sink_counter == 1
    assert session._guards.prev_test_parsed == {"case_one": "FAILED"}
    ledger.record_transform.assert_not_called()
    if failure == "mkdir":
        assert sink_dir.read_text() == "existing task file"


def test_saved_result_remains_retrievable_with_digest(projection):
    session, raw, ledger = projection
    result = project_and_sink(session, "bash", "check", raw, 3)
    assert result != raw
    assert "[digest]" in result
    assert "REQUIRED_DIAGNOSTIC" in result
    match = re.search(r'full_path="([^"]+)"', result)
    assert match is not None
    assert (session.cwd / match[1]).read_text() == raw
    ledger.record_transform.assert_called_once()
    record = ledger.record_transform.call_args.kwargs
    assert record["before"] == raw
    assert record["after"] == result


def test_missing_pointer_preserves_result_even_after_successful_write(projection):
    from dataclasses import replace
    session, raw, ledger = projection
    session.cfg = replace(session.cfg, sink_pointer="")
    assert project_and_sink(session, "bash", "check", raw, 3) == raw
    ledger.record_transform.assert_not_called()


@pytest.mark.parametrize("occupied_kind", ["file", "symlink"])
def test_saved_pointer_survives_counter_restart_and_existing_paths(projection, occupied_kind):
    session, raw, _ = projection
    first_pointer = sink_to_disk(session, raw, 3)
    first_path = session.cwd / re.search(r'full_path="([^"]+)"', first_pointer)[1]
    if occupied_kind == "symlink":
        target = session.cwd / "existing.txt"
        target.write_text(raw)
        first_path.unlink()
        first_path.symlink_to(target)
    session._sink_counter = 0
    second_pointer = sink_to_disk(session, "NEW RESULT", 3)
    second_path = session.cwd / re.search(r'full_path="([^"]+)"', second_pointer)[1]
    assert second_path != first_path
    assert first_path.read_text() == raw
    assert second_path.read_text() == "NEW RESULT"
    if occupied_kind == "symlink":
        assert first_path.is_symlink()


@pytest.mark.parametrize("head,tail", [(0, 0), (4, 0), (0, 4)])
def test_zero_preview_side_selects_no_characters(projection, head, tail):
    from dataclasses import replace
    session, _, ledger = projection
    session.cfg = replace(
        session.cfg, bash_transforms_structured_output_enabled=False,
        bash_transforms_sink_threshold_chars=1, sink_head_bytes=head,
        sink_tail_bytes=tail,
    )
    raw = "HEAD" + " full diagnostic " * 100 + "TAIL"
    result = project_and_sink(session, "bash", "check", raw, 3)
    assert "full diagnostic" not in result
    assert ("HEAD" in result) == bool(head)
    assert ("TAIL" in result) == bool(tail)
    saved_path = re.search(r'full_path="([^"]+)"', result)[1]
    assert (session.cwd / saved_path).read_text() == raw
    ledger.record_transform.assert_called_once()
