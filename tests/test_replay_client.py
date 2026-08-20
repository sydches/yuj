"""ReplayClient: recorded turns served through the live TurnResult contract.

Spec: docs/replay_mode_spec.md (turn alignment, divergence stop, stop-turn
boundary, prefix-only by construction).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from llm_solver.server.replay_client import (  # noqa: E402
    REPLAY_FINISH_REASON_EXHAUSTED,
    REPLAY_FINISH_REASON_STOP_TURN,
    VOLATILE_NORMALIZATION_VERSION,
    ReplayClient,
    ReplayDivergence,
)


def _resp(content=None, tool=None, finish="tool_calls"):
    msg = {"role": "assistant", "content": content}
    if tool:
        msg["tool_calls"] = [{"id": "t1", "type": "function",
                              "function": {"name": "bash",
                                           "arguments": json.dumps({"cmd": tool})}}]
    return {"choices": [{"message": msg, "finish_reason": finish}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}}


def _transcript(tmp_path, turns):
    """turns: list of (request_messages, response_dict)."""
    parts = []
    for i, (req_msgs, resp) in enumerate(turns, start=1):
        parts.append(f"=== turn {i:03d} input ===")
        parts.append(json.dumps({"messages": req_msgs}))
        parts.append(f"=== turn {i:03d} output ===")
        parts.append(json.dumps(resp))
    p = tmp_path / "host_task.log"
    p.write_text("\n".join(parts))
    return p


SYS = [{"role": "system", "content": "s"}, {"role": "user", "content": "task"}]


def test_serves_recorded_turns_in_order(tmp_path):
    p = _transcript(tmp_path, [
        (SYS, _resp(tool="ls")),
        (SYS + [{"role": "assistant", "content": None},
                {"role": "tool", "content": "file1"}], _resp(tool="cat file1")),
    ])
    c = ReplayClient(p)
    r1 = c.chat(SYS, [], turn=0)
    assert r1.tool_calls[0].arguments == {"cmd": "ls"}
    live = SYS + [{"role": "assistant", "content": None},
                  {"role": "tool", "content": "file1"}]
    r2 = c.chat(live, [], turn=1)
    assert r2.tool_calls[0].arguments == {"cmd": "cat file1"}


def test_divergence_stops_trace_level(tmp_path):
    # trace-level gate (spec): executed event compared to recorded event
    p = _transcript(tmp_path, [(SYS, _resp(tool="ls"))])
    trace = tmp_path / ".trace.jsonl"
    trace.write_text(json.dumps({
        "event": "tool_call", "turn_number": 1, "tool_name": "bash",
        "args_summary": "cmd='ls'", "result_summary": "file1"}) + "\n")
    c = ReplayClient(p, source_trace_path=trace)
    c.chat(SYS, [], turn=0)
    # identical event: no divergence
    c.verify_executed_turn({"event": "tool_call", "turn_number": 1,
                            "tool_name": "bash", "args_summary": "cmd='ls'",
                            "result_summary": "file1"})
    assert c.divergence is None
    with pytest.raises(ReplayDivergence):
        c.verify_executed_turn({"event": "tool_call", "turn_number": 1,
                                "tool_name": "bash", "args_summary": "cmd='ls'",
                                "result_summary": "DIFFERENT"})
    assert c.divergence and c.divergence["field"] == "result_summary"


def test_trace_level_fidelity_prefers_output_hash(tmp_path):
    p = _transcript(tmp_path, [(SYS, _resp(tool="cat big"))])
    trace = tmp_path / ".trace.jsonl"
    trace.write_text(json.dumps({
        "event": "tool_call", "turn_number": 1, "tool_name": "bash",
        "args_summary": "cmd='cat big'", "result_summary": "snippet A",
        "output_sha256": "abc"}) + "\n")
    c = ReplayClient(p, source_trace_path=trace)
    c.verify_executed_turn({
        "event": "tool_call", "turn_number": 1, "tool_name": "bash",
        "args_summary": "cmd='cat big'", "result_summary": "snippet B",
        "output_sha256": "abc",
    })
    assert c.divergence is None

    with pytest.raises(ReplayDivergence):
        c.verify_executed_turn({
            "event": "tool_call", "turn_number": 1, "tool_name": "bash",
            "args_summary": "cmd='cat big'", "result_summary": "snippet A",
            "output_sha256": "def",
        })
    assert c.divergence and c.divergence["field"] == "output_sha256"


def test_stop_turn_boundary_trace_numbering(tmp_path):
    # stop_turn is TRACE numbering (0-based): stop_turn=1 serves trace turns
    # 0 and 1 (transcript turns 1 and 2), then the sentinel
    p = _transcript(tmp_path, [(SYS, _resp(tool="ls")),
                               (SYS, _resp(tool="pwd")),
                               (SYS, _resp(tool="id"))])
    c = ReplayClient(p, stop_turn=1, strict_fidelity=False)
    assert c.chat(SYS, [], 0).tool_calls[0].arguments == {"cmd": "ls"}
    assert c.chat(SYS, [], 1).tool_calls[0].arguments == {"cmd": "pwd"}
    assert c.chat(SYS, [], 2).finish_reason == REPLAY_FINISH_REASON_STOP_TURN


def test_exhausted_recording(tmp_path):
    p = _transcript(tmp_path, [(SYS, _resp(tool="ls"))])
    c = ReplayClient(p, strict_fidelity=False)
    c.chat(SYS, [], 0)
    assert c.chat(SYS, [], 1).finish_reason == REPLAY_FINISH_REASON_EXHAUSTED


def test_build_assistant_message_shape(tmp_path):
    p = _transcript(tmp_path, [(SYS, _resp(tool="ls"))])
    c = ReplayClient(p)
    r = c.chat(SYS, [], 0)
    m = c.build_assistant_message(r.content, r.tool_calls)
    assert m["role"] == "assistant"
    assert json.loads(m["tool_calls"][0]["function"]["arguments"]) == {"cmd": "ls"}


def test_resolve_replay_source_reads_mode(tmp_path):
    from llm_solver.server.replay_client import resolve_replay_source
    run = tmp_path / "cell"
    (run / "harness_run" / "transcripts").mkdir(parents=True)
    (run / "host_task").mkdir()
    (run / "harness_run" / "transcripts" / "host_task.log").write_text(
        "=== turn 001 input ===\n{}\n=== turn 001 output ===\n{}")
    (run / "host_task" / ".trace.jsonl").write_text(json.dumps(
        {"event": "session_start", "session_number": 1,
         "context_contract": {"mode": "compound"}}) + "\n")
    transcript, trace, mode = resolve_replay_source(run)
    assert transcript.name == "host_task.log"
    assert trace is not None and trace.name == ".trace.jsonl"
    assert mode == "compound"


def test_load_replay_provenance_verifies_hashes(tmp_path):
    import hashlib
    from llm_solver.server.replay_client import load_replay_provenance
    run = tmp_path / "cell"
    (run / "harness_run").mkdir(parents=True)
    overlay = tmp_path / "row.toml"
    overlay.write_text("[loop]\nmax_turns = 5\n")
    sha = hashlib.sha256(overlay.read_bytes()).hexdigest()
    (run / "harness_run" / "session.json").write_text(json.dumps({
        "model": "qwen-test", "context_mode": "compound",
        "config_paths": [str(overlay)],
        "config_path_hashes": {str(overlay): sha}}))
    prov = load_replay_provenance(run)
    assert prov["model"] == "qwen-test"
    assert prov["config_paths"] == [str(overlay)]
    # drift the file -> refusal
    overlay.write_text("[loop]\nmax_turns = 6\n")
    try:
        load_replay_provenance(run)
        raise AssertionError("drift not detected")
    except ValueError as e:
        assert "drifted" in str(e)


def test_volatile_normalization_stat_device_and_timestamps(tmp_path):
    """Ignore overlay device IDs and stat timestamps during replay."""
    from llm_solver.server.replay_client import ReplayClient
    p = _transcript(tmp_path, [(SYS, _resp(tool="stat core.py"))])
    trace = tmp_path / ".trace.jsonl"
    rec = ("  File: core.py\n  Size: 3280  Blocks: 8\n"
           "Device: 4ah/74d  Inode: 919904  Links: 1\n"
           "Access: 2026-01-01 00:00:00.000000000 +0000\n"
           "Change: 2026-01-01 00:00:00.123456789 +0000\n")
    trace.write_text(json.dumps({
        "event": "tool_call", "turn_number": 1, "tool_name": "bash",
        "args_summary": "cmd='stat core.py'", "result_summary": rec}) + "\n")
    c = ReplayClient(p, source_trace_path=trace)
    live = rec.replace("4ah/74d", "38h/56d").replace(
        "Change: 2026-01-01 00:00:00.123456789 +0000",
        "Change: 2026-06-10 02:33:11.987654321 +0100")
    c.verify_executed_turn({"event": "tool_call", "turn_number": 1,
                            "tool_name": "bash",
                            "args_summary": "cmd='stat core.py'",
                            "result_summary": live})
    assert c.divergence is None  # faithful under named normalization
    # a REAL content diff still trips (size changed)
    import pytest as _pytest
    with _pytest.raises(Exception):
        c.verify_executed_turn({"event": "tool_call", "turn_number": 1,
                                "tool_name": "bash",
                                "args_summary": "cmd='stat core.py'",
                                "result_summary": live.replace("Size: 3280", "Size: 9999")})


def _verify_pair(tmp_path, recorded, live, args="cmd='x'"):
    """Helper: gate verdict for one recorded-vs-live result pair."""
    from llm_solver.server.replay_client import ReplayClient
    tmp_path.mkdir(parents=True, exist_ok=True)
    p = _transcript(tmp_path, [(SYS, _resp(tool="x"))])
    trace = tmp_path / ".trace.jsonl"
    trace.write_text(json.dumps({
        "event": "tool_call", "turn_number": 1, "tool_name": "bash",
        "args_summary": args, "result_summary": recorded}) + "\n")
    c = ReplayClient(p, source_trace_path=trace)
    c.verify_executed_turn({"event": "tool_call", "turn_number": 1,
                            "tool_name": "bash", "args_summary": args,
                            "result_summary": live})
    return c.divergence


def test_volatile_sed_temp_filename(tmp_path):
    """Normalize random temporary names from sed failures."""
    rec = "sed: couldn't open temporary file /testbed/requests/sed6RLVFx: Permission denied\n\n[exit code: 4]"
    live = rec.replace("sed6RLVFx", "sedB8wJeB")
    assert _verify_pair(tmp_path, rec, live) is None


def test_volatile_sed_scoped_to_error_message(tmp_path):
    """Narrowness: sedXXXXXX outside the temp-file error still compares."""
    import pytest as _pytest
    rec = "created helper sedAAAAAA.py\n"
    live = "created helper sedBBBBBB.py\n"
    with _pytest.raises(Exception):
        _verify_pair(tmp_path / "n", rec, live)


def test_volatile_diff_header_mtime(tmp_path):
    """Normalize timestamps in diff headers."""
    rec = ("--- /testbed/astropy/timeseries/core.py\t2026-05-16 02:00:00.000000000 +0000\n"
           "+++ /tmp/core_new.py\t2026-06-07 12:45:33.123456789 +0000\n"
           "@@ -60,7 +60,7 @@\n-old\n+new\n")
    live = rec.replace("2026-06-07 12:45:33.123456789", "2026-06-10 02:29:01.987654321")
    assert _verify_pair(tmp_path, rec, live) is None


def test_volatile_diff_body_still_trips(tmp_path):
    """Narrowness: the diff BODY is content — a hunk change still halts."""
    import pytest as _pytest
    rec = ("+++ /tmp/core_new.py\t2026-06-07 12:45:33 +0000\n@@ -1 +1 @@\n-old\n+new\n")
    live = rec.replace("+new", "+different")
    with _pytest.raises(Exception):
        _verify_pair(tmp_path / "n2", rec, live)


def test_volatile_python_tempfile_name(tmp_path):
    """Normalize random names from Python temporary files."""
    rec = "wrote /tmp/tmpgy77bvc_ then applied patch\n"
    live = rec.replace("tmpgy77bvc_", "tmpkfonobk3")
    assert _verify_pair(tmp_path, rec, live) is None


def test_volatile_tempfile_scoped_to_eight_random_chars(tmp_path):
    """Narrowness: tmp-prefixed names that are NOT the 8-char tempfile shape
    still compare (a genuine divergence in them halts)."""
    import pytest as _pytest
    rec = "using tmpdir_path for scratch\n"
    live = "using tmpdir_other for scratch\n"
    with _pytest.raises(Exception):
        _verify_pair(tmp_path / "n3", rec, live)


def test_volatile_v4_ls_total_blocks(tmp_path):
    """Normalize the block total from ls -la output."""
    rec = "total 88\ndrwxr-xr-x 2 root root 4096 file\n"
    assert _verify_pair(tmp_path, rec, rec.replace("total 88", "total 96")) is None


def test_volatile_v4_sigpipe_141_only(tmp_path):
    """Normalize the exit 141 and 0 race, but keep exit 2 distinct."""
    import pytest as _pytest
    rec = "some output\n\n[exit code: 141]"
    live = "some output"
    assert _verify_pair(tmp_path, rec, live) is None
    with _pytest.raises(Exception):
        _verify_pair(tmp_path / "n4", "x\n\n[exit code: 2]", "x\n\n[exit code: 1]")


def test_volatile_v4_network_noise(tmp_path):
    """Normalize retry and timing text when the network is off."""
    rec = ("Collecting pytest\n  Retrying (Retry(total=4, connect=None)) after connection broken\n"
           "Max retries exceeded with url: /simple/pytest/\n\n[exit code: 1]")
    live = rec.replace("total=4", "total=2").replace("/simple/pytest/", "/simple/pytest/ x")
    assert _verify_pair(tmp_path, rec, live) is None


def test_volatile_v4_pytest_duration(tmp_path):
    rec = "===== 12 passed in 3.42s =====\n"
    assert _verify_pair(tmp_path, rec, rec.replace("3.42", "5.07")) is None


def test_volatile_v4_set_repr_order(tmp_path):
    """Normalize set order, but keep set values distinct."""
    import pytest as _pytest
    rec = "_coord_names: {'b', 'z', 'a'}\n"
    live = "_coord_names: {'a', 'b', 'z'}\n"
    assert _verify_pair(tmp_path, rec, live) is None
    with _pytest.raises(Exception):
        _verify_pair(tmp_path / "n5", rec, "_coord_names: {'a', 'b', 'q'}\n")


def test_volatile_v4_ordering_only_lines(tmp_path):
    """Ignore mount-table order but still reject a changed line."""
    import pytest as _pytest
    rec = "overlay on / type overlay\nproc on /proc type proc\n"
    live = "proc on /proc type proc\noverlay on / type overlay\n"
    assert _verify_pair(tmp_path, rec, live) is None
    with _pytest.raises(Exception):
        _verify_pair(tmp_path / "n6", rec,
                     "proc on /proc type proc\noverlay on /x type overlay\n")


def test_volatile_v4_turn_minus_one_guard(tmp_path):
    """Do not compare an event that has no turn_number."""
    from llm_solver.server.replay_client import ReplayClient
    p = _transcript(tmp_path, [(SYS, _resp(tool="x"))])
    trace = tmp_path / ".trace.jsonl"
    trace.write_text(json.dumps({
        "event": "tool_call", "turn_number": 1, "tool_name": "bash",
        "args_summary": "a", "result_summary": "b"}) + "\n")
    c = ReplayClient(p, source_trace_path=trace)
    c.verify_executed_turn({"event": "tool_call", "tool_name": "bash",
                            "args_summary": "ZZZ", "result_summary": "ZZZ"})
    assert c.divergence is None


def test_volatile_v5_snapshot_path(tmp_path):
    """Normalize container snapshot IDs in overlay lowerdir paths."""
    rec = "overlay on / type overlay (rw,lowerdir=/mnt/x/snapshots/66483/fs:/mnt/x/snapshots/32326/fs)\n"
    live = rec.replace("66483", "71022").replace("32326", "31999")
    assert _verify_pair(tmp_path, rec, live) is None


def test_volatile_v5_chown_wall_collapses(tmp_path):
    """Normalize chown errors that vary with traversal and truncation."""
    rec = ("chown: changing ownership of '/testbed/lib/a.py': Operation not permitted\n"
           "chown: changing ownership of '/testbed/lib/b.py': Operation not permitted\n\n[exit code: 1]")
    live = ("chown: changing ownership of '/testbed/lib/z.py': Operation not permitted\n\n[exit code: 1]")
    assert _verify_pair(tmp_path, rec, live) is None


def test_volatile_v5_stash_pop_sha(tmp_path):
    """Normalize the run-created SHA from a dropped stash."""
    rec = "On branch base\nDropped refs/stash@{0} (c625b512aa)\n"
    live = rec.replace("c625b512aa", "e724bb7b00")
    assert _verify_pair(tmp_path, rec, live) is None


def test_volatile_v5_sympy_runner_banner(tmp_path):
    """Normalize SymPy runner seeds, hashes, and duration."""
    rec = ("test process starts\nrandom seed:        47291\n"
           "hash randomization: on (PYTHONHASHSEED=3158036575)\n"
           "=== tests finished: 47 passed, 1 failed, in 1.31 seconds ===\n")
    live = (rec.replace("47291", "90210")
            .replace("3158036575", "111")
            .replace("1.31", "2.05"))
    assert _verify_pair(tmp_path, rec, live) is None
    # verdict counts are content: a changed count still halts
    import pytest as _pytest
    with _pytest.raises(Exception):
        _verify_pair(tmp_path / "n7", rec, rec.replace("47 passed", "46 passed"))


def test_volatile_v6_ls_parent_dir_size(tmp_path):
    """Normalize the parent row size, but compare named row sizes."""
    import pytest as _pytest
    rec = ("total 516\n"
           "drwxrwxrwx 1 root root   4096 Jan  1  2020 .\n"
           "drwxr-xr-x 1 root root      6 Jan  1  2020 ..\n"
           "-rw-r--r-- 1 root root    129 Jan  1  2020 .codecov.yml\n")
    live = rec.replace("root      6 Jan  1  2020 ..", "root     10 Jan  1  2020 ..")
    assert _verify_pair(tmp_path, rec, live) is None
    with _pytest.raises(Exception):
        _verify_pair(tmp_path / "n8", rec,
                     rec.replace("129 Jan  1  2020 .codecov.yml",
                                 "999 Jan  1  2020 .codecov.yml"))


def test_volatile_v7_ls_all_dir_row_sizes(tmp_path):
    """Normalize directory row sizes, but compare file row sizes."""
    import pytest as _pytest
    rec = ("drwxrwxrwx 1 root root     25 Jan  1  2020 docs\n"
           "-rw-r--r-- 1 root root    129 Jan  1  2020 setup.py\n")
    live = rec.replace("root     25 Jan", "root     33 Jan")
    assert _verify_pair(tmp_path, rec, live) is None
    with _pytest.raises(Exception):
        _verify_pair(tmp_path / "n9", rec, rec.replace("129", "777"))


def test_volatile_v7_chown_whole_line(tmp_path):
    """Normalize chown message forms and the reported file set."""
    rec = ("chown: changing ownership of '/a/x.py': Operation not permitted\n"
           "chown: cannot read directory '/a/sub': Permission denied\n\n[exit code: 1]")
    live = "chown: changing ownership of '/a/zzz.py': Operation not permitted\n\n[exit code: 1]"
    assert _verify_pair(tmp_path, rec, live) is None


def test_volatile_v7_sphinx_scm_hash(tmp_path):
    rec = "Running Sphinx v5.0.0+/c9af4d7\n"
    assert _verify_pair(tmp_path, rec, rec.replace("c9af4d7", "ab12cd3")) is None


def test_volatile_v7_dd_timing(tmp_path):
    import pytest as _pytest
    rec = "1024 bytes (1.0 kB) copied, 0.000139519 s, 126 MB/s\n"
    live = "1024 bytes (1.0 kB) copied, 0.000200013 s, 89 MB/s\n"
    assert _verify_pair(tmp_path, rec, live) is None
    with _pytest.raises(Exception):
        _verify_pair(tmp_path / "n10", rec, rec.replace("1024 bytes", "2048 bytes"))


def test_volatile_v8_snapshot_id_without_fs_suffix(tmp_path):
    """Normalize snapshot IDs in upperdir and workdir paths."""
    rec = "upperdir=/mnt/x/snapshots/66484/work,workdir=/mnt/x/snapshots/66484\n"
    live = rec.replace("66484", "68206")
    assert _verify_pair(tmp_path, rec, live) is None


def test_volatile_v8_harness_summary_annotations(tmp_path):
    """Normalize summary counts that vary with traversal order."""
    rec = ("chown: changing ownership of '/a/x.py': Operation not permitted\n\n"
           "[... 82816 chars omitted ...]\n"
           "chown: changing ownership of '/a/y.py': Operation not permitted\n"
           "  ... [×1488 similar lines]\n"
           "chown: changing ownership of '/a/z': Operation not permitted\n\n[exit code: 1]")
    live = ("chown: changing ownership of '/b/q.pdf': Operation not permitted\n\n"
            "[... 80120 chars omitted ...]\n"
            "chown: changing ownership of '/b/r.py': Operation not permitted\n"
            "  ... [×993 similar lines]\n"
            "chown: changing ownership of '/b/s': Operation not permitted\n\n[exit code: 1]")
    assert _verify_pair(tmp_path, rec, live) is None


def test_volatile_v9_sphinx_latex_date(tmp_path):
    """Normalize the Sphinx build date in LaTeX output."""
    import pytest as _pytest
    rec = "\\\\title{x}\n\\\\date{Jun 07, 2026}\n"
    live = rec.replace("Jun 07", "Jun 10")
    assert _verify_pair(tmp_path, rec, live) is None
    with _pytest.raises(Exception):  # title text is content
        _verify_pair(tmp_path / "n11", rec, rec.replace("{x}", "{y}"))


def test_volatile_v10_runner_date_stamp(tmp_path):
    rec = "Date: 2026-06-07T13:07:56\n1 passed\n"
    live = rec.replace("2026-06-07T13:07:56", "2026-06-10T04:00:08")
    assert _verify_pair(tmp_path, rec, live) is None


def test_volatile_v10_stat_directory_size_only(tmp_path):
    """Normalize directory sizes from stat, but compare file sizes."""
    import pytest as _pytest
    rec = "  Size: 25        Blocks: 0          IO Block: 4096   directory\n"
    live = rec.replace("Size: 25", "Size: 33")
    assert _verify_pair(tmp_path, rec, live) is None
    frec = "  Size: 3280      Blocks: 8          IO Block: 4096   regular file\n"
    with _pytest.raises(Exception):
        _verify_pair(tmp_path / "n12", frec, frec.replace("3280", "9999"))


def test_volatile_v10_exit_120_pairs_with_1_only(tmp_path):
    """Normalize shutdown exits 120 and 1, but keep exit 2 distinct."""
    import pytest as _pytest
    rec = "ImportError: x\n\n[exit code: 1]"
    live = "ImportError: x\n\n[exit code: 120]"
    assert _verify_pair(tmp_path, rec, live) is None
    with _pytest.raises(Exception):
        _verify_pair(tmp_path / "n13", rec, rec.replace("code: 1]", "code: 2]"))


def test_volatile_v11_stat_birth_leading_space(tmp_path):
    """Normalize the indented Birth timestamp from GNU stat."""
    rec = " Birth: 2026-06-07 12:45:18.000000000 +0000\n"
    live = " Birth: 2026-06-10 04:00:08.000000000 +0000\n"
    assert _verify_pair(tmp_path, rec, live) is None


def test_volatile_v12_sidecar_recapture_cases(tmp_path):
    """Sidecar recapture drain: seven process/container volatile fields."""
    assert VOLATILE_NORMALIZATION_VERSION == "replay_volatile_norm_v14"

    cases = [
        (
            "libcap: <CDLL 'libcap.so.2', handle 195c2130 at 0xXXXX>\n",
            "libcap: <CDLL 'libcap.so.2', handle 24e5130 at 0xXXXX>\n",
        ),
        ("PID: 452\n", "PID: 447\n"),
        ("0.1.0.dev1+g9fc64e8.d20260609\n",
         "0.1.0.dev1+g9fc64e8.d20260611\n"),
        ("Traceback (most recent call last):\n",
         "Matplotlib is building the font cache; this may take a moment.\n"
         "Traceback (most recent call last):\n"),
        (
            "File stat: os.stat_result(st_mode=33188, st_ino=28437238921, "
            "st_dev=64, st_nlink=1)\n",
            "File stat: os.stat_result(st_mode=33188, st_ino=28437238921, "
            "st_dev=78, st_nlink=1)\n",
        ),
        (
            "   1K-blocks       Used   Available Use% Mounted on\n"
            "overlay        15413026816 4686698380 10726328436  31% /\n",
            "   1K-blocks       Used   Available Use% Mounted on\n"
            "overlay        15413026816 4705412756 10707614060  31% /\n",
        ),
        (
            "patch: **** Can't create temporary file requests/models.py.oLcYHHw : Permission denied\n",
            "patch: **** Can't create temporary file requests/models.py.oDVutZW : Permission denied\n",
        ),
    ]
    for i, (recorded, live) in enumerate(cases):
        assert _verify_pair(tmp_path / f"v12_{i}", recorded, live) is None


def test_volatile_v12_narrowness(tmp_path):
    """v12 metadata rules must not hide meaningful path/content changes."""
    import pytest as _pytest

    with _pytest.raises(Exception):
        _verify_pair(
            tmp_path / "df_mount",
            "overlay        1 2 3 4 /testbed\n",
            "overlay        1 2 3 4 /other\n",
        )
    with _pytest.raises(Exception):
        _verify_pair(
            tmp_path / "patch_target",
            "patch: **** Can't create temporary file requests/models.py.oLcYHHw : Permission denied\n",
            "patch: **** Can't create temporary file requests/sessions.py.oDVutZW : Permission denied\n",
        )


def test_volatile_v13_sidecar_recapture_cases(tmp_path):
    """Sidecar recapture v13: next-layer pipe and stat inode metadata."""
    assert _verify_pair(
        tmp_path / "pipe_inode",
        "1 -> pipe:[165995190]\n2 -> /dev/null\n",
        "1 -> pipe:[181215652]\n2 -> /dev/null\n",
    ) is None
    assert _verify_pair(
        tmp_path / "stat_inode",
        "Device: <volatile>\tInode: 24086873590  Links: 1\n",
        "Device: <volatile>\tInode: 21924096843  Links: 1\n",
    ) is None


def test_volatile_v13_narrowness(tmp_path):
    import pytest as _pytest

    with _pytest.raises(Exception):
        _verify_pair(
            tmp_path / "pipe_target",
            "1 -> pipe:[165995190]\n2 -> /dev/null\n",
            "1 -> pipe:[181215652]\n2 -> /tmp/out\n",
        )
    with _pytest.raises(Exception):
        _verify_pair(
            tmp_path / "stat_links",
            "Device: <volatile>\tInode: 24086873590  Links: 1\n",
            "Device: <volatile>\tInode: 21924096843  Links: 2\n",
        )


def test_volatile_v14_proc_self_fd_path(tmp_path):
    """Sidecar recapture v14: /proc/self/fd symlink target embeds PID."""
    assert _verify_pair(
        tmp_path / "proc_self_fd",
        "3 -> /proc/459/fd\n",
        "3 -> /proc/452/fd\n",
    ) is None


def test_volatile_v14_proc_path_narrowness(tmp_path):
    import pytest as _pytest

    with _pytest.raises(Exception):
        _verify_pair(
            tmp_path / "proc_status",
            "3 -> /proc/459/status\n",
            "3 -> /proc/452/status\n",
        )
    with _pytest.raises(Exception):
        _verify_pair(
            tmp_path / "proc_fd_number",
            "3 -> /proc/459/fd\n",
            "4 -> /proc/452/fd\n",
        )
