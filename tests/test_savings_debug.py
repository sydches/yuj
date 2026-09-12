"""Debug preview equivalence and task accounting cleanup."""
import json
from pathlib import Path
import random
import subprocess
import sys

import pytest

from scripts.llm_solver.harness import savings, savings_debug


def old_preview(before, after):
    if before == after:
        return []
    prefix = 0
    while prefix < min(len(before), len(after)) and before[prefix] == after[prefix]:
        prefix += 1
    left, right = len(before), len(after)
    while left > prefix and right > prefix and before[left-1] == after[right-1]:
        left -= 1
        right -= 1
    return [savings_debug._change_region(before, after, prefix, left, prefix, right)]


@pytest.mark.parametrize('before,after', [
    ('', ''), ('', 'abc'), ('abc', ''), ('same', 'same'), ('abc', 'xyz'),
    ('aaaa', 'aaa'), ('aaa', 'aaaa'), ('ababa', 'babab'),
    ('hé🙂\r\nold\n末', 'hé🙂\r\nnew\n末'), ('e\u0301', 'é'),
    ('a'*84000, 'a'*83999+'b'), ('a'*1048576, 'b'+'a'*1048575),
    ('🙂'*42000+'old'+'末'*42000, '🙂'*42000+'new'+'末'*42000),
])
def test_preview_matches_original_for_edges_unicode_and_large_repetition(before, after):
    assert savings_debug._changed_snippets(before, after) == old_preview(before, after)


def test_preview_matches_original_across_varied_insert_delete_replace():
    rng = random.Random(730)
    alphabet = 'ab\n\r\té🙂末\u0301'
    for _ in range(1000):
        before = ''.join(rng.choices(alphabet, k=rng.randrange(200)))
        start, stop = sorted([rng.randrange(len(before)+1), rng.randrange(len(before)+1)])
        after = before[:start] + ''.join(rng.choices(alphabet, k=rng.randrange(30))) + before[stop:]
        assert savings_debug._changed_snippets(before, after) == old_preview(before, after)


@pytest.mark.parametrize('failure', [RuntimeError, KeyboardInterrupt, SystemExit])
def test_actual_driver_closes_ledger_on_setup_failure(tmp_path, monkeypatch, failure):
    from _config_helpers import make_config
    from scripts.llm_solver.harness._loop import driver
    path = tmp_path / 'ledger.jsonl'
    handles = []

    def fail_setup(*args):
        ledger = savings.open_ledger(path, transform_log_mode='debug')
        handles.append(ledger._file)
        ledger.record_transform('b', 'l', 'm', before='old', after='new', surface='tool_output')
        raise failure('interrupted setup')

    monkeypatch.setattr(driver, 'resolve_run_paths', fail_setup)
    with pytest.raises(failure, match='interrupted setup'):
        driver.solve_task(tmp_path, make_config(), None)
    assert handles[0].closed
    assert isinstance(savings.get_ledger(), savings._NullLedger)
    row = json.loads(path.read_text())
    assert (tmp_path / row['input_full_path']).read_text() == 'old'
    assert (tmp_path / row['output_full_path']).read_text() == 'new'


@pytest.mark.parametrize('signal_name', ['SIGINT', 'SIGTERM'])
def test_real_signals_close_debug_ledger(tmp_path, signal_name):
    script = r'''
import json, os, signal, sys
from pathlib import Path
from scripts.llm_solver.harness import savings
from scripts.llm_solver.harness._loop.interrupted_turn import ExitDiagnostics
root = Path(sys.argv[1])
handles = []
@savings.savings_ledger_scope
def task():
    with ExitDiagnostics(root / 'trace.jsonl', session_number=1):
        ledger = savings.open_ledger(root / 'ledger.jsonl', transform_log_mode='debug')
        handles.append(ledger._file)
        ledger.record_transform('b', 'l', 'm', before='a'*84000, after='b'*84000, surface='tool_output')
        os.kill(os.getpid(), getattr(signal, sys.argv[2]))
try:
    task()
except (KeyboardInterrupt, SystemExit):
    pass
assert handles[0].closed
assert isinstance(savings.get_ledger(), savings._NullLedger)
row = json.loads((root / 'ledger.jsonl').read_text())
assert (root / row['input_full_path']).read_text() == 'a'*84000
assert (root / row['output_full_path']).read_text() == 'b'*84000
assert sys.argv[2] in (root / 'trace.jsonl').read_text()
'''
    result = subprocess.run([sys.executable, '-c', script, str(tmp_path), signal_name],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
