"""Search-tool provenance comes only from the bound task environment."""
import subprocess
from unittest.mock import MagicMock

import pytest

from _config_helpers import make_config
from llm_solver.harness.solver import collect_provenance, _task_search_versions
from llm_solver.harness.task_files import NamespaceFiles
from llm_solver.harness.task_path import activate_task_files


def test_collect_provenance_does_not_substitute_host_search_versions(tmp_path):
    """An unbound task has unknown versions even when host tools exist."""
    prov = collect_provenance(make_config(), task_cwd=tmp_path)
    assert "rg_version" not in prov
    assert "grep_version" not in prov


@pytest.mark.parametrize("has_rg", [True, False])
def test_collect_provenance_search_versions_use_task_path(tmp_path, has_rg):
    task_bin = tmp_path / "bin"
    task_bin.mkdir()
    expected = {"grep_version": "task grep 1.2", "rg_version": ""}
    for tool in (["rg", "grep"] if has_rg else ["grep"]):
        binary = task_bin / tool
        binary.write_text('#!/bin/bash\nprintf "task ' + tool + ' 1.2\\nextra line\\n"\n')
        binary.chmod(0o755)
        expected[tool + "_version"] = "task " + tool + " 1.2"
    calls = []

    def run(script, args, data):
        calls.append(args)
        return subprocess.run(
            ["/bin/bash", "--noprofile", "--norc", "-c", script, "version-test", *args],
            env={"PATH": str(task_bin)}, input=data, capture_output=True, check=False,
        )

    files = NamespaceFiles(str(tmp_path), run, binding={"test": "task versions"})
    with activate_task_files(files, host_root=tmp_path):
        prov = collect_provenance(make_config(), task_cwd=tmp_path)
    assert {key: prov[key] for key in expected} == expected
    assert calls == [["rg", "grep"]]


def test_search_version_probe_failure_remains_unknown():
    files = MagicMock()
    files.run.side_effect = OSError("task transport unavailable")
    assert _task_search_versions(files) == {}
