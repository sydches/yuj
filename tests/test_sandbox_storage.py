"""Installed libraries use sandbox storage without application-specific defaults."""
import importlib.util
import json
from pathlib import Path
import shlex
import sys
import textwrap

import pytest

from scripts.llm_solver.harness.sandbox.env_policy import DEFAULT_FIXED_ENVIRONMENT
from test_sandbox_filesystem import run, task_view


@pytest.mark.parametrize("home_kind", ["observed", "nested", "read_only"])
def test_matplotlib_uses_private_storage_without_fixed_directory(task_view, home_kind):
    if not importlib.util.find_spec("matplotlib"):
        pytest.skip("optional Matplotlib installation unavailable")
    task, home, env = task_view
    if home_kind == "nested":
        home = home / "another" / "layout"
        home.mkdir(parents=True)
    elif home_kind == "read_only":
        home = Path("/usr")
    env = {**env, **DEFAULT_FIXED_ENVIRONMENT, "HOME": str(home),
           "PATH": str(Path(sys.executable).parent) + ":" + env["PATH"]}
    assert "MPLCONFIGDIR" not in env
    if home_kind != "read_only":
        private = home / ".config" / "matplotlib" / "operator-secret"
        private.parent.mkdir(parents=True)
        private.write_text("SYNTHETIC HOST CONFIG")
    source = textwrap.dedent('''
        import json, os
        from pathlib import Path
        assert "MPLCONFIGDIR" not in os.environ
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import pyplot
        config = Path(matplotlib.get_configdir())
        cache = Path(matplotlib.get_cachedir())
        assert not (config / "operator-secret").exists()
        for directory in (config, cache):
            (directory / "storage-check").write_text("private write")
        figure, axes = pyplot.subplots()
        axes.plot([0, 1], [0, 1])
        figure.savefig("plot.png")
        pyplot.close(figure)
        assert Path("plot.png").read_bytes().startswith(b"\\x89PNG\\r\\n\\x1a\\n")
        print(json.dumps({"version": matplotlib.__version__,
                          "config": str(config), "cache": str(cache)}))
    ''')
    result = run(task, env, shlex.quote(sys.executable) + " -c " + shlex.quote(source))
    assert result.returncode == 0, result.stderr
    observed = json.loads(result.stdout)
    storage = Path("/tmp") if home_kind == "read_only" else home
    assert all(Path(observed[key]).is_relative_to(storage) for key in ("config", "cache"))
    assert (task / "plot.png").is_file()
    if home_kind != "read_only":
        assert private.read_text() == "SYNTHETIC HOST CONFIG"
        assert not (home / ".cache").exists()
        assert not (private.parent / "storage-check").exists()


def test_other_library_uses_observed_private_home(task_view):
    if not importlib.util.find_spec("platformdirs"):
        pytest.skip("optional platformdirs installation unavailable")
    task, home, env = task_view
    env = {**env, **DEFAULT_FIXED_ENVIRONMENT,
           "PATH": str(Path(sys.executable).parent) + ":" + env["PATH"]}
    assert "MPLCONFIGDIR" not in env
    source = textwrap.dedent('''
        from pathlib import Path
        from platformdirs import user_cache_path, user_config_path
        for directory in (user_cache_path("storage-fixture"), user_config_path("storage-fixture")):
            assert directory.is_relative_to(Path.home())
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "marker").write_text("private write")
        print("storage works")
    ''')
    result = run(task, env, shlex.quote(sys.executable) + " -c " + shlex.quote(source))
    assert result.returncode == 0 and result.stdout == "storage works\n", result.stderr
    assert not list(home.iterdir())


def test_application_environment_is_observed_or_declared_not_invented():
    from _config_helpers import make_config
    from scripts.llm_solver.config import load_config
    from scripts.llm_solver.harness.sandbox.env_policy import EnvironmentPolicy

    cfg = load_config()
    assert "MPLCONFIGDIR" not in cfg.sandbox_env_set
    assert "MPLCONFIGDIR" not in DEFAULT_FIXED_ENVIRONMENT
    assert "MPLCONFIGDIR" not in make_config().sandbox_env_set
    observed = {"MPLCONFIGDIR": "/permitted/environment/path"}
    assert EnvironmentPolicy(inherit="all").resolve(observed) == observed
    policy = EnvironmentPolicy(inherit="all", set={"MPLCONFIGDIR": "/declared/policy/path"})
    assert policy.resolve(observed)["MPLCONFIGDIR"] == "/declared/policy/path"


def test_direct_config_dispatch_preserves_library_selected_storage(task_view, monkeypatch):
    from _config_helpers import make_config
    from scripts.llm_solver.harness.tools import dispatch

    if not importlib.util.find_spec("matplotlib"):
        pytest.skip("optional Matplotlib installation unavailable")
    task, home, env = task_view
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("PATH", str(Path(sys.executable).parent) + ":" + env["PATH"])
    cfg = make_config(sandbox_bash=True, sandbox_required=True,
                      bash_transforms_universal_enabled=False, bash_quirks_forbidden_enabled=False)
    source = textwrap.dedent('''
        import os
        from pathlib import Path
        assert "MPLCONFIGDIR" not in os.environ
        import matplotlib
        config = Path(matplotlib.get_configdir())
        assert config.is_relative_to(Path.home())
        (config / "dispatch-check").write_text("private cache")
        print("library chose writable storage")
    ''')
    facts = {}
    result = dispatch("bash", {"cmd": shlex.join([sys.executable, "-c", source])},
                      cwd=str(task), cfg=cfg, execution_metadata=facts)
    assert facts["exit_status"] == 0, result
    assert "library chose writable storage" in result
    assert not list(home.iterdir())
