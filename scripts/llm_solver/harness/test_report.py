"""Collect invocation-owned runner reports separately from diagnostic stdout."""
from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
import os
import shlex
import subprocess
import uuid
import xml.etree.ElementTree as ET

from .task_path import TaskPath, resolve_task_path
from .time_budget import BudgetExhausted


def parse_junit(data: bytes) -> dict:
    """Read complete JUnit cases; captured stdout never supplies a verdict."""
    root = ET.fromstring(data)
    if root.tag not in {"testsuites", "testsuite"}:
        raise ValueError("not a JUnit report")
    tests = {}
    for case in root.iter("testcase"):
        name = case.get("name")
        if not name:
            raise ValueError("test case has no identity")
        identity = f"{case.get('classname', '')}::{name}"
        if identity in tests:
            raise ValueError("duplicate test identity")
        verdict = ("ERROR" if case.find("error") is not None else
                   "FAILED" if case.find("failure") is not None else
                   "SKIPPED" if case.find("skipped") is not None else "PASSED")
        tests[identity] = verdict
    if not tests:
        raise ValueError("report contains no test cases")
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    if sum(int(suite.get("tests", "-1")) for suite in suites) != len(tests):
        raise ValueError("test case count does not match report coverage")
    summary = {key: sum(v == value for v in tests.values()) for key, value in
               (("passed", "PASSED"), ("failed", "FAILED"),
                ("errors", "ERROR"), ("skipped", "SKIPPED"))}
    return {"tests": tests, "summary": summary, "failure_details": []}


@dataclass
class TestReportCapture:
    environment: dict
    runner: str
    completed_exit_codes: tuple = ()
    invocation: str = field(default_factory=lambda: uuid.uuid4().hex)
    path: object = None
    plugin_path: object = None
    per_invocation: bool = False
    record: dict = field(default_factory=dict)

    def finish(self, exit_status, timed_out=False):
        self.record.update({"runner": self.runner, "invocation": self.invocation,
                       "status": "unavailable", "exit_status": exit_status,
                       "timed_out": bool(timed_out)})
        if self.path is None or timed_out or exit_status not in self.completed_exit_codes:
            return
        try:
            paths = self.report_paths()
            if not paths:
                raise ValueError('no owned invocation reports')
            tests, roots, hashes = {}, set(), []
            for path in paths:
                data = path.read_bytes()
                context = ET.fromstring(data).get('yuj_collection_root', '')
                if self.per_invocation and not context:
                    raise ValueError('report has no observed collection root')
                roots.add(context)
                parsed = parse_junit(data)
                prefix = f'{self.runner}:{context}::' if context else f'{self.runner}:'
                for key, value in parsed['tests'].items():
                    identity = prefix + key
                    if identity in tests and tests[identity] != value:
                        raise ValueError('invocations disagree on a test verdict')
                    tests[identity] = value
                hashes.append(hashlib.sha256(data).hexdigest())
            summary = {key: sum(v == value for v in tests.values()) for key, value in
                       (('passed', 'PASSED'), ('failed', 'FAILED'), ('errors', 'ERROR'), ('skipped', 'SKIPPED'))}
            self.record.update(status='available', tests=tests, summary=summary, failure_details=[],
                collection_roots=sorted(roots), report_count=len(paths), report_hashes=hashes,
                sha256=hashes[0] if len(hashes) == 1 else hashlib.sha256(''.join(hashes).encode()).hexdigest())
        except (OSError, ValueError, ET.ParseError, BudgetExhausted, subprocess.TimeoutExpired) as error:
            self.record["reason"] = str(error)

    def report_paths(self):
        return sorted(self.path.parent.glob(self.path.name + '.*.xml')) if self.per_invocation else [self.path]


@contextmanager
def capture_test_report(cwd, cfg, environment=None, *, command=None):
    """Use a reporting option declared by the selected runner, when enabled.

    The file is freshly reserved in the execution view, read after process
    completion and removed before returning control. It records what the runner
    reports, not a claim that arbitrary task/runner code is adversary-proof.
    """
    from ..language_quirks import load_run_tests_quirk_object, FORMATS_DIR
    from ..bash_quirks import load_output_control
    from ..bash_quirks._output import _is_test_command

    if cfg is None or not (
        getattr(cfg, "done_require_pretest_parity", False)
        or int(getattr(cfg, "post_mutation_verification_gate_after", 0) or 0) > 0
    ):
        yield None
        return
    quirk = load_run_tests_quirk_object(cwd, runner=cfg.analysis_task_format)
    if command is not None:
        control = load_output_control(FORMATS_DIR / f"{quirk.runner}.toml")
        if control is None or not _is_test_command(command, control):
            yield None
            return
    spec = quirk.extra_fields.get("native_report") or {}
    capture = TestReportCapture(dict(os.environ if environment is None else environment),
                                quirk.runner, tuple(spec.get("completed_exit_codes", ())))
    # Report availability must not prevent an otherwise runnable command.
    try:
        if spec.get("format") == "junit" and spec.get("environment") and spec.get("argument"):
            root = resolve_task_path(cwd, ".")
            directory = root / ".tool_output"
            if directory.is_symlink():
                raise ValueError("report directory is a symlink")
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"report_{capture.invocation}.xml"
            if isinstance(path, TaskPath):
                path.files.create_bytes(str(path), b"")
            else:
                with path.open("xb"):
                    pass
            capture.path = path
            plugin = spec.get('collection_plugin')
            if plugin:
                from .component_selection import _create
                module = 'yuj_report_' + capture.invocation
                code_path = directory / (module + '.py')
                _create(code_path, (FORMATS_DIR / plugin).read_bytes())
                capture.plugin_path = code_path
                capture.per_invocation = True
                capture.environment.update(
                    PYTHONPATH=os.pathsep.join(filter(None, [str(directory), capture.environment.get('PYTHONPATH')])),
                    PYTEST_PLUGINS=','.join(filter(None, [capture.environment.get('PYTEST_PLUGINS'), module])),
                    YUJ_NATIVE_REPORT=str(path))
            variable = spec["environment"]
            option = spec["argument"].format(path=str(path))
            capture.environment[variable] = (
                capture.environment.get(variable, "") + " " + shlex.quote(option)
            ).strip()
    except (OSError, ValueError) as error:
        capture.record["reason"] = str(error)
    try:
        yield capture
    finally:
        if capture.path is not None:
            try:
                for path in capture.report_paths():
                    path.unlink(missing_ok=True)
                capture.path.unlink(missing_ok=True)
                if capture.plugin_path is not None:
                    capture.plugin_path.unlink(missing_ok=True)
                    cache = capture.plugin_path.parent / '__pycache__'
                    for path in cache.glob(capture.plugin_path.stem + '.*.pyc'):
                        path.unlink(missing_ok=True)
            except (OSError, BudgetExhausted, subprocess.TimeoutExpired) as error:
                capture.record["cleanup_error"] = str(error)
