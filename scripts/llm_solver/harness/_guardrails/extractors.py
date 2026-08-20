"""Pure extraction / classification / signature helpers — extracted from guardrails.py."""
from __future__ import annotations

import hashlib
import json
import re
import shlex
from typing import Any

from ..bash_write_classification import is_bash_legacy_mutation_like
from .._shell_patterns import TEST_COMMAND_RE as _TEST_COMMAND_RE
from ..tool_specs import GUARDRAIL_MUTATION_TOOL_NAMES
from .state import GuardrailState


# Canonical set of file-mutating tool names for guardrails.
MUTATION_TOOLS = GUARDRAIL_MUTATION_TOOL_NAMES


def _extract_read_path(
    tc_name: str,
    tc_args: dict | None = None,
    *,
    focus_key: str = "",
    focus_display: str = "",
) -> str:
    if tc_name == "read" and isinstance(tc_args, dict):
        raw = tc_args.get("path") or tc_args.get("file_path")
        if isinstance(raw, str):
            return raw
    if tc_name == "bash" and focus_key.startswith("file:"):
        return focus_display
    return ""


def _is_concrete_file_path(path: str) -> bool:
    if not path:
        return False
    candidate = path.split("::", 1)[0].rstrip(",")
    if candidate in {".", "..", "/"}:
        return False
    if candidate.endswith("/"):
        return False
    name = candidate.rsplit("/", 1)[-1]
    return bool(name) and name not in {".", ".."}


def _extract_bash_cmd(tc_name: str, tc_args: dict | None = None) -> str:
    if tc_name != "bash" or not isinstance(tc_args, dict):
        return ""
    raw = tc_args.get("cmd")
    return raw if isinstance(raw, str) else ""


def _is_bash_write_like(tc_name: str, tc_args: dict | None = None) -> bool:
    """True for bash commands that structurally look like source mutation."""
    cmd = _extract_bash_cmd(tc_name, tc_args)
    return is_bash_legacy_mutation_like(cmd)


def _is_test_command(tc_name: str, tc_args: dict | None = None) -> bool:
    """True if the call invokes the test gate.

    Covers two shapes:
      - bash with a pytest/cargo/go-test-style command line — matched
        via _TEST_COMMAND_RE on the extracted bash command.
      - the dedicated `run_tests` tool — language_quirks-driven, the
        canonical gate when a runner is registered.
    """
    if tc_name == "run_tests":
        return True
    return bool(_TEST_COMMAND_RE.search(_extract_bash_cmd(tc_name, tc_args)))


def _is_test_read(
    tc_name: str,
    tc_args: dict | None = None,
    *,
    focus_key: str = "",
    focus_display: str = "",
) -> bool:
    path = _extract_read_path(tc_name, tc_args, focus_key=focus_key, focus_display=focus_display)
    return _is_concrete_file_path(path) and _looks_like_test_path(path)


def _is_concrete_read(
    tc_name: str,
    tc_args: dict | None = None,
    *,
    focus_key: str = "",
    focus_display: str = "",
) -> bool:
    path = _extract_read_path(tc_name, tc_args, focus_key=focus_key, focus_display=focus_display)
    return _is_concrete_file_path(path)


def _clear_commit_contract(state: GuardrailState) -> None:
    state.commit_pending = False
    state.commit_source_path = ""
    state.commit_violation_count = 0
    state.commit_turns_since_arm = 0
    state.contract_block_sig = ""
    state.contract_block_count = 0


def _clear_recovery_mode(state: GuardrailState) -> None:
    state.recovery_mode_active = False
    state.recovery_reason = ""
    state.recovery_target = ""
    state.recovery_turns_since_arm = 0
    state.contract_block_sig = ""
    state.contract_block_count = 0


def _clear_mutation_repeat_state(state: GuardrailState) -> None:
    state.mutation_repeat_sig = ""
    state.mutation_repeat_target = ""
    state.mutation_repeat_count = 0
    state.mutation_repeat_block_sig = ""
    state.mutation_repeat_block_count = 0


def _arm_recovery_mode(state: GuardrailState, *, reason: str, target: str) -> None:
    state.recovery_mode_active = True
    state.recovery_reason = reason
    state.recovery_target = target
    state.recovery_turns_since_arm = 0


def _contract_violation_signature(
    cfg: Any,
    tc_name: str,
    tc_args: dict | None = None,
    *,
    focus_key: str = "",
    focus_display: str = "",
) -> str:
    if getattr(cfg, "contract_equivalent_action_classes_enabled", False):
        coarse = _equivalent_contract_violation_signature(
            tc_name, tc_args, focus_key=focus_key, focus_display=focus_display
        )
        if coarse:
            return coarse
    if focus_key:
        return focus_key
    if focus_display:
        return f"{tc_name}:{focus_display}"
    raw = json.dumps(tc_args or {}, sort_keys=True)
    return f"{tc_name}:{raw}"


def _equivalent_contract_violation_signature(
    tc_name: str,
    tc_args: dict | None = None,
    *,
    focus_key: str = "",
    focus_display: str = "",
) -> str:
    """Collapse semantically-equivalent non-progress moves into stable classes."""
    if tc_name == "read":
        path = _extract_read_path(tc_name, tc_args, focus_key=focus_key, focus_display=focus_display)
        if path:
            return f"read:{path}"
    if tc_name == "bash":
        cmd = (_extract_bash_cmd(tc_name, tc_args) or "").strip().lower()
        target = (focus_display or "").strip().rstrip("/")
        if "python -c" in cmd and "import " in cmd:
            return "bash:python-c-import-probe"
        if cmd.startswith("ls ") or cmd.startswith("ls\t"):
            return f"bash:ls:{target or '.'}"
        if cmd.startswith("find ") or cmd.startswith("cd ") and " find " in cmd:
            return f"bash:find:{target or '.'}"
        if cmd.startswith("pwd"):
            return "bash:pwd"
    if tc_name in ("glob", "grep"):
        path = (tc_args or {}).get("path", "")
        if isinstance(path, str) and path:
            return f"{tc_name}:path:{path}"
    return ""


def _mutation_signature(
    tc_name: str,
    tc_args: dict | None = None,
    *,
    focus_display: str = "",
) -> tuple[str, str]:
    if tc_name not in MUTATION_TOOLS or not isinstance(tc_args, dict):
        return "", ""
    if tc_name == "apply_patch":
        # apply_patch is multi-file by construction — no single `path`
        # argument. Sign over the patch text itself; target is "<patch>"
        # as a stable label so digest replay tools can group repeats.
        payload = str(tc_args.get("patch", ""))
        digest = hashlib.sha1(payload.encode("utf-8", errors="ignore")).hexdigest()[:12]
        return f"apply_patch:{digest}", "<patch>"
    raw_path = tc_args.get("path") or tc_args.get("file_path") or focus_display
    path = raw_path if isinstance(raw_path, str) else ""
    if not _is_concrete_file_path(path):
        path = focus_display if _is_concrete_file_path(focus_display) else "current file"
    if tc_name in ("edit", "str_replace"):
        # str_replace shares edit's old/new contract in historical traces.
        # Read both canonical kwarg names so traces written under either
        # vocabulary hash distinctly.
        payload = json.dumps(
            {
                "old": tc_args.get("old_str") or tc_args.get("old_string", ""),
                "new": tc_args.get("new_str") or tc_args.get("new_string", ""),
            },
            sort_keys=True,
            ensure_ascii=True,
        )
    else:
        payload = str(tc_args.get("content", ""))
    digest = hashlib.sha1(payload.encode("utf-8", errors="ignore")).hexdigest()[:12]
    return f"{tc_name}:{path}:{digest}", path




def _contract_abort_allowed(
    state: GuardrailState,
    cfg: Any,
    *,
    lane: str,
) -> bool:
    if getattr(cfg, "contract_abort_requires_zero_mutation", False) and state.has_mutated:
        return False
    if lane == "commit":
        min_turns = int(getattr(cfg, "contract_abort_min_turns_since_commit_arm", 0) or 0)
        if min_turns > 0 and state.commit_turns_since_arm < min_turns:
            return False
        return True
    min_turns = int(getattr(cfg, "contract_abort_min_turns_since_recovery_arm", 0) or 0)
    if min_turns > 0 and state.recovery_turns_since_arm < min_turns:
        return False
    return True


def _record_contract_block(state: GuardrailState, sig: str) -> int:
    if not sig:
        state.contract_block_sig = ""
        state.contract_block_count = 0
        return 0
    if state.contract_block_sig == sig:
        state.contract_block_count += 1
    else:
        state.contract_block_sig = sig
        state.contract_block_count = 1
    return state.contract_block_count




# js/ts leaf suffixes that are unambiguously test files regardless of
# directory context (`foo.test.js`, `foo.spec.tsx`, ...).
_JS_TEST_LEAF_SUFFIXES = (
    ".test.js", ".test.ts", ".test.jsx", ".test.tsx",
    ".spec.js", ".spec.ts", ".spec.jsx", ".spec.tsx",
)


def _looks_like_test_path(path: str) -> bool:
    """True iff `path` is plausibly a test target (file or directory).

    A plain substring match on "test" was way too loose: it accepted
    `testbed/`, `attestation.py`, `latest/`, `nottest_helpers.py`. The
    rule requires either:
      - a path SEGMENT that is exactly `test` or `tests`, or
      - the leaf filename to start with `test_` or end with `_test.py`
        / `_tests.py`.

    The original pytest-only rule missed Go, JavaScript, TypeScript,
    and Rust conventions, so the "did you read the test file first" guardrail
    was effectively inert on those languages. Extended additively below
    — every existing python match keeps matching unchanged:
      - go: `foo_test.go`
      - js/ts: `foo.test.js`, `foo.spec.ts`, ..., `__tests__/` directory
      - rust: `tests/` directory (already covered by the `tests` segment
        rule above) and `foo_test.rs` leaf files
    """
    if not path:
        return False
    p = path.lower()
    parts = [seg for seg in p.replace("\\", "/").split("/") if seg]
    if not parts:
        return False
    # `tests`/`test` dir covers python + go + rust conventions;
    # `__tests__` covers the js/ts convention.
    if any(seg in ("test", "tests", "__tests__") for seg in parts):
        return True
    leaf = parts[-1]
    if leaf.startswith("test_") or leaf.startswith("tests_"):
        return True
    # Match `_test.py`, `_tests.py`, `_test.pyi`, etc. but only when the
    # filename ends with the suffix exactly (not "latest_results.py").
    for ext in (".py", ".pyi"):
        if leaf.endswith("_test" + ext) or leaf.endswith("_tests" + ext):
            return True
    # go: `foo_test.go`.
    if leaf.endswith("_test.go"):
        return True
    # rust: `foo_test.rs` leaf (the `tests/` dir case is already covered
    # by the segment check above).
    if leaf.endswith("_test.rs"):
        return True
    # js/ts: `foo.test.js`, `foo.spec.ts`, etc.
    if leaf.endswith(_JS_TEST_LEAF_SUFFIXES):
        return True
    return False


def _canon_test_path(path: str) -> str:
    return path.split("::", 1)[0].lstrip("./").rstrip("/") or path


# Runner-binary tokens across languages (mirrors the union of per-quirk
# `verification_patterns` in language_quirks/*.toml: pytest, go, cargo,
# jest/mocha/vitest, ctest/make).
_INTERPRETER_TOKEN_RE = re.compile(
    r"^python(?:\d+(?:\.\d+)?)?$|^pytest$|"
    r"^go$|^cargo$|^npm$|^yarn$|^pnpm$|^npx$|^make$|"
    r"^jest$|^mocha$|^vitest$|^ctest$"
)

# Runner binaries that take a bare test-ish subcommand word (`go test`,
# `cargo test`, `npm run test`, `make check`, ...) — that subcommand word
# must not itself be mistaken for the test-file target.
_RUNNER_BINARIES = {"go", "cargo", "npm", "yarn", "pnpm", "npx", "make"}
_RUNNER_SUBCOMMAND_TOKENS = {"test", "tests", "check", "build", "vet", "clippy", "run"}


def _extract_test_target(cmd: str) -> str:
    """Pull the test-file/dir argument out of a verification command.

    Skips interpreter-binary tokens like `/usr/bin/python3`,
    `/opt/miniconda3/.../python`, `pytest` — those are runner paths,
    not test targets, and treating them as targets caused
    `same_target_count` to ratchet on the runner rather than on the
    actual test file.

    It also skips go/cargo/npm/jest-family
    runner binaries and the bare subcommand word that follows them
    (`go test`, `cargo test`, `npm run test`) so it isn't mistaken for a
    file target, then recognize go (`foo_test.go`), rust (`tests/...`,
    `foo_test.rs`) and js/ts (`foo.test.js`, `foo.spec.ts`) targets via
    the broadened `_looks_like_test_path`. Some invocations (bare
    `go test ./...`, `cargo test --test name`) have no clean file
    target — returning "" for those is expected, not a bug.
    """
    if not cmd or "test" not in cmd.lower():
        return ""
    try:
        tokens = shlex.split(cmd, posix=True)
    except ValueError:
        return ""
    runner_prefix = False
    for token in tokens:
        candidate = token.split("::", 1)[0].rstrip(",")
        # Strip system-binary directories before the `_looks_like_test_path`
        # check — `/usr/bin/python3` and `/opt/miniconda3/envs/testbed/bin/pytest`
        # would otherwise be considered "test paths" because they contain
        # `python` (no longer matches new _looks_like_test_path) but their
        # trailing `bin/...` directory may still resemble a test path under
        # adversarial inputs. Belt-and-braces check for the two known
        # offenders.
        if candidate.startswith(("/usr/", "/opt/")):
            runner_prefix = False
            continue
        leaf = candidate.rsplit("/", 1)[-1]
        if _INTERPRETER_TOKEN_RE.match(leaf):
            runner_prefix = leaf in _RUNNER_BINARIES
            continue
        if runner_prefix and leaf in _RUNNER_SUBCOMMAND_TOKENS:
            continue
        runner_prefix = False
        if _looks_like_test_path(candidate) and (
            "/" in candidate
            or candidate.startswith("/")
            or candidate.lower().startswith("test")
            or leaf.endswith(_JS_TEST_LEAF_SUFFIXES)
            or leaf.endswith(("_test.go", "_test.rs"))
        ):
            return candidate
    return ""


def _test_target_is_covered(state: GuardrailState, target: str) -> bool:
    if not target:
        return False
    canon = _canon_test_path(target)
    if canon in state.test_file_reads:
        return True
    prefix = canon.rstrip("/")
    return any(read == prefix or read.startswith(prefix + "/") for read in state.test_file_reads)


_ERROR_SIG_RE = re.compile(r"\[exit code:\s*(\d+)\]")


def _error_signature(result: str) -> str:
    """Hash an ERROR result down to a stable signature.

    Two heuristics, in order:
      1. `[exit code: N]` marker (bash) — signature = "exit:N".
      2. First non-empty line of the ERROR body — signature = the line.

    Two error results share a signature when their exit code matches OR
    when their first-line body matches (e.g. both `ImportError: cannot
    import name 'X'`). This catches the "model repeats the same wrong
    fix" pattern: the surrounding turns may differ, but the error
    signature is stable.
    """
    m = _ERROR_SIG_RE.search(result)
    if m:
        return f"exit:{m.group(1)}"
    body = result[len("ERROR:"):] if result.startswith("ERROR:") else result
    for line in body.splitlines():
        line = line.strip()
        if line:
            return line[:120]
    return ""




def _record_mutation_repeat_block(state: GuardrailState, sig: str) -> int:
    if not sig:
        state.mutation_repeat_block_sig = ""
        state.mutation_repeat_block_count = 0
        return 0
    if state.mutation_repeat_block_sig == sig:
        state.mutation_repeat_block_count += 1
    else:
        state.mutation_repeat_block_sig = sig
        state.mutation_repeat_block_count = 1
    return state.mutation_repeat_block_count
