"""Post-edit validation hook — declared checks run after successful
``edit`` / ``write`` / ``apply_patch`` tool calls.

Borrowed in spirit from Goose's recipe ``retry.checks`` with
``SuccessCheck::Shell`` (``crates/goose/src/agents/retry.rs``). The
harness provides the mechanism (match trigger + when-predicate, run
shell command, interpret exit code); the *policy* of what to run and
what failure means lives entirely in user-declared config.

Schema per check (see config.toml [post_edit_check]):
    name     — ledger mechanism tag
    trigger  — "edit" | "write" | "apply_patch" | combinations
               separated by "|" (e.g. "edit|write|apply_patch")
    when     — safe-eval predicate over {path, ext}; "" = always
    cmd      — shell command; {path} substituted with shlex-quoted path
    on_fail  — "append" | "warn" | "block"

The harness never supplies a default check — an empty list is a
first-class configuration. Policy is always declared.

`warn` vs `append`: today both surface the failure tail in the next
tool result, identical at the model interface. The distinction is
recorded on the savings ledger only — `warn` events are tagged so
operators can find them, but the model sees the same text either
way. `block` rejects the next tool call with the failure tail as
the rejection text.
"""
from __future__ import annotations

import ast
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .._shared.post_edit_spec import PostEditCheckSpec, parse_post_edit_check_spec
from ..config import Config


# ── Safe-eval for `when` predicates ──────────────────────────────────────

_ALLOWED_NAMES = {"path", "ext"}
_ALLOWED_STR_METHODS = {"startswith", "endswith", "lower", "upper"}


class PredicateError(ValueError):
    """Raised when a `when` predicate uses a disallowed construct.

    Per the no-bullshit policy: no silent failures, no "predicate
    looked funny so we skipped the check." An invalid predicate is a
    config error, surfaced loudly.
    """


def _check_node(node: ast.AST) -> None:
    """Walk an AST and reject anything outside the predicate whitelist."""
    if isinstance(node, ast.Expression):
        _check_node(node.body)
        return
    if isinstance(node, ast.Constant):
        if not isinstance(node.value, (str, int, float, bool, type(None))):
            raise PredicateError(f"constant type {type(node.value).__name__!r}")
        return
    if isinstance(node, ast.Name):
        if node.id not in _ALLOWED_NAMES:
            raise PredicateError(f"name {node.id!r} not in {_ALLOWED_NAMES}")
        return
    if isinstance(node, ast.BoolOp):
        for v in node.values:
            _check_node(v)
        return
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        _check_node(node.operand)
        return
    if isinstance(node, ast.Compare):
        _check_node(node.left)
        for op in node.ops:
            if not isinstance(
                op, (ast.Eq, ast.NotEq, ast.In, ast.NotIn)
            ):
                raise PredicateError(f"comparison op {type(op).__name__!r}")
        for c in node.comparators:
            _check_node(c)
        return
    if isinstance(node, ast.Tuple) or isinstance(node, ast.List):
        for e in node.elts:
            _check_node(e)
        return
    if isinstance(node, ast.Attribute):
        if not isinstance(node.value, ast.Name):
            raise PredicateError("attribute access on non-name")
        if node.value.id not in _ALLOWED_NAMES:
            raise PredicateError(
                f"attribute on {node.value.id!r} not in {_ALLOWED_NAMES}"
            )
        if node.attr not in _ALLOWED_STR_METHODS:
            raise PredicateError(f"method {node.attr!r} not allowed")
        return
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Attribute):
            raise PredicateError("bare function call")
        _check_node(node.func)
        for a in node.args:
            _check_node(a)
        if node.keywords:
            raise PredicateError("keyword arguments not allowed")
        return
    raise PredicateError(f"unsupported node {type(node).__name__}")


def eval_when(expr: str, *, path: str, ext: str) -> bool:
    """Evaluate a `when` predicate under the whitelist.

    Empty expression = True (always-fire). Raises :class:`PredicateError`
    on any disallowed construct — callers surface that as a config
    error, not silently skip the check.
    """
    expr = (expr or "").strip()
    if not expr:
        return True
    tree = ast.parse(expr, mode="eval")
    _check_node(tree)
    # Safe builtins = none. Locals = only path/ext strings.
    return bool(eval(
        compile(tree, "<when>", "eval"),
        {"__builtins__": {}},
        {"path": path, "ext": ext},
    ))


# ── Result type ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PostEditResult:
    """Outcome of the post-edit check chain for one tool call.

    action ∈ {"ok", "append", "warn", "block"}. "ok" means either no
    check fired or every fired check succeeded. "append" / "warn" /
    "block" mirror the on_fail value of the check that produced the
    first non-zero exit (checks after a terminal action are skipped).
    """
    action: str = "ok"
    output: str = ""
    check_name: str = ""


# ── Ledger helper ────────────────────────────────────────────────────────

def _record_event(
    mechanism: str, *, check_name: str, path: str, output_chars: int
) -> None:
    from .savings import get_ledger
    get_ledger().record(
        bucket="post_edit_validation",
        layer="harness",
        mechanism=mechanism,
        input_chars=0,
        output_chars=int(output_chars),
        measure_type="exact",
        ctx={"check": check_name, "path": path},
    )


# ── Entry point ──────────────────────────────────────────────────────────

def run_post_edit_checks(
    path: str, *, cwd: str, cfg: Config | None, trigger: str,
) -> PostEditResult:
    """Walk the declared check list; return the first terminal outcome.

    Evaluation order:
      1. Check trigger matches the caller's trigger ("edit" / "write").
      2. Evaluate `when` predicate against {path, ext}; skip if False.
      3. Run `cmd` inside the bwrap sandbox via tools.bash.
      4. Non-zero exit → return PostEditResult(on_fail, tail_text, name).
         Zero exit → record "ok" event, continue to the next check.

    Trigger string `"edit|write"` matches both triggers. Unknown
    triggers never match.
    """
    if cfg is None or not cfg.post_edit_check_enabled:
        return PostEditResult()
    checks = cfg.post_edit_checks or []
    if not checks:
        return PostEditResult()
    # Single-suffix `ext` is what the `when` predicates document.
    # Pre-fix the multi-suffix `ext = "".join(suffixes)` was computed
    # but unused — `short_ext` was the actual binding. Renamed to the
    # documented name; the multi-suffix form was removed because no
    # check spec used it.
    ext = Path(path).suffix

    from .sandbox.env_policy import active_environment
    from .tools import (
        _bash_unreadable_paths,
        _effective_command_environment,
        bash,
    )
    effective_env, allow_login_shell = active_environment()
    if effective_env is None:
        effective_env, allow_login_shell = _effective_command_environment(cfg)

    for raw_spec in checks:
        spec = parse_post_edit_check_spec(raw_spec)
        if trigger not in spec.triggers:
            continue
        when_result = eval_when(
            spec.when, path=path, ext=ext,
        )
        if not when_result:
            continue
        template: str = spec.cmd
        try:
            cmd = template.format(path=shlex.quote(path))
        except (KeyError, IndexError, ValueError) as e:
            # A malformed cmd template ({foo} where foo is not 'path')
            # previously
            # surfaced as a bare KeyError — name the spec so the
            # operator can find the bad entry.
            raise ValueError(
                f"post_edit_check {spec.name!r} has malformed cmd "
                f"template {template!r}: {e}"
            ) from e
        # Forward sandbox_required and unreadable_paths so strict-mode
        # runs don't silently degrade
        # their post-edit checks to unsandboxed.
        out = bash(
            cmd, cwd=cwd, timeout=cfg.post_edit_check_timeout,
            sandbox=cfg.sandbox_bash, bwrap_bin=cfg.bwrap_bin,
            sandbox_required=getattr(cfg, "sandbox_required", False),
            unreadable_paths=_bash_unreadable_paths(cwd, cfg),
            sandbox_backend=getattr(cfg, "sandbox_backend", "bwrap"),
            container_runtime=getattr(
                cfg, "sandbox_container_runtime", "docker"
            ),
            container_image=getattr(cfg, "sandbox_container_image", ""),
            container_flags=tuple(
                getattr(cfg, "sandbox_container_flags", ()) or ()
            ),
            effective_env=effective_env,
            allow_login_shell=allow_login_shell,
        )
        failed = ("[exit code:" in out) or out.startswith("ERROR")
        if not failed:
            _record_event(
                "ok", check_name=spec.name, path=path, output_chars=0,
            )
            continue
        tail = (
            f"\n\n[post-edit check '{spec.name}' failed for {path}]\n"
            f"{out.strip()}"
        )
        on_fail = spec.on_fail
        _record_event(
            on_fail, check_name=spec.name, path=path,
            output_chars=len(tail),
        )
        return PostEditResult(
            action=on_fail, output=tail, check_name=spec.name,
        )
    return PostEditResult()


# Back-compat shim for callers that used the old string-return contract.
def run_post_edit_check(
    path: str, *, cwd: str, cfg: Config | None, trigger: str = "edit|write",
) -> str:
    """Legacy call site: returns the tail string only.

    Retained so older call sites keep working until fully migrated to
    ``run_post_edit_checks``. New callers should use the dataclass
    result so they can react to block / warn actions.
    """
    res = run_post_edit_checks(path, cwd=cwd, cfg=cfg, trigger=trigger)
    return res.output
