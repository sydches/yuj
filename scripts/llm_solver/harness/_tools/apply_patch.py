"""apply_patch tool: dispatcher around the apply_patch DSL parser/applier."""
from ...config import Config


def apply_patch_tool(patch: str, *, cwd: str, cfg: Config) -> str:
    """Tool dispatcher for the apply_patch DSL.

    Wraps the parser + verifier + applier in harness/apply_patch.py with
    the standard enabled-knob gate and error rendering. Returns the
    success envelope from verify_and_apply OR a single ERROR line on
    parse / verify failure (the model needs the error text to
    reformulate).
    """
    from ..apply_patch import (
        parse_patch as _parse_patch,
        verify_and_apply as _verify_and_apply,
        render_error as _render_apply_patch_error,
        PatchParseError, PatchVerifyError,
    )
    if not getattr(cfg, "tools_apply_patch_enabled", False):
        return _render_apply_patch_error(
            "disabled",
            "apply_patch tool is disabled (tools.apply_patch.enabled=false)",
        )
    try:
        ops = _parse_patch(patch)
    except PatchParseError as e:
        return _render_apply_patch_error("parse", str(e))
    try:
        envelope = _verify_and_apply(ops, cwd)
    except PatchVerifyError as e:
        return _render_apply_patch_error(getattr(e, "kind", "verify"), str(e))
    # F5: record savings on the success path. apply_patch substitutes
    # for N round-trip edit() calls; bucket "apply_patch_vs_edit_loop"
    # captures the input/output shape so a run-level analyzer can
    # estimate the realised lift. Mirrors list_definitions F7.
    n_adds = sum(1 for op in ops if op.kind == "add")
    n_deletes = sum(1 for op in ops if op.kind == "delete")
    n_updates = sum(1 for op in ops if op.kind == "update")
    n_hunks = sum(len(op.hunks) for op in ops if op.kind == "update")
    from ..savings import get_ledger
    get_ledger().record(
        bucket="apply_patch_vs_edit_loop",
        layer="harness",
        mechanism="apply_patch",
        input_chars=len(patch),
        output_chars=len(envelope),
        measure_type="exact",
        ctx={
            "n_ops": len(ops),
            "n_adds": n_adds,
            "n_deletes": n_deletes,
            "n_updates": n_updates,
            "n_hunks": n_hunks,
        },
    )
    # Post-edit checks fire for each touched add or update. Deleted files
    # are skipped because checking a deleted file
    # is undefined). All check outputs are concatenated to the success
    # envelope. Note: unlike write/edit, apply_patch does NOT revert on
    # block — multi-file backup before verify_and_apply is a larger
    # refactor; today the model sees the patch applied + the block
    # message and must issue a corrective patch on the next turn.
    from ..post_edit import run_post_edit_checks
    failed_tail = ""
    for op in ops:
        if op.kind == "delete":
            continue
        res = run_post_edit_checks(op.path, cwd=cwd, cfg=cfg, trigger="apply_patch")
        if res.action != "ok":
            failed_tail += res.output
    if failed_tail:
        envelope = envelope + failed_tail
    return envelope
