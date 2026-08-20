"""write tool: create or overwrite a file."""
from ...config import Config
from ._common import _resolve


def write(path: str, content: str, *, cwd: str,
          cfg: Config | None = None) -> str:
    """Create or overwrite a file.

    When post-edit validation is enabled and matching checks fire,
    their outcome is applied:
      - on_fail="append" / "warn": tail appended to the OK result
      - on_fail="block": file reverted to prior state; ERROR returned
    """
    # Path with embedded newline or NUL: write would succeed but the
    # OK message line-breaks in the model's view, and any subsequent
    # tool call referencing the same path is fragile. Refuse early.
    if "\n" in path or "\x00" in path:
        return f"ERROR: path contains forbidden character (newline or NUL)"
    try:
        target = _resolve(cwd, path)
    except ValueError as e:
        return f"ERROR: {e}"
    # Directory target: write_text() on a directory raises IsADirectoryError
    # which the outer Exception catch surfaces as the opaque
    # `ERROR: [Errno 21] Is a directory: '<absolute path>'`. The model
    # has no signal that the path needs to change. Refuse early with an
    # actionable message.
    if target.is_dir():
        return (
            f"ERROR: {path} is a directory — choose a different name "
            "or remove the directory first."
        )
    existed_before = target.exists()
    # Snapshot prior content as raw bytes — read_text() would raise
    # UnicodeDecodeError on a binary file (escaping the inner OSError
    # catch and crashing the turn since dispatch only catches
    # KeyError/TypeError). read_bytes() can't decode-fail; write_bytes()
    # on revert is byte-perfect (no CRLF→LF translation loss).
    previous_bytes: bytes | None = None
    if existed_before:
        try:
            previous_bytes = target.read_bytes()
        except OSError:
            previous_bytes = None
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        head = f"OK: wrote {len(content)} bytes to {path}"
        from ..post_edit import run_post_edit_checks
        res = run_post_edit_checks(path, cwd=cwd, cfg=cfg, trigger="write")
        if res.action == "block":
            if previous_bytes is not None:
                target.write_bytes(previous_bytes)
            elif not existed_before:
                try:
                    target.unlink()
                except OSError:
                    pass
            return (
                f"ERROR: write blocked by post-edit check "
                f"'{res.check_name}' for {path}{res.output}"
            )
        return head + res.output
    except Exception as e:
        return f"ERROR: {e}"
