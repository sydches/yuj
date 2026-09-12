"""write tool: create or overwrite a file."""
import stat

from ...config import Config
from .. import local_file_access as file_access
from ._common import _is_external_readonly_path, _resolve, _skill_readable_roots


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
    if cfg is not None and _is_external_readonly_path(
        cwd,
        path,
        readonly_roots=_skill_readable_roots(cfg),
    ):
        return f"ERROR: skill path is read-only: {path}"
    try:
        target = _resolve(cwd, path)
    except ValueError as e:
        return f"ERROR: {e}"
    # Directory target: write_text() on a directory raises IsADirectoryError
    # which the outer Exception catch surfaces as the opaque
    # `ERROR: [Errno 21] Is a directory: '<absolute path>'`. The model
    # has no signal that the path needs to change. Refuse early with an
    # actionable message.
    try:
        prior = file_access.stat(cwd, target)
    except FileNotFoundError:
        prior = None
    except OSError as exc:
        return f"ERROR: {exc}"
    if prior is not None and stat.S_ISDIR(prior.st_mode):
        return (
            f"ERROR: {path} is a directory — choose a different name "
            "or remove the directory first."
        )
    existed_before = prior is not None
    # Snapshot prior content as raw bytes — read_text() would raise
    # UnicodeDecodeError on a binary file (escaping the inner OSError
    # catch and crashing the turn since dispatch only catches
    # KeyError/TypeError). read_bytes() can't decode-fail; write_bytes()
    # on revert is byte-perfect (no CRLF→LF translation loss).
    previous_bytes: bytes | None = None
    if existed_before:
        try:
            previous_bytes = file_access.read_bytes(cwd, target)
        except OSError:
            previous_bytes = None
    try:
        file_access.write_text(cwd, target, content, create_parents=True)
        head = f"OK: wrote {len(content)} bytes to {path}"
        from ..post_edit import run_post_edit_actions
        res = run_post_edit_actions(path, cwd=cwd, cfg=cfg, trigger="write")
        if res.action == "block":
            if previous_bytes is not None:
                file_access.write_bytes(cwd, target, previous_bytes)
            elif not existed_before:
                try:
                    file_access.unlink(cwd, target)
                except OSError:
                    pass
            return (
                f"ERROR: write blocked by post-edit check "
                f"'{res.check_name}' for {path}{res.output}"
            )
        return head + res.output
    except Exception as e:
        return f"ERROR: {e}"
