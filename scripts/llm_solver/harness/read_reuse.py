"""Session-local reuse of consecutive, unchanged source inspections."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import hashlib
import json
from pathlib import PurePosixPath
import shlex

_ACTIVE = ContextVar('completed_read_reuse', default=None)


def source_search(command):
    """Recognize literal grep plus optional cd and head/tail, without effects."""
    from .command_redirect import split_shell_fragments, strip_leading_assignments
    from .shell_verification import _literal_simple_command
    from ..bash_quirks.grep_annotations import source_grep
    fragments = split_shell_fragments(command)
    for fragment in fragments:
        if (not _literal_simple_command(fragment.text)
                or strip_leading_assignments(fragment.text).strip() != fragment.text.strip()):
            return None
        if fragment.operator_before not in ('', '&&', '|') or fragment.operator_after not in ('', '&&', '|'):
            return None
        words = shlex.split(fragment.text)
        if not words or words[0] not in {'cd', 'grep', 'egrep', 'fgrep', 'rg', 'head', 'tail'}:
            return None
        # Pattern files, preprocessors and unknown short options add inputs
        # outside the observed search scope. Refuse them, including -ifFILE.
        for word in words[1:]:
            if word.startswith('-') and not word.startswith('--'):
                flags = word[1:]
                if any(char not in 'rRinNhHwvxFEIasm e0123456789+-' for char in flags):
                    return None
        # In particular, rg --pre can execute an arbitrary program.
        allowed = {'--include', '--exclude', '--exclude-dir', '--glob', '--iglob',
                   '--color', '--line-number', '--no-line-number', '--with-filename',
                   '--no-filename', '--recursive', '--ignore-case', '--invert-match',
                   '--word-regexp', '--line-regexp', '--extended-regexp', '--fixed-strings',
                   '--regexp', '--max-count', '--lines'}
        if any(word.startswith('--') and word != '--' and word.split('=', 1)[0] not in allowed
               for word in words[1:]):
            return None
    return source_grep(fragments, strip_leading_assignments)


def active_reuse():
    return _ACTIVE.get()


def retain_visible_reference(session, messages):
    """Do not point at an answer that context projection has removed."""
    previous = getattr(session, '_completed_read_reuse', None)
    if not previous:
        return
    def texts(value):
        if isinstance(value, str):
            yield value
        elif isinstance(value, list):
            for item in value:
                yield from texts(item)
        elif isinstance(value, dict):
            for key in ('content', 'text'):
                yield from texts(value.get(key))
    if not any(previous['output'] in text for text in texts(messages)):
        session._completed_read_reuse = None


def contained_search_paths(paths, roots, excluded):
    for value in paths:
        path = PurePosixPath(value)
        if path.is_absolute():
            for root in roots:
                try:
                    path = path.relative_to(root)
                    break
                except ValueError:
                    pass
            else:
                return False
        if not path.parts or '..' in path.parts:
            return False
        if any(path == PurePosixPath(item) or PurePosixPath(item) in path.parents for item in excluded):
            return False
    return True


def inventory_stamp(entries, binding):
    # Refuse links/special files: their bytes or effects can lie outside this
    # observed task tree. Include metadata, not source contents or host reads.
    if any(entry[0] not in (b'f', b'd') for entry in entries.values()):
        return None
    return hashlib.sha256(repr((binding, sorted(entries.items()))).encode()).hexdigest()


@dataclass
class ReuseCall:
    session: object
    name: str
    arguments: dict
    turn: int
    signature: str
    previous: dict | None
    before: str | None = None
    after: str | None = None
    reused: bool = False

    def lookup(self, cwd, cfg):
        previous = self.previous
        # Let the first turn after a quiet period deliver its warning before
        # reuse removes the completed observation from the next guard check.
        if self.turn <= getattr(cfg, 'guardrails_arm_after_turn', 0) + (cfg.duplicate_warn_count > 0):
            return None
        if not previous or previous['count'] < max(2, cfg.duplicate_warn_count):
            return None
        if self.name == 'read':
            from ._guardrails.verification import _file_revision
            current = _file_revision(cwd, str(self.arguments.get('path', '')))
        else:
            current = self.before
        if not current or current != previous['inputs']:
            self.previous = None
            return None
        from ._tools._common import ToolExecutionText
        self.reused = True
        return ToolExecutionText(
            f"[harness: unchanged since turn {previous['origin']}; reused the same selected output. "
            "Change the query or range to inspect something else.]",
            executed=False, exit_status=0, verification_status='not_a_check',
        )

    def finish(self, result, facts):
        if self.reused:
            if self.name != 'read' and self.after != self.before:
                self.session._completed_read_reuse = None
                return
            self.session._completed_read_reuse = {**self.previous, 'turn': self.turn}
            return
        if (not facts.get('executed') or facts.get('security_blocked_stage')
                or facts.get('timed_out') or str(result).startswith('ERROR:')):
            self.session._completed_read_reuse = None
            return
        if self.name == 'read':
            inputs = (facts.get('inspection_evidence') or {}).get('sha256')
        else:
            inputs = self.before if self.before == self.after else None
            if (facts.get('file_changes') or {}).get('status') != 'unchanged_metadata':
                inputs = None
        if not inputs:
            self.session._completed_read_reuse = None
            return
        digest = hashlib.sha256(str(result).encode()).hexdigest()
        same = self.previous and self.previous['inputs'] == inputs and self.previous['digest'] == digest
        self.session._completed_read_reuse = dict(
            signature=self.signature, turn=self.turn, inputs=inputs, digest=digest,
            output=str(result),
            count=self.previous['count'] + 1 if same else 1,
            origin=self.previous['origin'] if same else self.turn,
        )


@contextmanager
def reuse_scope(session, tc, turn, *, allow_reuse=True):
    cfg = getattr(session, 'cfg', None)
    if cfg is None:
        yield None
        return
    previous = getattr(session, '_completed_read_reuse', None)
    enabled = (allow_reuse and cfg.duplicate_guard_enabled
               and not getattr(getattr(session, '_plan_mode', None), 'active', False)
               and not getattr(getattr(session, 'client', None), 'is_replay', False))
    manager = getattr(session, '_process_manager', None)
    if manager is not None:
        try:
            enabled &= manager.has_pending_observations() is False
        except Exception:
            enabled = False
    if not enabled:
        session._completed_read_reuse = None
        yield None
        return
    eligible = tc.name in {'read', 'grep'}
    if tc.name == 'grep':
        parts = PurePosixPath(str(tc.arguments.get('path', '.'))).parts
        eligible &= not any(part in {'..', '.git', '.solver', '.tool_output'} for part in parts)
    if tc.name == 'bash' and not tc.arguments.get('background'):
        search = source_search(str(tc.arguments.get('cmd', '')))
        # Only explicit task subtrees. Shell searches of Git administration,
        # parent directories or other resources need their own observations.
        eligible = bool(search and search.paths and all(
            path not in ('.', '/') and '..' not in PurePosixPath(path).parts
            and not any(part in {'.git', '.solver', '.tool_output'} for part in PurePosixPath(path).parts)
            and not PurePosixPath(path).is_absolute()
            for path in search.paths))
        environment = getattr(session, '_effective_env', None) or {}
        eligible &= not any(key in {'BASH_ENV', 'RIPGREP_CONFIG_PATH', 'GREP_OPTIONS'}
                            or key.startswith('BASH_FUNC_') for key in environment)
        eligible &= not getattr(session, '_allow_login_shell', False)
    signature = json.dumps([tc.name, tc.arguments, id(cfg), str(session.cwd),
                            dict(getattr(session, '_effective_env', None) or {})], sort_keys=True)
    if not previous or previous['signature'] != signature or previous['turn'] != turn - 1:
        previous = None
    call = ReuseCall(session, tc.name, tc.arguments, turn, signature, previous) if enabled and eligible else None
    if call is None:
        session._completed_read_reuse = None
    token = _ACTIVE.set(call)
    try:
        yield call
    finally:
        _ACTIVE.reset(token)
