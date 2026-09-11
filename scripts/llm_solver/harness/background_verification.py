"""Bind a background check to observed mutations without changing its command."""
import hashlib
import json
from pathlib import Path

from ._guardrails.verification import _file_revision
from .shell_verification import shell_verification_status


class BackgroundVerification:
    def __init__(self, state, cwd):
        self.state = state
        self.cwd = Path(cwd)
        self.starts = {}

    def start(self, manager, command):
        from .process_manager import ProcessManager, ReplayProcessManager
        if isinstance(manager, ReplayProcessManager):
            # Replay's recorded poll decision owns historical truth. Do not
            # inspect today's task files to reconstruct that decision.
            return manager.start(command).result
        if isinstance(manager, ProcessManager):
            manager.poll_metadata = self.poll_metadata
        state = self.state()
        revisions = {path: _file_revision(self.cwd, path)
                     for path in state.verification_file_revisions}
        snapshot = (command, state.mutation_count, state.has_mutated, revisions)
        started = manager.start(command)
        self.starts[started.proc_id] = snapshot
        return started.result

    def poll_metadata(self, proc_id, exit_code):
        snapshot = self.starts.get(proc_id)
        if snapshot is None:
            return {"verification_status": "unknown_process"}
        command, generation, after_mutation, revisions = snapshot
        state = self.state()
        matches = None if exit_code is None else (after_mutation and state.mutation_count == generation and
                   all(digest and _file_revision(self.cwd, path) == digest
                       for path, digest in revisions.items()))
        evidence = {
            "proc_id": proc_id,
            "command_sha256": hashlib.sha256(command.encode()).hexdigest(),
            "start_mutation_count": generation,
            "observed_mutation_count": state.mutation_count,
            "known_source_count": len(revisions),
            "source_revisions_sha256": hashlib.sha256(json.dumps(revisions, sort_keys=True).encode()).hexdigest(),
            "started_after_mutation": after_mutation,
            "revision_matches": matches,
        }
        status = ("process_running" if exit_code is None else "stale_revision" if not matches
                  else shell_verification_status(command, exit_code))
        return {"verification_status": status, "verification_evidence": evidence}
