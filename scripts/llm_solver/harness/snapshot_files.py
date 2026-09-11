"""Capture native Git turn snapshots without writing to the task repository."""
import json
import os
import re
import stat
from dataclasses import asdict
from pathlib import Path

from .._shared.telemetry_paths import ensure_telemetry_dir, telemetry_dir
from .task_path import TaskPath, activate_task_files
from .task_files import NamespaceFiles
from .workspace_checkpoints import WorkspaceCheckpointStore, WorkspaceCheckpointError, _safe_relative_path

RECORDS_NAME = 'turn_snapshot_records.jsonl'
STORE_NAME = '.turn_snapshot_git'


def legacy_container_snapshot_files(workspace):
    """Bind the standalone legacy API to its inspected container, once."""
    from .task_environment import discover_task_environment, docker_client_fingerprint, TaskEnvironmentUnavailable, verify_docker_identity
    from ._tools._run_in_sandbox import _execute
    from .time_budget import execution_deadline, remaining_before
    from .process_identity import guarded_process_argv
    task = discover_task_environment(workspace)
    if not task.container_id:
        raise TaskEnvironmentUnavailable('snapshot target has no inspected container identity')
    selector = os.environ.get('YUJ_CONTAINER', '')
    client = docker_client_fingerprint()

    def run(script, args, data):
        if os.environ.get('YUJ_CONTAINER', '') != selector or docker_client_fingerprint() != client:
            raise TaskEnvironmentUnavailable('snapshot container selection changed after binding')
        verify_docker_identity(task)
        command = guarded_process_argv(
            ['docker', 'exec', '-i', '--workdir', task.working_directory, task.container_id],
            ['bash', '--noprofile', '--norc', '-c', script, 'yuj-snapshot', *args], task.process_identity)
        return _execute(command, input_bytes=data, binary=True,
                        timeout=remaining_before(execution_deadline()))

    return NamespaceFiles(task.working_directory, run, binding=asdict(task))


class NativeTurnSnapshotStore(WorkspaceCheckpointStore):
    def __init__(self, workspace, files, owned_paths):
        self.files = files
        self.owned_paths = owned_paths
        self.artifact_decisions = {}
        object_format = self.native_git(['rev-parse', '--show-object-format']).stdout.decode().strip()
        super().__init__(workspace, shadow_dir=ensure_telemetry_dir(workspace) / STORE_NAME,
                         object_format=object_format)
        self.task_head = self.native_git(['rev-parse', '--verify', 'HEAD']).stdout.decode().strip()
        if not re.fullmatch(r'[0-9a-f]{40}|[0-9a-f]{64}', self.task_head):
            raise WorkspaceCheckpointError('native Git returned an invalid HEAD')

    def native_git(self, args, *, data=None):
        result = self.files.run('exec "$@"', [self.files._utility('git'),
            '-c', f'safe.directory={self.files.root}', '-C', str(self.files.root), *args], data)
        if result.returncode:
            raise WorkspaceCheckpointError('native snapshot Git failed: '
                + result.stderr.decode(errors='replace').strip())
        return result

    def import_parent(self):
        self._ensure_initialized()
        if self._git(['cat-file', '-e', self.task_head], check=False).returncode:
            pack = self.native_git(['pack-objects', '--stdout', '--revs'],
                                   data=(self.task_head + '\n').encode()).stdout
            self._git(['unpack-objects', '-q'], input_bytes=pack)

    def _current_commit(self):
        return self.task_head

    def _is_excluded(self, path):
        if super()._is_excluded(path):
            return True
        if not any(path == owned or path.startswith(owned + '/') for owned in self.owned_paths):
            return False
        observation = self.files.observe_host_entry(path, self.workspace / path)
        excluded = observation['relation'] == 'same_entry'
        self.artifact_decisions[path] = {**observation, 'excluded': excluded}
        return excluded

    def _candidate_paths(self):
        result = self.native_git(['ls-files', '-z', '--cached', '--others', '--exclude-standard', '--', '.'])
        root = TaskPath(self.files, self.files.root)
        paths = []
        for value in result.stdout.split(b'\0'):
            if not value:
                continue
            relative = _safe_relative_path(os.fsdecode(value))
            if self._is_excluded(relative):
                continue
            try:
                mode = (root / relative).lstat().st_mode
            except FileNotFoundError:
                continue
            if stat.S_ISREG(mode) or stat.S_ISLNK(mode):
                paths.append(relative)
        return sorted(set(paths))


def capture_native_snapshot(workspace, turn, *, files, owned_paths):
    store = NativeTurnSnapshotStore(workspace, files, owned_paths)
    store.import_parent()
    with activate_task_files(files, host_root=workspace):
        checkpoint = store.capture(turn)
    store._git(['update-ref', f'refs/yuj/snapshots/{checkpoint.commit}', checkpoint.commit])
    record = {'turn': int(turn), 'sha': checkpoint.commit, 'storage': 'private_git_v1',
              'task_head': store.task_head, 'task_binding': files.binding,
              'artifact_decisions': store.artifact_decisions}
    with (ensure_telemetry_dir(workspace) / RECORDS_NAME).open('a') as output:
        output.write(json.dumps(record, sort_keys=True) + '\n')
    return checkpoint.commit


def snapshot_object_store(workspace, sha):
    """Locate a recorded snapshot's Git objects, including older local snapshots."""
    records = telemetry_dir(Path(workspace)) / RECORDS_NAME
    if records.is_file():
        for line in records.read_text().splitlines():
            record = json.loads(line)
            if record.get('sha') == sha and record.get('storage') == 'private_git_v1':
                return telemetry_dir(Path(workspace)) / STORE_NAME
    return Path(workspace)
