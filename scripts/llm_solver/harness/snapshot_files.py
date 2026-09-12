"""Capture native Git turn snapshots without writing to the task repository."""
import json
import os
import re
import stat
import subprocess
from dataclasses import asdict
from pathlib import Path

from .._shared.telemetry_paths import ensure_telemetry_dir, telemetry_dir
from .task_path import activate_task_files
from .task_files import NamespaceFiles
from .workspace_checkpoints import WorkspaceCheckpointStore, WorkspaceCheckpointError, _safe_relative_path
from . import local_file_access
from .checkpoint_files import checkpoint_entry_path
from .time_budget import execution_deadline, remaining_before

RECORDS_NAME = 'turn_snapshot_records.jsonl'
STORE_NAME = '.turn_snapshot_git'


def legacy_container_snapshot_files(workspace):
    """Bind the standalone legacy API to its inspected container, once."""
    from .task_environment import discover_task_environment, docker_client_fingerprint, TaskEnvironmentUnavailable
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
        paths = []
        for value in result.stdout.split(b'\0'):
            if not value:
                continue
            relative = _safe_relative_path(os.fsdecode(value))
            if self._is_excluded(relative):
                continue
            paths.append(relative)
        modes = self.files.entry_modes(paths)
        return sorted({relative for relative in paths
                       if (mode := modes.get(str(self.files.root / relative))) is not None
                       and (stat.S_ISREG(mode) or stat.S_ISLNK(mode))})


class LocalTurnSnapshotStore(WorkspaceCheckpointStore):
    """Keep local snapshots private and never discover an enclosing repository."""

    def __init__(self, workspace, owned_paths):
        self.workspace = Path(workspace).resolve()
        self.owned_paths = owned_paths
        # A linked worktree's external Git metadata is outside this task root.
        self.task_git = local_file_access.is_dir(self.workspace, self.workspace / '.git')
        object_format = (self.task_git_command(['rev-parse', '--show-object-format']).stdout.decode().strip()
                         if self.task_git else 'sha1')
        super().__init__(self.workspace, shadow_dir=ensure_telemetry_dir(self.workspace) / STORE_NAME,
                         object_format=object_format)
        self.task_head = None
        if self.task_git:
            result = self.task_git_command(['rev-parse', '--verify', 'HEAD'])
            self.task_head = result.stdout.decode().strip()
            if not re.fullmatch(r'[0-9a-f]{40}|[0-9a-f]{64}', self.task_head):
                raise WorkspaceCheckpointError('task Git returned an invalid HEAD')

    def task_git_command(self, args, *, data=None):
        environment = {name: value for name, value in os.environ.items() if not name.startswith('GIT_')}
        environment.update(GIT_CONFIG_NOSYSTEM='1', GIT_CONFIG_GLOBAL=os.devnull,
                           GIT_ATTR_NOSYSTEM='1', GIT_OPTIONAL_LOCKS='0', LC_ALL='C')
        result = subprocess.run(
            ['git', f'--git-dir={self.workspace / ".git"}', f'--work-tree={self.workspace}',
             '-c', f'safe.directory={self.workspace}', '-c', 'core.fsmonitor=false', *args],
            cwd=self.workspace, env=environment, input=data, capture_output=True,
            timeout=remaining_before(execution_deadline()),
        )
        if result.returncode:
            raise WorkspaceCheckpointError('task snapshot Git failed: '
                                           + result.stderr.decode(errors='replace').strip())
        return result

    def import_parent(self):
        self._ensure_initialized()
        if self.task_head and self._git(['cat-file', '-e', self.task_head], check=False).returncode:
            pack = self.task_git_command(['pack-objects', '--stdout', '--revs'],
                                         data=(self.task_head + '\n').encode())
            self._git(['unpack-objects', '-q'], input_bytes=pack.stdout)

    def _current_commit(self):
        return self.task_head if self.task_head else super()._current_commit()

    def _is_excluded(self, path):
        return super()._is_excluded(path) or any(
            path == owned or path.startswith(owned + '/') for owned in self.owned_paths)

    def _candidate_paths(self):
        if not self.task_git:
            return super()._candidate_paths()
        result = self.task_git_command(['ls-files', '-z', '--cached', '--others', '--exclude-standard', '--', '.'])
        paths = []
        for raw in result.stdout.split(b'\0'):
            if not raw:
                continue
            relative = _safe_relative_path(os.fsdecode(raw))
            if self._is_excluded(relative):
                continue
            target = checkpoint_entry_path(self.workspace, self.workspace / relative)
            try:
                mode = local_file_access.stat(self.workspace, target).st_mode
            except FileNotFoundError:
                continue
            if stat.S_ISREG(mode) or stat.S_ISLNK(mode):
                paths.append(relative)
        return sorted(set(paths))


def capture_local_snapshot(workspace, turn, *, owned_paths):
    store = LocalTurnSnapshotStore(workspace, owned_paths)
    store.import_parent()
    checkpoint = store.capture(turn)
    store._git(['update-ref', f'refs/yuj/snapshots/{checkpoint.commit}', checkpoint.commit])
    record = {'turn': int(turn), 'sha': checkpoint.commit, 'storage': 'private_git_v1',
              'task_head': store.task_head, 'task_binding': {'access': 'local',
              'working_directory': str(store.workspace)}, 'artifact_decisions': {}}
    with (ensure_telemetry_dir(workspace) / RECORDS_NAME).open('a') as output:
        output.write(json.dumps(record, sort_keys=True) + '\n')
    return checkpoint.commit


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
