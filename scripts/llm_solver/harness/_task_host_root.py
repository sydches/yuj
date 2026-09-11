"""Retain local task directory identities for native execution."""
import os
import weakref


class HostTaskRoot:
    """Retain the selected directory identity while local executors use it.

    Checking the name detects replacement. Local bwrap launches bind through
    the descriptor so a later name change cannot redirect the task mount.
    """

    def __init__(self, path):
        self.path = path
        descriptor = os.open(path, os.O_PATH | os.O_DIRECTORY | os.O_CLOEXEC)
        self.descriptor = descriptor
        self._release = weakref.finalize(self, os.close, descriptor)
        value = os.fstat(descriptor)
        self.identity = (value.st_dev, value.st_ino)

    def verify(self):
        from .task_environment import TaskEnvironmentUnavailable
        try:
            current = os.stat(self.path)
        except OSError as error:
            raise TaskEnvironmentUnavailable('task root changed or became unavailable') from error
        if (current.st_dev, current.st_ino) != self.identity:
            raise TaskEnvironmentUnavailable('task root changed after execution binding')


def capture_host_task_root(cwd, *, sandbox=True, sandbox_backend='bwrap'):
    """Bind local bwrap's host directory; remote namespaces own their roots."""
    from .sandbox import container_mode
    from .task_path import active_task_files
    if not sandbox or sandbox_backend != 'bwrap' or container_mode() is not None:
        return None
    root = getattr(active_task_files(cwd), '_host_task_root', None)
    if root is None:
        from .sandbox._filesystem import _ACTIVE as filesystem_active
        view = filesystem_active.get()
        if view is not None and view.cwd == os.path.abspath(cwd):
            root = view.task_root
    if root is None:
        from .sandbox._filesystem import _ADMISSION
        admission = _ADMISSION.get()
        if admission is not None and admission.task_root is not None:
            if admission.task_root.path != os.path.abspath(cwd):
                raise RuntimeError('sandbox task root changed during startup')
            root = admission.task_root
        else:
            root = HostTaskRoot(cwd)
            if admission is not None:
                admission.bind_root(root)
    root.verify()
    return root
