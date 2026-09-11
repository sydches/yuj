"""Startup-fixed runtime mounts for the Linux bwrap filesystem boundary.

Runtime layout knowledge is descriptor data. Free-form task text and later
commands cannot grant host paths. Only components of an observed installed
runtime are admitted; neither a PATH directory nor its parent is a grant.
"""
from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field, replace
import hashlib
import os
from pathlib import Path
import shutil
import stat
import threading
import tomllib
import weakref

from ..._shared.paths import package_data_path

SYSTEM_ROOTS = ("/usr", "/bin", "/sbin", "/lib", "/lib64")
SYSTEM_FILES = (
    "/etc/ld.so.cache", "/etc/ld.so.conf", "/etc/ld.so.conf.d",
    "/etc/alternatives", "/etc/nsswitch.conf", "/etc/passwd", "/etc/group",
    "/etc/localtime", "/etc/timezone", "/etc/ssl/certs",
)
MAX_CANDIDATES = 64


class ReadOnlySource:
    """Pin an admitted file or directory until bwrap mounts it read-only."""

    def __init__(self, path):
        self.path = path
        self.descriptor = os.open(path, os.O_PATH | os.O_CLOEXEC)
        self._release = weakref.finalize(self, os.close, self.descriptor)
        observed = os.fstat(self.descriptor)
        if not (stat.S_ISDIR(observed.st_mode) or stat.S_ISREG(observed.st_mode)):
            raise RuntimeError('sandbox runtime resource is not a file or directory')
        self.identity = (observed.st_dev, observed.st_ino)

    def verify(self):
        try:
            observed = os.stat(self.path)
        except OSError as error:
            raise RuntimeError('sandbox runtime resource changed after startup') from error
        if (observed.st_dev, observed.st_ino) != self.identity:
            raise RuntimeError('sandbox runtime resource changed after startup')


@dataclass(frozen=True)
class Mount:
    source: str
    target: str
    evidence: str
    kind: str = "read_only"
    source_binding: ReadOnlySource | None = field(default=None, init=False, compare=False, repr=False)

    def __post_init__(self):
        if self.kind == 'read_only':
            object.__setattr__(self, 'source_binding', ReadOnlySource(self.source))


@dataclass(frozen=True)
class FilesystemView:
    cwd: str
    home: str
    mounts: tuple[Mount, ...]
    unresolved: tuple[str, ...]
    descriptor_sha256: str
    runtime_bindings: tuple[tuple[str, str], ...] = ()
    native_selections: tuple[dict, ...] = ()
    private_directories: tuple[str, ...] = ()
    task_root: object | None = field(default=None, compare=False, repr=False)
    host_alias: str = ''

    def record(self):
        storage = "private"
        if not self.home:
            storage = "unavailable"
        elif Path(self.home).is_relative_to(self.cwd):
            storage = "task"
        elif self.home == "/" or any(Path(self.home).is_relative_to(p) for p in SYSTEM_ROOTS):
            storage = "read_only_runtime"
        return {
            "policy": "linux-runtime-components-v1",
            "task_root": self.cwd, "home": self.home, "home_storage": storage,
            "mounts": [{name: getattr(m, name) for name in ('source', 'target', 'evidence', 'kind')}
                       for m in self.mounts],
            "unresolved": list(self.unresolved),
            "descriptor_sha256": self.descriptor_sha256,
            "runtime_bindings": dict(self.runtime_bindings),
            "native_selections": list(self.native_selections),
            "private_directories": list(self.private_directories),
        }


class FilesystemAdmission:
    """One task's startup result, shared with executors created before it."""

    def __init__(self):
        self.view = None
        self.task_root = None
        self._lock = threading.RLock()

    def bind_root(self, root):
        with self._lock:
            if self.task_root is not None:
                self.task_root.verify()
                if root.path != self.task_root.path or root.identity != self.task_root.identity:
                    raise RuntimeError('sandbox task root changed during startup')
            else:
                self.task_root = root

    def check_pending(self):
        with self._lock:
            if self.view is not None:
                raise RuntimeError('sandbox filesystem view is already admitted for this task')

    def admit(self, view):
        with self._lock:
            self.check_pending()
            if view.task_root is not None:
                self.bind_root(view.task_root)
            self.view = view

    def resolve(self):
        with self._lock:
            if self.view is None and _ADMISSION.get() is not self:
                raise RuntimeError('startup filesystem view was not admitted for this executor')
            return self.view


_ADMISSION: ContextVar[FilesystemAdmission | None] = ContextVar('filesystem_admission', default=None)
_ACTIVE: ContextVar[FilesystemView | None] = ContextVar("sandbox_filesystem", default=None)
_SCOPED: ContextVar[bool] = ContextVar("sandbox_filesystem_scoped", default=False)


def discover_filesystem_view(cwd, environment, readable_paths=(), unreadable_paths=()):
    """Inspect paths and bounded metadata without executing host programs."""
    task = Path(cwd).resolve()
    if task == Path("/"):
        raise RuntimeError("sandbox task root cannot be the host root")
    from ..prompt_imports import _UnreadableMatcher
    blocked = _UnreadableMatcher(task, unreadable_paths)
    home_value = environment.get("HOME", "")
    home = Path(home_value) if home_value.startswith("/") else None
    raw = package_data_path(
        __package__.rsplit(".harness", 1)[0] + ".language_quirks", "runtime.toml",
    ).read_bytes()
    descriptor = tomllib.loads(raw.decode())
    layouts = descriptor["filesystem"]["layouts"]
    mounts = {}
    unresolved = set()

    def inside(path, root):
        return path == root or root in path.parents

    def covered(path):
        return inside(path, task) or any(inside(path, Path(p)) for p in SYSTEM_ROOTS)

    def mount(path, evidence):
        path = Path(path)
        if blocked.blocks(path) or not path.exists() or inside(path, task):
            return
        resolved = path.resolve(strict=True)
        if not (resolved.is_dir() or resolved.is_file()):
            return  # Never admit a socket or device as a runtime resource.
        kind = "task_alias" if inside(resolved, task) else "read_only"
        mounts[str(path)] = Mount(str(resolved), str(path), evidence, kind)

    for name in (*SYSTEM_ROOTS, *SYSTEM_FILES):
        mount(name, "linux_runtime")

    seen = set()

    def inspect_base_metadata(prefix, layout):
        if not layout.get("base_directories_file"):
            return
        metadata = prefix / layout["base_directories_file"]
        if blocked.blocks(metadata) or not inside(metadata.resolve(), prefix.resolve()):
            return
        try:
            with metadata.open("rb") as stream:
                raw = stream.read(65537)
            if len(raw) > 65536:
                unresolved.add("runtime_metadata_limit")
                return
            for line in raw.decode("utf-8").splitlines():
                key, separator, value = line.partition("=")
                if separator and key.strip() in layout["base_directories_keys"]:
                    directory = Path(value.strip())
                    if directory.is_absolute():
                        inspect_prefix(directory.parent, f"metadata={metadata}:{key.strip()}")
        except (OSError, UnicodeError):
            unresolved.add("runtime_metadata_unavailable")

    def inspect_prefix(prefix, evidence):
        prefix = Path(prefix).absolute()
        resolved_prefix = prefix.resolve()
        if prefix in seen or covered(prefix) or blocked.blocks(prefix):
            return
        if len(seen) >= MAX_CANDIDATES:
            unresolved.add("runtime_candidate_limit")
            return
        seen.add(prefix)
        # A fabricated marker cannot turn a host root, home or task ancestor
        # into a runtime grant. Components must remain within this prefix.
        if (len(resolved_prefix.parts) < 3 or (home and resolved_prefix == home.resolve())
                or resolved_prefix in task.parents):
            return
        for layout in layouts:
            markers = [prefix / p for p in layout["markers"]]
            if not all(not blocked.blocks(p) and p.exists() and (
                inside(p.resolve(), resolved_prefix)
                or str(p.relative_to(prefix)) in layout.get("base_executables", [])
            ) for p in markers):
                continue
            label = f"{evidence}; layout={layout['name']}"
            for component in layout["components"]:
                path = prefix / component
                if path.exists() and inside(path.resolve(), resolved_prefix):
                    mount(path, label)
            inspect_base_metadata(prefix, layout)
            for inventory in layout.get("inventories", []):
                if "{home}" in inventory and home is None:
                    continue
                directory = Path(inventory.format(prefix=prefix, home=home))
                if blocked.blocks(directory):
                    continue
                try:
                    with os.scandir(directory) as entries:
                        for i, entry in enumerate(entries):
                            if i >= MAX_CANDIDATES:
                                unresolved.add("runtime_inventory_limit")
                                break
                            if entry.is_dir(follow_symlinks=False):
                                inspect_prefix(Path(entry.path), f"inventory={directory}")
                except FileNotFoundError:
                    pass
                except OSError:
                    unresolved.add("runtime_inventory_unavailable")
            for registry in layout.get("registries", []):
                if "{home}" in registry and home is None:
                    continue
                path = Path(registry.format(prefix=prefix, home=home))
                if blocked.blocks(path):
                    continue
                try:
                    with path.open() as stream:
                        for i in range(MAX_CANDIDATES + 1):
                            line = stream.readline(4097)
                            if not line:
                                break
                            if i == MAX_CANDIDATES or len(line) > 4096:
                                unresolved.add("runtime_registry_limit")
                                break
                            candidate = Path(line.strip())
                            if candidate.is_absolute():
                                inspect_prefix(candidate, f"registry={path}")
                    mount(path, f"runtime_registry:{layout['name']}")
                except FileNotFoundError:
                    pass
                except (OSError, UnicodeError):
                    unresolved.add("runtime_registry_unavailable")
            return

    commands = descriptor["discovery"]["commands"]
    path_entries = environment.get("PATH", os.defpath).split(os.pathsep)
    if len(path_entries) > MAX_CANDIDATES:
        unresolved.add("startup_PATH_limit")
    search_path = os.pathsep.join(str(task / p) for p in path_entries[:MAX_CANDIDATES])
    for command in commands:
        executable = shutil.which(command, path=search_path)
        if not executable:
            continue
        path = Path(executable).absolute()
        if inside(path, task) or blocked.blocks(path):
            continue
        resolved = path.resolve()
        evidence = f"startup_PATH:{command}"
        # Exact executables retain standalone installed tools without exposing
        # every file beside them. Their runtime dependencies use layout rules.
        if not covered(path):
            mount(path, evidence)
        if not covered(resolved):
            mount(resolved, evidence)
        # Directory mounts retain symlinks. Admit their intermediate targets
        # too, so a venv link through an external launcher keeps resolving.
        link = path
        visited_links = set()
        while link.is_symlink() and link not in visited_links:
            visited_links.add(link)
            if not covered(link):
                mounts[str(link)] = Mount(str(resolved), str(link), evidence, "runtime_alias")
            target = link.readlink()
            link = Path(os.path.abspath(target if target.is_absolute() else link.parent / target))
            if blocked.blocks(link):
                break
            if not covered(link):
                mount(link, evidence)
        for candidate in (path.parent.parent, resolved.parent.parent):
            inspect_prefix(candidate, evidence)
        try:
            with resolved.open("rb") as stream:
                header = stream.readline(4096)
            if header.startswith(b"#!/"):
                interpreter = Path(os.fsdecode(header[2:].split()[0]))
                inspect_prefix(interpreter.parent.parent, f"shebang={path}")
        except OSError:
            pass
        if not covered(resolved) and not any(
            m.evidence.endswith(tuple(f"layout={x['name']}" for x in layouts))
            and inside(resolved, Path(m.target)) for m in mounts.values()
        ):
            unresolved.add(f"unverified_dependencies:{command}")

    # A task-local environment may name or link to a base installation.
    # Only registered metadata and executable links enter layout validation.
    try:
        with os.scandir(task) as entries:
            for i, entry in enumerate(entries):
                if i >= MAX_CANDIDATES:
                    unresolved.add("task_environment_scan_limit")
                    break
                if blocked.blocks(Path(entry.path)) or not entry.is_dir(follow_symlinks=False):
                    continue
                prefix = Path(entry.path)
                for layout in layouts:
                    if all(not blocked.blocks(prefix / p) and (prefix / p).is_file()
                           for p in layout["markers"]):
                        inspect_base_metadata(prefix, layout)
                        for name in layout.get("base_executables", []):
                            executable = prefix / name
                            if executable.is_symlink() and executable.exists():
                                inspect_prefix(executable.resolve().parent.parent,
                                               f"task_runtime_link={executable}")
    except OSError:
        unresolved.add("task_environment_scan_unavailable")

    for name in readable_paths:
        from ..task_path import captured_host_path
        path = Path(captured_host_path(cwd, name)).resolve(strict=True)
        if not path.is_dir():
            raise RuntimeError(f"sandbox readable path is not a directory: {path}")
        if path in task.parents:
            raise RuntimeError("sandbox readable resource cannot be a task ancestor")
        mount(path, "declared_readable_resource")
    directories = {Path(m.target) for m in mounts.values() if Path(m.source).is_dir()}
    # A directory bind already preserves its executable symlinks. Binding a
    # child file again can follow that symlink before its target exists in the
    # new namespace, and is unnecessary once the component is admitted.
    selected = tuple(m for m in mounts.values()
                     if not any(p in directories for p in Path(m.target).parents))
    return FilesystemView(str(task), str(home) if home else "", selected,
                          tuple(sorted(unresolved)), hashlib.sha256(raw).hexdigest())


def filesystem_view(cwd, environment, readable_paths=(), unreadable_paths=()):
    view = _ACTIVE.get()
    if view is not None:
        if (os.path.abspath(cwd) not in (view.cwd, view.host_alias)
                and str(Path(cwd).resolve()) != view.cwd):
            raise RuntimeError("sandbox task root changed after startup")
        if view.task_root is not None:
            view.task_root.verify()
        return view
    return discover_filesystem_view(cwd, environment, readable_paths, unreadable_paths)


def capture_frozen_filesystem_view(task_root):
    """Retain the local startup view or its task-owned pending admission."""
    if task_root is None:
        return None
    from ..task_path import active_task_files
    files = active_task_files(task_root.path)
    retained = getattr(files, '_filesystem_view', None)
    if retained is not None:
        task_root.verify()
        return retained
    view = _ACTIVE.get()
    if view is None:
        admission = _ADMISSION.get()
        if admission is not None:
            admission.bind_root(task_root)
        return admission
    if task_root.path != view.cwd:
        raise RuntimeError('sandbox task root differs from the frozen filesystem view')
    task_root.verify()
    return view


def resolve_filesystem_view(view):
    return view.resolve() if isinstance(view, FilesystemAdmission) else view


def freeze_filesystem_view(cwd, environment, readable_paths=(), unreadable_paths=(),
                           *, deadline=None, bwrap_bin="bwrap"):
    admission = _ADMISSION.get() if _SCOPED.get() else None
    if admission is not None:
        admission.check_pending()
    from ..task_path import capture_task_execution_paths, capture_host_task_root
    host_alias = os.path.abspath(cwd)
    cwd, unreadable_paths, readable_paths = capture_task_execution_paths(
        cwd, unreadable_paths, readable_paths)
    task_root = capture_host_task_root(cwd)
    view = replace(discover_filesystem_view(cwd, environment, readable_paths, unreadable_paths),
                   task_root=task_root, host_alias=host_alias)
    if deadline is not None:
        from ._toolchain_selection import select_native_toolchain
        view = select_native_toolchain(view, environment, unreadable_paths,
                                       deadline=deadline, bwrap_bin=bwrap_bin)
    if task_root is not None:
        task_root.verify()
    if _SCOPED.get():
        if admission is not None:
            admission.admit(view)
        _ACTIVE.set(view)
    return view


class BoundTaskArgv(list):
    """Keep mount source descriptors alive until their bwrap launch."""

    def __init__(self, values, task_root, resources=(), descriptor_arguments=()):
        super().__init__(values)
        self.task_root = task_root
        self.resources = tuple(resources)
        self.descriptor_arguments = tuple(descriptor_arguments)
        sources = ((*self.resources, task_root) if task_root is not None else self.resources)
        self.pass_fds = tuple(dict.fromkeys(source.descriptor for source in sources))


def build_filesystem_argv(view, *, task_writable=True, task_root=None):
    # bwrap supplies an empty tmpfs root when no root bind is specified.
    argv = ["--tmpfs", "/tmp"]
    resources = []
    descriptor_arguments = []
    if view.home not in {"", "/", "/tmp"} and not any(
        Path(view.home).is_relative_to(Path(p)) for p in SYSTEM_ROOTS
    ):
        argv += ["--tmpfs", view.home]
    for directory in view.private_directories:
        argv += ["--tmpfs", directory]
    for mount in view.mounts:
        if mount.kind == "task_alias":
            # Never hand bwrap a mutable task path as another host bind source.
            # This target resolves against the namespace's existing task view,
            # even if task code changes the file or link while bwrap starts.
            argv += ["--symlink", mount.source, mount.target]
            continue
        if str(Path(mount.target).resolve(strict=True)) != mount.source:
            raise RuntimeError("sandbox runtime resource changed after startup")
        if mount.kind == "runtime_alias":
            argv += ["--symlink", mount.source, mount.target]
            continue
        mount.source_binding.verify()
        resources.append(mount.source_binding)
        descriptor_arguments.append(len(argv) + 1)
        argv += ["--ro-bind-fd", str(mount.source_binding.descriptor), mount.target]
    if task_root is None:
        task_source = view.cwd
        bind = '--bind' if task_writable else '--ro-bind'
    else:
        if task_root.path != view.cwd:
            raise RuntimeError('sandbox task root differs from the bound directory')
        task_root.verify()
        task_source = str(task_root.descriptor)
        bind = '--bind-fd' if task_writable else '--ro-bind-fd'
        descriptor_arguments.append(len(argv) + 1)
    argv += [bind, task_source, view.cwd,
             "--proc", "/proc", "--dev", "/dev"]
    return BoundTaskArgv(argv, task_root, resources, descriptor_arguments)
