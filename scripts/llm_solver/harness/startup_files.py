"""Prepare declared startup resources and read task metadata in its namespace."""
from dataclasses import replace
from contextlib import contextmanager
from functools import wraps
from pathlib import Path, PurePosixPath

from .task_file_runtime import task_file_scope


@contextmanager
def startup_file_scope(cwd, cfg, *, effective_env=None, allow_login_shell=None,
                       ignore_policy=None, unreadable_paths=None):
    """Select the command view for task-owned startup documents."""
    from .task_path import startup_task_path
    view_cfg = cfg if unreadable_paths is None else replace(
        cfg, unreadable_paths=tuple(unreadable_paths))
    with task_file_scope(cwd, view_cfg, environment=effective_env,
                         allow_login_shell=allow_login_shell,
                         ignore_policy=ignore_policy):
        yield startup_task_path(cwd)


def discover_task_skills(cwd, cfg, *, environment=None, allow_login_shell=None,
                         ignore_policy=None, unreadable_paths=None):
    from .skills import discover_skills
    from .sandbox import container_mode
    discovery_cfg = cfg if unreadable_paths is None else replace(cfg, unreadable_paths=tuple(unreadable_paths))
    if cfg.skills_enabled and cfg.sandbox_bash and cfg.sandbox_backend == 'bwrap' and container_mode() is None:
        # The bwrap view has private home storage. Discover configured global
        # resource names from that view's environment, then admit their host
        # sources read-only for startup inspection. Only validated package
        # directories enter the final model-facing config.
        with task_file_scope(cwd, discovery_cfg, environment=environment,
                             allow_login_shell=allow_login_shell,
                             ignore_policy=ignore_policy) as files:
            resources = []
            for value in (*cfg.skills_dirs, *cfg.skill_paths):
                expanded = files.expand_path(value, variables=True)
                if not PurePosixPath(expanded).is_absolute() or PurePosixPath(expanded).is_relative_to(str(cwd)):
                    continue
                candidate = Path(expanded)
                if candidate.is_file():
                    candidate = candidate.parent
                if candidate.is_dir():
                    resources.append(str(candidate.resolve()))
            discovery_cfg = replace(discovery_cfg, skills_readable_dirs=tuple(dict.fromkeys(
                (*cfg.skills_readable_dirs, *resources))))
    with task_file_scope(cwd, discovery_cfg, environment=environment,
                         allow_login_shell=allow_login_shell,
                         ignore_policy=ignore_policy):
        return discover_skills(
            cwd, enabled=cfg.skills_enabled, skills_dirs=cfg.skills_dirs,
            skill_paths=cfg.skill_paths, root_markers=cfg.project_root_markers,
            unreadable_paths=discovery_cfg.unreadable_paths,
        )


def file_scoped_prompt_assembly(function):
    @wraps(function)
    def scoped(cfg, client, work_dir, *args, **kwargs):
        masks = kwargs.get('unreadable_paths')
        view_cfg = cfg if masks is None else replace(cfg, unreadable_paths=tuple(masks))
        with task_file_scope(
            work_dir, view_cfg, environment=kwargs.get('effective_env'),
            allow_login_shell=kwargs.get('allow_login_shell'),
            ignore_policy=kwargs.get('ignore_policy'),
        ):
            return function(cfg, client, work_dir, *args, **kwargs)
    return scoped
