---
layout: default
title: Sandbox
nav_order: 8
---

# Sandbox

A sandbox limits what a model shell command can reach. Yuj uses Linux
`bubblewrap` (`bwrap`) by default and can instead start a short-lived local
Docker or Podman container for each command.

You do not need Docker for normal use.

Activate the environment where Yuj is installed before you run a command on
this page. Otherwise, replace `yuj` with that environment's `bin/yuj` path.

## Choose a mode

| Your situation | What to do |
| --- | --- |
| You run Yuj on Linux | Keep `[sandbox].backend = "bwrap"` and leave `YUJ_CONTAINER` unset. |
| You want a first-class Docker/Podman boundary | Select `backend = "container"` and an already-local trusted image as shown below. |
| You use Windows | Run Yuj in WSL2. Install `bwrap` in WSL2. |
| You use macOS | Run Yuj in a Linux virtual machine. Install `bwrap` in that machine. |
| Yuj already runs inside a secure container | Set `YUJ_CONTAINER=ambient`. |
| Another program created a task container | Set `YUJ_CONTAINER` to that container ID. |
| You accept shell commands with no sandbox | Apply the TOML file shown below. |

The `YUJ_CONTAINER` rows are legacy outer-container modes. Yuj does not create
or secure that outer container. They are distinct from the first-class
`[sandbox].backend = "container"` mode, and Yuj rejects a configuration that
sets both.

In `ambient` mode, Yuj tests whether it can remove network access with
`unshare -n`. Yuj warns when the test fails. The model shell can then use the
outer container's network access.

Set `YUJ_AMBIENT_UNSHARE_NET=0` only when you want to skip this test. This
setting does not block network access.

## What `bwrap` allows

With the normal strict settings:

- The model can write in the current project directory.
- The model can read the rest of the host file system.
- The model cannot write to the rest of the host file system.
- The model shell has no network access.
- Each shell call gets a new `/tmp`.
- The shell can use `/proc` and `/dev` inside their own namespaces.
- A temporary file system covers `.git/hooks` in the project.
- Yuj mounts the Docker socket when that socket exists on the host.

The Docker socket can give a model command access to the Docker service. Do
not expose the socket when the task must not use Docker.

When Agent Skills are enabled, Yuj adds validated external skill directories
to the read-only set. `read` can open their `SKILL.md` files and resources, and
`bwrap` mounts those directories read-only. File mutation tools still reject
every external skill path.

## Use the first-class container backend

Create a small overlay that names an image already present on the host:

```toml
[sandbox]
backend = "container"
container_runtime = "docker"  # or "podman"
container_image = "local/yuj-task@sha256:YOUR_DIGEST"
container_flags = ["--memory", "4g", "--pids-limit", "512"]
```

Yuj always uses `--pull=never`. Acquire and review the image yourself. Before
the model starts, Yuj finds the runtime, inspects the local image ID, and pins
commands to that ID. The image must contain `/usr/bin/env` and `/bin/bash`.
Also inspect its declared `VOLUME` paths, which become writable temporary
mounts even when the image root is read-only.

Check the local runtime and image before you use this backend:

```bash
docker version
docker image inspect --format='{{.Id}}' IMAGE
docker image inspect --format='{{json .Config.Volumes}}' IMAGE
docker system df
```

For each command, Yuj:

- mounts only the task directory read-write, at the same absolute path, plus
  each startup-validated external Agent Skill directory read-only at its
  absolute path;
- uses a read-only image root plus an ephemeral `/tmp`;
- disables the network and does not mount the Docker socket or host home;
- drops all capabilities, enables no-new-privileges, and uses private PID and
  IPC namespaces;
- overlays configured unreadable files and directories; and
- clears the image environment before applying Yuj's effective command
  environment.

Yuj accepts only resource limits and metadata flags that do not change
execution. It rejects mount, network, environment, entry-point, device,
privilege, security, and unknown flags. Each tool call gets a new container;
this backend does not reuse the persistent `bwrap` shell.

## Control the command environment

`[sandbox.env]` gives one environment to foreground and background shell
calls, `run_tests`, post-edit checks, and language servers. It applies under
every sandbox mode and on an explicitly unsandboxed command path. It does not
apply to trusted lifecycle hooks.

The default `inherit = "core"` inherits only `PATH`, `HOME`, `LANG`, and
`TERM` when present. Inherited names containing `KEY`, `SECRET`, or `TOKEN`
are removed by default. Fixed `set` values override inherited values, and
case-insensitive wildcard filters can exclude names or form a final include
allowlist. See [Configuration](configuration.html#control-the-command-environment)
for the exact order and settings.

Yuj builds this mapping once at run start. It clears the child environment
before applying it, even in ambient and unsandboxed modes. Yuj itself and the
model client keep the host environment. Saved provenance lists effective
variable names but redacts fixed values.

With `[tools].sandbox_required = true`, a missing runtime or local image stops
the task before a model command. Setting it to false prints one startup warning
and then runs commands without a sandbox.

The read-only host view can still contain private files. Hide selected paths
with `[sandbox].unreadable_paths`:

```toml
[sandbox]
unreadable_paths = [
  "/absolute/path/to/private-file",
  "/absolute/path/to/answer-keys/**",
]
```

Use an absolute path or a glob pattern.

In strict mode, Yuj treats a missing path with no glob as an error. Add
`optional:` before a path when its absence is valid.

For repository-owned model-view rules, use `.yujignore` instead of repeating
paths in host configuration. Yuj loads it once from the task root, applies its
Gitignore-style rules uniformly to file/search tools, and adds currently
matched paths to the unreadable masks used by bwrap and container commands.
Simple captured `ls`, `cat`, and `head` calls are filtered before execution so
an individually masked file is not exposed merely as a directory entry. See
[Configuration](configuration.html#hide-repository-paths-from-the-model-view)
for syntax, precedence, and trace provenance.

## Keep lifecycle hooks outside the task

Lifecycle hooks are trusted host programs. They do not run inside `bwrap`, a
first-class container, or a legacy task container. They also keep the Yuj
process environment instead of `[sandbox.env]`. A hook may therefore reach
host files and networks that model commands cannot reach.

The model cannot choose the command, but model and tool data can reach the
hook's JSON input. Use an absolute operator-owned path. Review the program,
validate every input field, and keep secrets out of command arguments. Do not
store the executable or any imported hook code in the task directory.

With `sandbox_required = true`, Yuj rejects a hook executable or path argument
that resolves inside the task directory. This includes commands such as
`python /task/hook.py`. It checks during startup and again before each launch,
so a task cannot add or retarget the path after startup. This check does not
sandbox the hook or prove that an external program is safe.

With `sandbox_required = false`, Yuj allows task-owned hook paths to run with
host permissions. Do this only for a repository you trust. See
[Configuration](configuration.html#run-trusted-lifecycle-hooks) for the full
handler contract.

## Turn the sandbox off

Create a small TOML file:

```toml
# no-sandbox.toml
[tools]
sandbox_bash = false
sandbox_required = false
```

Apply the file:

```bash
yuj code --config no-sandbox.toml "Your task"
```

Without the sandbox, model shell commands use your normal account permissions.
They can reach anything that your account can reach.

The approval check still pauses before specific risky commands. It does not
form a general security boundary.

## Python code mode

`exec_cell` uses the selected sandbox, command environment, writable task
directory, no-network boundary, read-only skill roots, and unreadable-path
masks. Unlike `bash` with `sandbox_required = false`, a cell never falls back
to the host. It returns an error when the sandbox is off, missing, or broken.

The injected `read`, `grep`, `glob`, `list_definitions`, and `bash` functions
return through the host dispatcher, but the model-written Python stays inside
the sandbox. `exec_cell_timeout` covers both the process and its inner calls.
An in-sandbox timer also stops the code if the container client disconnects.

## Separate model work from Yuj records

The sandbox controls shell commands that the model asks Yuj to run.

Yuj itself writes the trace, checkpoints, metrics, and session data. These
writes do not pass through the model shell. An installed package uses
`$XDG_STATE_HOME/yuj`, or `~/.local/state/yuj`; an editable/source checkout
uses its `.llm_assist/` directory. `HARNESS_ASSIST_HOME` overrides either.

Some Yuj-owned processes use different boundaries:

| Process or record | Boundary |
| --- | --- |
| Compaction hook | Runs inside the main Yuj process with host permissions. Enable only reviewed code. |
| File checkpoint store | Lives outside the task directory and is masked from model file and shell tools. Restore is not model-callable. |
| Language server | Runs under the selected command sandbox, file masks, and no-network rule. Yuj starts it only when needed and never downloads it. |
| Background command | Uses the same sandbox as `bash`. Yuj exposes output only through traced polls and stops the process group at session end. |

Read [Compaction hooks](compaction.html) for the trusted Python contract.

After most run segments, Yuj also tries to run `git add -A` and make a
checkpoint commit when the target repository has uncommitted changes. That
Git operation does not pass through the model approval check. Start with a
clean target repository.

Read [Saved files](harness_artifacts.html) for the full file list.

`tests/test_sandbox_escape.py` checks the `bwrap` boundary. To run the same
path, write, host-path, network, socket, and unreadable-mask checks through the
container backend without pulling an image:

```bash
YUJ_TEST_CONTAINER_IMAGE=LOCAL_IMAGE pytest -q \
  tests/test_container_backend_live.py
```
