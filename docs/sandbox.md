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

Activate the Yuj virtual environment before you run a command on this page.
Otherwise, replace `yuj` with `/path/to/yuj/.venv/bin/yuj`.

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

## Use the first-class container backend

Create a small overlay that names an image already present on the host:

```toml
[sandbox]
backend = "container"
container_runtime = "docker"  # or "podman"
container_image = "local/yuj-task@sha256:YOUR_DIGEST"
container_flags = ["--memory", "4g", "--pids-limit", "512"]
```

Yuj runs the runtime with `--pull=never`; image acquisition is always a
separate operator action. Before the model starts, Yuj resolves the runtime,
inspects the local image ID, and pins later commands to that ID. The image must
contain `/usr/bin/env` and `/bin/bash`. Review and trust the image itself: an
image can contain executable startup behavior or declare `VOLUME` paths that
become writable ephemeral mounts despite a read-only root.

Check the local substrate before a campaign:

```bash
docker version
docker image inspect --format='{{.Id}}' IMAGE
docker image inspect --format='{{json .Config.Volumes}}' IMAGE
docker system df
```

For each command the backend:

- mounts only the task directory read-write, at the same absolute path;
- uses a read-only image root plus an ephemeral `/tmp`;
- disables the network and does not mount the Docker socket or host home;
- drops all capabilities, enables no-new-privileges, and uses private PID and
  IPC namespaces;
- overlays configured unreadable files and directories; and
- clears the image environment before applying Yuj's effective command
  environment.

Extra container flags use a fail-closed allowlist. Resource and inert metadata
options are accepted. Mount, network, environment, entrypoint, device,
privilege, security, and unknown flags are rejected. Container commands are
per-call; Yuj's persistent bwrap shell is not used.

## Control the command environment

`[sandbox.env]` applies one resolved environment to every command surface,
including foreground and background shell calls, `run_tests`, post-edit
checks, and language servers. It applies under bwrap, the first-class
container backend, both legacy container modes, and an explicitly unsandboxed
command path.

The default `inherit = "core"` inherits only `PATH`, `HOME`, `LANG`, and
`TERM` when present. Inherited names containing `KEY`, `SECRET`, or `TOKEN`
are removed by default. Fixed `set` values override inherited values, and
case-insensitive wildcard filters can exclude names or form a final include
allowlist. See [Configuration](configuration.html#control-the-command-environment)
for the exact order and settings.

Yuj resolves this mapping once at run start. It clears the child environment
before applying the mapping, including in ambient and unsandboxed command
modes. The harness and provider client keep their own host environment. Trace
provenance contains only the effective variable names. Fixed values are
redacted from saved resolved configuration.

With `[tools].sandbox_required = true`, a missing runtime or local image stops
the task before a model command. Setting it to false explicitly permits one
loud startup warning followed by unsandboxed command execution.

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

## Separate model work from Yuj records

The sandbox controls shell commands that the model asks Yuj to run.

Yuj itself writes the trace, checkpoints, metrics, and session data. These
writes do not pass through the model shell. In normal CLI use, Yuj saves them
under `.llm_assist/` in the Yuj installation.

When `[tools].file_checkpoints_enabled` is on, Yuj also writes an independent
shadow-Git repository outside the task directory after each potentially
mutating tool call. Its absolute path is added to the sandbox masks. The model
cannot use file tools or shell commands to inspect it, and restore is not a
model-callable operation.

Configured language servers are harness-owned session children, but they run
under the same filesystem masks and no-network boundary as model shell calls.
They start lazily, are never downloaded at runtime, and are stopped at session
teardown. If a configured binary is absent, Yuj warns once and continues
without diagnostics.

Background commands use the same selected sandbox backend, writable task
directory, unreadable masks, and no-network policy as synchronous `bash`.
Each command has its own process group. The harness captures combined output
outside the model's control, exposes bytes only through explicit traced polls,
and terminates all remaining groups before the session trace writer closes.

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
