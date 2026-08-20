---
layout: default
title: Sandbox
nav_order: 8
---

# Sandbox

A sandbox limits what a model shell command can reach. Yuj normally uses
Linux `bubblewrap` (`bwrap`) for this job.

You do not need Docker for normal use.

Activate the Yuj virtual environment before you run a command on this page.
Otherwise, replace `yuj` with `/path/to/yuj/.venv/bin/yuj`.

## Choose a mode

| Your situation | What to do |
| --- | --- |
| You run Yuj on Linux | Leave `YUJ_CONTAINER` unset. Yuj uses `bwrap`. |
| You use Windows | Run Yuj in WSL2. Install `bwrap` in WSL2. |
| You use macOS | Run Yuj in a Linux virtual machine. Install `bwrap` in that machine. |
| Yuj already runs inside a secure container | Set `YUJ_CONTAINER=ambient`. |
| Another program created a task container | Set `YUJ_CONTAINER` to that container ID. |
| You accept shell commands with no sandbox | Apply the TOML file shown below. |

Yuj does not create or secure the outer container in either container mode.
The user or external launcher must do that work.

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

After most run segments, Yuj also tries to run `git add -A` and make a
checkpoint commit when the target repository has uncommitted changes. That
Git operation does not pass through the model approval check. Start with a
clean target repository.

Read [Saved files](harness_artifacts.html) for the full file list.

The `tests/test_sandbox_escape.py` tests check the `bwrap` boundary.
