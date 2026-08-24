---
layout: default
title: Run a local model
nav_order: 7
---

# Run a local model

Yuj can connect to any model server that offers a compatible API. You may
start that server yourself. You do not need the launch helpers on this page
for normal use.

The public repository also includes helpers for `llama-server` and vLLM. Use a
runtime file when you need to repeat a fixed server setup.

## Choose a path

| Need | Use |
| --- | --- |
| Connect Yuj to a server that is already running | `yuj setup --provider local` |
| Start a released server setup with fixed sampling values | `scripts/serve.sh RUNTIME.toml` |
| Check a runtime file without starting a server | `scripts/serve/llama_server.py --print` or `scripts/serve/vllm.py --print` |
| Start one GGUF from the one shipped launch-ready profile | `python -m scripts.llm_solver.server launch` |
| Wait for a `llama-server` health check | `python -m scripts.llm_solver.server wait` |

Use the runtime-file path for a paper comparison. The profile launcher does
not apply all fields that a fixed comparison needs.

Read [Extend Yuj with TOML files](extending-yuj.html) when you make a runtime
file or model profile for another model.

## Connect to a server that is already running

Ask the server for its model ID:

```bash
curl -fsS http://localhost:8080/v1/models
```

Save that exact ID:

```bash
yuj setup --provider local --model YOUR_SERVED_MODEL_ID
```

Add `--base-url` when the server does not use
`http://localhost:8080/v1`.

## Start a released runtime file

Run these commands from the Yuj repository.

Install the named server first. Put each local model path in a private copy of
the runtime file. Do not commit private paths or keys.

The repository currently ships `llama_server` runtime files under
`configs/runtime/`. Check one before you start it:

```bash
python3 scripts/serve/llama_server.py --print \
  configs/runtime/llama-devstral2-24b-q4km-5090-ctx20480.toml
```

The check validates the required fields and prints the command. A missing
model or template path produces a warning, not a failed check.

Start the server only after you review that command:

```bash
scripts/serve.sh \
  configs/runtime/llama-devstral2-24b-q4km-5090-ctx20480.toml
```

`scripts/serve.sh` reads `[launch].runtime`. It then replaces itself with the
named server process.

For `llama_server`, the helper uses
`[launch.llama_server].binary`. It uses `~/.local/bin/llama-server` when that
field is absent.

For `vllm`, set `VLLM_VENV` to a Python environment that contains the `vllm`
command:

```bash
export VLLM_VENV=/path/to/vllm-venv
scripts/serve.sh /path/to/private-vllm-runtime.toml
```

If `VLLM_VENV` is absent, the helper stops and asks you to set it.

After the server starts, ask `/v1/models` for the model ID. A
`llama_server` runtime may contain `served_name`, but the current translator
does not pass that field to `llama-server`.

## Runtime file format

A runtime file uses TOML and starts with this version:

```toml
schema_version = 1
```

The server translator reads `[launch]`. The measurement harness can also read
`[model]` when you pass the same file with `--config`.

### Shared launch fields

| Field | Required by | Meaning |
| --- | --- | --- |
| `runtime` | Both | Use `llama_server` or `vllm`. |
| `model_path` | Both | Read the model from this path. Relative paths start at the Yuj repository. `~` and environment variables expand. |
| `host` | Both | Listen on this host. |
| `port` | Both | Listen on this port. |
| `max_model_len` | Both | Set the server input limit. |
| `served_name` | vLLM | Set the model ID returned by vLLM. The llama translator does not use this field. |
| `max_num_seqs` | Optional | Set parallel slots. The two servers use different flag names. |
| `chat_template_path` | Optional | Pass this template file to the server. |
| `cpu_offload_gb` | vLLM only | Set vLLM CPU offload memory. |
| `gpu_memory_utilization` | vLLM only | Set vLLM GPU memory use. |

### Sampling fields

Put sampling values under `[launch.sampling]`.

Both translators require these fields:

```text
temperature
top_k
top_p
min_p
presence_penalty
repetition_penalty
seed
```

The vLLM translator also requires `frequency_penalty`.

The llama translator maps these values to separate `llama-server` flags. The
vLLM translator passes the whole table through
`--override-generation-config`.

### `llama-server` fields

Put server-specific values under `[launch.llama_server]`.

| Field | Server flag or use |
| --- | --- |
| `binary` | Server program; defaults to `~/.local/bin/llama-server` |
| `n_cpu_moe` | `--n-cpu-moe` |
| `cpu_mask` | `--cpu-mask` |
| `cpu_strict` | `--cpu-strict` |
| `threads` | `--threads` |
| `prio` | `--prio` |
| `flash_attn` | `--flash-attn on` when true |
| `cache_type_k`, `cache_type_v` | Key and value cache types |
| `jinja` | `--jinja` when true |
| `no_context_shift` | `--no-context-shift` when true |
| `batch`, `ubatch` | `-b`, `-ub` |
| `mmap` | `--mmap` when true |
| `n_predict` | `-n` |
| `n_gpu_layers` | `--n-gpu-layers` |
| `spec_type`, `spec_draft_n_max` | Speculative decoding settings |

### vLLM fields

Put vLLM-specific values under `[launch.vllm]`.

| Field | vLLM flag |
| --- | --- |
| `reasoning_parser` | `--reasoning-parser` |
| `tool_call_parser` | `--tool-call-parser` |
| `enable_auto_tool_choice` | `--enable-auto-tool-choice` when true |
| `moe_backend` | `--moe-backend` |
| `attention_backend` | `--attention-backend` |
| `enable_prefix_caching` | `--enable-prefix-caching` when true |

The public repository does not ship a vLLM runtime file. Make a private file
that follows this table. Check it before you run it:

```bash
python3 scripts/serve/vllm.py --print /path/to/private-vllm-runtime.toml
```

## Record the runtime used by a measurement

The server process receives `YUJ_SERVING_OVERLAY`, but a separate Yuj process
does not read that value. The current code does not record the runtime file
automatically.

Pass the same runtime file to the measurement command when you want it in the
recorded config list:

```bash
.venv/bin/python -m scripts.llm_solver RUN_DIR \
  --task /path/to/task \
  --config /path/to/runtime.toml \
  --prompt-text "Fix the failing tests."
```

This command reads `[model]` and ignores `[launch]`. The server helper reads
`[launch]`.

## Profile launcher

The repository also has a shorter `llama-server` launcher:

```bash
.venv/bin/python -m scripts.llm_solver.server launch \
  --profile qwen3.6-35b-a3b \
  --wait
```

The public release can launch only `qwen3.6-35b-a3b` this way. The other
shipped profiles do not contain a server model path. Use their runtime files
instead.

The profile launcher does not apply the profile's `[sampling]` table. It also
looks only for a file named `chat_template.jinja`. None of the shipped custom
templates uses that name. Do not use this launcher for a fixed comparison.

### Profile launcher commands

| Command | What it does |
| --- | --- |
| `python -m scripts.llm_solver.server launch` | Start `llama-server` from one profile. |
| `python -m scripts.llm_solver.server wait` | Poll `/health` until it returns `ok` or time runs out. |
| `python -m scripts.llm_solver.server stop` | Send `SIGKILL` to every process whose command line matches `llama-server`. |

Review running processes before you use `stop`. It does not stop only the
process that `launch` started.

| `launch` option | What it does |
| --- | --- |
| `--profile NAME` | Load this profile. Required. |
| `--port N` | Replace the port from `[server].base_url`. |
| `--gguf PATH` | Replace the profile's GGUF after the profile supplies a non-empty model path. |
| `--ctx N` | Replace the profile context size. |
| `--log PATH` | Write server output to this file. Without it, Yuj discards server output. |
| `--wait` | Wait for `/health` after launch. |
| `--timeout N` | Stop waiting after this many seconds. The default comes from `[server].launch_timeout`. |

| Other command | Options |
| --- | --- |
| `wait` | `--port N`, `--timeout N` |
| `stop` | `--settle N` waits this many seconds after the signals. |

The server command and each subcommand accept `-h` and `--help`.

## Profiles are trusted code

A profile tells Yuj how to shape model messages, tool schemas, and model
replies. Yuj first tries an exact profile name. It then tries one matching
`[profile].family`. It uses `_base` when neither match exists. A profile can
inherit another profile.

A profile may load Python files from its `normalize` and `denormalize`
folders. The current loader does not run a security check before it imports
them. Use only profiles that you trust.

Set `[model].profile_name` when the server's model ID differs from the profile
folder name. Read [Configuration](configuration.html) for the setting order.
