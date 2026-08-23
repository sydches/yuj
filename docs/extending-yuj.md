---
layout: default
title: Extend Yuj with TOML files
nav_order: 5
---

# Extend Yuj with TOML files

A Yuj quirk is a TOML rule, not a Python plugin. You can read it in a minute,
share it with another user, and measure what it changes.

Use *extension* as the general name. A runtime file and a model profile are
extensions. Use *quirk* for a data rule that handles known model, language,
shell, or tool behavior.

Yuj reads these files at runtime. Use the file that owns your change. Use the
supported fields and rule types to extend Yuj without changing Python. This
guide also states when a change still needs Python.

Run the commands on this page from the Yuj repository. Activate its virtual
environment first. Otherwise, replace `yuj` with `.venv/bin/yuj`.

## Choose the right file

Yuj uses two kinds of TOML file.

- A settings file changes fields in `config.toml`.
- A descriptor file defines data for one loader.

Pass a settings file with `--config`.
Do not pass a descriptor file with `--config`.

| You want to change | File | How Yuj reads it | Python needed? |
| --- | --- | --- | --- |
| One coding session | A small settings file | `yuj code --config FILE.toml` | No, for an existing setting. |
| A model and local server | `configs/runtime/*.toml` | The config loader reads `[model]`. `scripts/serve.sh` reads `[launch]`. | No, for a supported server and field. |
| A model's message or tool-call format | `profiles/NAME/profile.toml` and its rule files | The profile loader selects it by name or family. | No, for a supported field or rule. A new transform needs Python. |
| A named subagent | `agents/NAME.toml` and its Markdown prompt | The optional `task` tool loads the descriptor by agent name. | No, for the current descriptor format. |
| A test runner or language | `scripts/llm_solver/language_quirks/NAME.toml` | The language loader finds matching project files. | No, for the current descriptor format. |
| A shell rewrite, refusal, redirect, or redaction | `scripts/llm_solver/bash_quirks/*.toml` | The shell tool loads each rule list. | No, for the current rule types. |
| The current `glob` refusal text | `scripts/llm_solver/tool_quirks/glob.toml` | The `glob` result filter reads it. | No. |
| An existing tool's input shape | `profiles/_base/tool_schemas.toml` | The tool-schema loader reads it. | The handler must already accept the same inputs. |
| The text sent with each tool | `profiles/_base/tool_descriptions/MODE/*.txt` | The tool-schema loader reads one complete mode. | No. |
| A new tool or server type | Python and its data files | The dispatcher or server launcher must know the new type. | Yes. |

## Share an extension

Yuj has no command that downloads or installs extensions. Share the named file
or directory. Ask the other user to review it before copying it into Yuj.

Yuj stores model quirks as profiles. It stores model-server values in runtime
files. It stores language, shell, and tool quirks in the matching directories.

| Extension | Share this | Add it this way |
| --- | --- | --- |
| Model server setup | One `configs/runtime/*.toml` file | Copy it to a private path. Replace local model and template paths. Pass it to `scripts/serve.sh` and `--config`. |
| Model message and tool-call format | One `profiles/NAME/` directory | Copy the directory under `profiles/`. Review every Python module before use. |
| Named subagent | One `agents/NAME.toml` file and its prompt | Copy both under `agents/`. Review the model profile, complete tool allowlist, prompt, turn limit, and read-only setting. |
| Test runner or language | One `language_quirks/NAME.toml` file | Copy it under `scripts/llm_solver/language_quirks/`. Give it a unique `detection_priority`. |
| Shell rewrite, refusal, redirect, or redaction | One or more TOML rule entries | Review and merge the entries into the matching fixed file under `bash_quirks/`. The current loader ignores other TOML files in that directory. |
| `glob` refusal text | The changed entries from `tool_quirks/glob.toml` | Review and merge them into that fixed file. The current loader does not scan extra tool-quirk files. |
| Tool description mode | One complete `tool_descriptions/MODE/` directory | Copy the directory under `profiles/_base/tool_descriptions/`. Include one `.txt` file for every tool. |

Do not share an API key, private host name, or private filesystem path. Keep
those values in `config.local.toml` or in a private runtime copy.

Test a shared extension before you use it on a real repository. The sections
below give the test command for each extension type.

A TOML file is easier to review than a Python plugin. Do not trust a file only
because it uses TOML. Review shell commands, executable paths, regular
expressions, and text sent to the model. Review every Python module named by a
profile.

## Contribute an extension

Yuj currently invites contributions for model profiles and quirk files.
Keep one pull request focused on one extension.

Include these items:

- the profile or quirk file;
- a short account of the behavior that the file handles;
- a test that fails without the extension and passes with it;
- the exact command used to check the extension;
- paired results when you claim that the extension changes task outcomes.

Do not include a private path, host name, API key, model file, task record, or
run log.

Open an issue before you change Python in the core harness. A new rule type,
tool, loader, or server type needs code review beyond the extension file.

Do not call an extension community-tested until another user has tested it.

## Understand the settings layers

Yuj calls a small TOML settings file an overlay. An overlay contains only the
fields that it changes.

For the installed `yuj` command, Yuj applies settings in this order:

1. `config.toml`
2. `config.local.toml`
3. the selected treatment or plain base
4. each `--config` overlay from left to right
5. command options

A later value replaces an earlier value for the same field.

Use a small overlay for a normal coding session:

```toml
# my-run.toml
[model]
context_size = 32768
tokenizer_id = ""

[loop]
max_turns = 120

[tools.run_tests]
enabled = true
```

Apply it after setup:

```bash
yuj code --config my-run.toml "Fix the issue and run the tests."
```

Read [Configuration](configuration.html) for the full setting order and the
main setting groups.

## Know the repository layout

The public data files have separate jobs.

| Path | Job |
| --- | --- |
| `config.toml` | Supply normal defaults and the full public settings shape. |
| `config.local.toml` | Keep this machine's model service and key reference. Git ignores this file. |
| `configs/regimes/` | Define the released treatment and plain bases. |
| `configs/runtime/` | Define one model and one local server setup. |
| `configs/paper/` | Fix the layer order and detector limits for released paper comparisons. |
| `configs/treatment/` | Supply the released detector data and response overlays. |
| `agents/` | Define named agents for the optional sequential `task` tool. |
| `profiles/` | Adapt model messages, tool schemas, and replies. |
| `scripts/llm_solver/language_quirks/` | Describe test runners and their output. |
| `scripts/llm_solver/bash_quirks/` | Describe shell rewrites, refusals, and redactions. |
| `scripts/llm_solver/tool_quirks/` | Describe supported result changes for non-shell tools. |

Do not mix these jobs in one file.

## Add another model

Start with the standard `_base` profile when the model service accepts
OpenAI-style messages and tool calls.

### Connect to a server that already runs

Ask the server for its model ID:

```bash
curl -fsS http://localhost:8080/v1/models
```

Save that ID:

```bash
yuj setup --provider local --model YOUR_SERVED_MODEL_ID
```

Run a smoke task before you change a real repository:

```bash
yuj smoke
```

Read [Getting started](getting-started.html) for online services and custom
addresses.

### Make a runtime file

Make a runtime file when you want Yuj to start `llama-server` or vLLM with
fixed values.

The `[model]` table configures the harness. The `[launch]` tables configure the
model server.

Start a `llama-server` file with this shape:

```toml
schema_version = 1

[model]
name = "YOUR_SERVED_MODEL_ID"
profile_name = "_base"
context_size = 32768
tokenizer_id = ""

[launch]
runtime = "llama_server"
model_path = "~/models/your-model.gguf"
host = "127.0.0.1"
port = 8080
max_model_len = 32768
max_num_seqs = 1

[launch.sampling]
temperature = 0
top_k = 1
top_p = 1
min_p = 0
presence_penalty = 0
repetition_penalty = 1
seed = 42

[launch.llama_server]
n_gpu_layers = 99
flash_attn = true
cache_type_k = "q8_0"
cache_type_v = "q8_0"
jinja = true
```

Keep local model paths in a private copy.
Do not commit a key or a private host path.

Check the file before you start the server:

```bash
python3 scripts/serve/llama_server.py --print /path/to/my-runtime.toml
```

Start the server after you review the printed command:

```bash
scripts/serve.sh /path/to/my-runtime.toml
```

Ask `/v1/models` for the actual model ID after the server starts.
Put that ID in `[model].name`.

Apply the same file to a coding session when you want its `[model]` values:

```bash
yuj code --config /path/to/my-runtime.toml "Fix the issue."
```

The config loader ignores `[launch]`. The server helper ignores `[model]`.
Read [Run a local model](serving_overlay.html) for every launch field and the
vLLM differences.

### Add a model profile

Add a profile when `_base` does not match the model's message or tool-call
format.

Create `profiles/my-model/profile.toml`:

```toml
[profile]
format_version = 1
canonical_version = "openai-v1"
name = "my-model"
family = "my-model-family"
inherits = "_base"

[model]
supports_tool_calls = true
supports_system_role = true
supports_prefill = false

[capacity]
max_tools = 8
simplify_schemas = false
```

Select it from a settings or runtime file:

```toml
[model]
profile_name = "my-model"
```

Yuj first looks for an exact profile directory. It next looks for one profile
with the requested `[profile].family`. It uses `_base` when neither exists.
More than one family match is an error.

Use these profile fields for these active jobs:

| Profile field | Active job |
| --- | --- |
| `[profile].inherits` | Load a parent profile first. |
| `[model].supports_tool_calls` | Send or omit the tool schema list. |
| `[model].supports_system_role` | Keep or fold the system message. |
| `[model].supports_prefill` | Authorize assistant-prefill length continuation for this exact profile and chat template. This does not claim that every provider accepts llama-server continuation extras. |
| `[capacity].preamble` | Add text before the system prompt. |
| `[capacity].max_tools` | Limit the number of enabled tools sent to the model. |
| `[capacity].simplify_schemas` | Remove descriptions from tool schemas. |
| `[normalize].rules` | Apply supported rules to a model reply. |
| `[denormalize].rules` | Choose how Yuj sends the system message. |
| `[server]` | Supply values to the limited profile-based `llama-server` launcher. |

The TOML reply rules can remove text with a regular expression. They can map
finish reasons. They can also add missing tool-call IDs. Use a trusted Python
module for a more complex message or reply change.

Yuj imports profile Python modules without checking them first. Use only
profile code that you trust.

Do not use profile `[model].context_size` to set the live server limit. Use
runtime `[model].context_size` for the harness. Use
`[launch].max_model_len` for the server.

Do not use profile `[model].chat_template` to select a template file. Use
runtime `[launch].chat_template_path`.

Profile `[tokens].method` currently supports only `chars_div_4`.
Profile `[tokens].tokenizer` is not used for the harness preflight count. Use
runtime or settings field `[model].tokenizer_id` for that count.

Load the profile without starting a model:

```bash
.venv/bin/python -c \
  'from pathlib import Path; from scripts.llm_solver.server import load_profile; p = load_profile("my-model", Path("profiles")); print(p.name, p.family, p.max_tools)'
```

Run the full tests after you add profile rules or modules.

## Add a named agent

Named agents are descriptors for the optional `task` tool. Add
`agents/my-agent.toml` and keep its system prompt under `agents/`:

```toml
[agent]
model_profile = "_base"
tools = ["read", "glob", "grep", "bash", "done"]
system_prompt_file = "prompts/my-agent.md"
max_turns = 12
read_only = true
```

The descriptor must contain exactly one `[agent]` table. `model_profile`
selects a profile name or family under `profiles/`; `tools` is the complete
model-facing allowlist. The prompt must be an existing Markdown file below
`agents/`.
Agents default to read-only, which rejects mutation tools and limits `bash` to
a small inspection-command allowlist. Set `read_only = false` only when the
agent is deliberately allowed to modify the task directory.

Enable the caller with `[tools].task_enabled = true`. The public depth and turn
caps remain authoritative over descriptor values. See
[`agents/README.md`](https://github.com/sydches/yuj/blob/main/agents/README.md)
for the complete validation rules.

## Add a language or test runner

The `run_tests` tool checks files in the task repository. It picks the first
matching runner in `detection_priority` order. A lower number runs first.

The public release includes `pytest`, Cargo, Go, Jest, and CTest descriptors.
It uses pytest when no descriptor matches.

Add `scripts/llm_solver/language_quirks/my-runner.toml` with this shape:

```toml
name = "my-runner"
description = "My project test runner"

verification_patterns = [
  '''(?:^|[\s/'"])my-test\b''',
]

[output_control]
failure_only_flag = ""
passed_marker = "PASS"
failed_marker = "FAIL"

[run_tests]
detection_priority = -1
env_activate_prefix = ""
base_cmd = "my-test"
detect_files = ["my-project.toml"]
arg_path_style = "positional"
arg_k_template = "--name {expr}"
arg_last_failed = ""
status_default = "failed"

[run_tests.status_map]
0 = "passed"
```

Use a unique `detection_priority`.
Use a lower number when this runner must win in a mixed-language repository.

`detect_files` checks the task repository root. It does not search every
subdirectory.

Use `arg_path_style = "positional"` when `run_tests(path=...)` should append
the path. Use `arg_path_style = "ignored"` when the runner cannot take a path.

Add `[output_parser.summary]` or `[output_parser.per_test]` only when the
runner has stable output that the current parser can read. The exit status
still supplies the main `passed` or `failed` result.

`generic.toml` has no `[run_tests]` table. It helps Yuj recognize verification
commands, but `run_tests` never selects it as a runner.

The `run_tests` tool detects its runner even when `[analysis].task_format`
names another format. Set the format to `auto` when shell-output rules must
follow the detected runner:

```toml
[analysis]
task_format = "auto"
```

Set `task_format = "my-runner"` when every task must use that descriptor.

The `[output_control]` table takes effect only when
`[loop].bash_transforms_task_format_enabled` is true. The `[output_parser]`
table also needs `[loop].bash_transforms_structured_output_enabled = true`.

Check which runner Yuj selects:

```bash
.venv/bin/python -c \
  'from scripts.llm_solver.language_quirks import detect_runner, load_run_tests_quirk_object; path = "/path/to/project"; print(detect_runner(path)); print(load_run_tests_quirk_object(path))'
```

Run the language and tool tests:

```bash
.venv/bin/python -m pytest -q \
  tests/test_run_tests_tool.py \
  tests/test_composability.py
```

## Change tool behavior

### Change an existing tool setting

Use a settings overlay for an existing switch or limit:

```toml
[tools]
glob_max_matches_per_page = 40
glob_max_listed_paths = 100
glob_refuse_unscoped_recursive = true

[tools.run_tests]
enabled = true
timeout = 300
```

Read [Model tools](model-tools.html) for every public tool and input.

### Add a shell rule

Use the rule file that matches the action:

| File | Entry | What one entry does |
| --- | --- | --- |
| `bash_quirks/rewrites.toml` | `[[rewrite]]` | Match a command and add one flag unless `skip_if` matches. Yuj applies at most one rewrite rule to a command. |
| `bash_quirks/forbidden.toml` | `[[forbidden]]` | Match a command and replace it with a failed refusal command. Yuj uses the first match. |
| `bash_quirks/forbidden.toml` | `[[redirect]]` | Match the full command and compound fragments before rewrites, then return a typed error naming a dedicated tool. The rule applies only while that tool is active. |
| `bash_quirks/redactions.toml` | `[[redaction]]` | Replace matching result text before Yuj shortens or saves it. Yuj applies all rules in order. |

The shell paths start at `scripts/llm_solver/`.

Set `[loop].bash_transforms_universal_enabled = false` to turn off the
universal rewrite list. Set `[loop].bash_quirks_forbidden_enabled = false` to
turn off the forbidden list. Set `[tools].bash_redirect_read_side = true` to
activate redirect rules targeting `read`, `grep`, or `glob`; redirect rules
targeting `write` or `edit` remain active. Every redirect is gated on its
target appearing in the effective model-facing tool set. Secret redaction has
no off switch.

Test a new rule against both a matching command and a command that must stay
unchanged:

```bash
.venv/bin/python -m pytest -q \
  tests/test_bash_forbidden.py \
  tests/test_bash_quirks_defect_fixes.py
```

### Change a non-shell tool result

`scripts/llm_solver/tool_quirks/glob.toml` owns only the two refusal messages
for `glob`. Put its numeric limits in a settings overlay under `[tools]`.

The current `tool_quirks` loader is not a general tool plugin loader. A new
result transform needs Python in `tool_quirks/transforms.py` and a call from
the tool handler.

### Change tool schemas or descriptions

`profiles/_base/tool_schemas.toml` owns the input shape that the model sees.
The Python handler must accept the same inputs.

Do not add a new tool only to `tool_schemas.toml`. A new tool also needs a
`ToolSpec`, a handler, dispatch wiring, and tests.

Add a new description mode without changing Python. Create one `.txt` file
for every tool under `profiles/_base/tool_descriptions/MODE/`. Select the mode
with `[experiment].tool_desc` or measurement option `--tool-desc`.

Run the schema tests after either change:

```bash
.venv/bin/python -m pytest -q \
  tests/test_composability.py \
  tests/test_harness_pipeline_session.py \
  tests/test_run_tests_tool.py
```

## Know when TOML is not enough

| Change | Why Python is needed |
| --- | --- |
| A new model server type | `scripts/serve.sh` needs a new case and a translator. |
| A model message or reply change beyond the profile rule types | The rule engine cannot keep state or run general code. |
| A new tool | Yuj needs a handler, registry data, dispatch code, and tests. |
| A new non-shell tool result change | The tool handler must call it. |
| A new settings field | `Config` and the config loader must define it. |
| A new TOML rule shape | Its loader must parse and apply it. |

## Contribute a profile or quirk

Start a contribution with one data-only profile or quirk. Do not mix a core
Python change into the same contribution.

A data-only profile contains no Python module.

Include these items:

- State the behavior that the extension handles.
- State the model, language, tool, and version in scope.
- Add the smallest TOML file, profile directory, or rule entry that handles it.
- Add a test that fails without the extension and passes with it.
- Give the before-and-after measurement command and result.
- Remove keys, private paths, task data, and private notes.

Keep benchmark tasks, launch control, scoring code, and raw benchmark records
in the external benchmark repository. Link to a result when another person
needs to check it.

If an extension needs Python, state that boundary first. Keep the Python change
separate so reviewers can judge the data and code on their own.

## Measure an extension or another model

Share the measurement with the extension. Record the source revision, every
settings file in order, the task set, the checking method, and the result.

For a settings overlay, run the same measurement once without the overlay and
once with it. For a profile, change only `[model].profile_name`. For a language
descriptor or one rule in a fixed quirk file, compare two source revisions and
state that the extension is the only intended difference.

Yuj records the run. The external benchmark checks and scores the result.

Treat a comparison on another model as a new experiment. Do not call it an
exact reproduction of a released paper comparison.

For a plain run, apply the plain base before the model runtime:

```bash
.venv/bin/python -m scripts.llm_solver RUN_DIR \
  --task /path/to/task \
  --config configs/regimes/baselines/plain_long_solve.toml \
  --config /path/to/my-runtime.toml \
  --context full \
  --prompt-text "Fix the failing tests."
```

For a treatment run, apply the model runtime before the treatment base:

```bash
.venv/bin/python -m scripts.llm_solver RUN_DIR \
  --task /path/to/task \
  --config /path/to/my-runtime.toml \
  --config configs/regimes/treatment.toml \
  --context halflife \
  --prompt-text "Fix the failing tests."
```

Keep the model, task, prompt, input limit, sampling values, and checking method
the same across the two runs. Record every overlay and its order.

Keep task setup, launch control, and scoring in the external benchmark
repository. Read [Measurements](measurement.html) for the command contract.
Read the [paper configuration guide](https://github.com/sydches/yuj/tree/main/configs/paper)
only when you reproduce a released paper comparison.
