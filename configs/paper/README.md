# Paper configurations

Use these files to reproduce the four primary Qwen3.6 pressure comparisons in
the paper. A normal coding session does not need them.

The public result table contains ten comparisons. This guide gives the complete
harness file order for four of them. The
[public source records](../../paper/provenance/README.md) cover all ten
comparisons.

Start with the root `config.toml`. Apply the listed files from left to right.
The last file wins when two files set the same field.

The CLI default and the paper treatments share the same
`configs/regimes/treatment.toml` base. An exact paper treatment also needs the
listed runtime and detector-threshold layers. Likewise, `yuj --no-treatment`
selects the same plain base and full-context choice used by the
paper controls, but an exact paper control also needs its listed runtime layer.

| Paper comparison | Control layers and context mode | Treatment layers and context mode |
|---|---|---|
| Verified 20,480 | `configs/regimes/baselines/plain_long_solve.toml` -> `configs/runtime/llama-qwen36-35b-a3b-q4kxl-5090-no-offload-mtp-ctx20480.toml`; `full` | `configs/runtime/llama-qwen36-35b-a3b-q4kxl-5090-no-offload-mtp-ctx20480.toml` -> `configs/regimes/treatment.toml` -> `configs/paper/thresholds/verified_20480.toml`; `halflife` |
| Verified 43,008 | `configs/regimes/baselines/plain_long_solve.toml` -> `configs/runtime/llama-qwen36-35b-a3b-q4kxl-5090-no-offload-mtp-ctx43008.toml`; `full` | `configs/runtime/llama-qwen36-35b-a3b-q4kxl-5090-no-offload-mtp-ctx43008.toml` -> `configs/regimes/treatment.toml` -> `configs/paper/thresholds/verified_43008.toml`; `halflife` |
| SWE-bench Pro 49,152 | `configs/regimes/baselines/plain_long_solve.toml` -> `configs/runtime/llama-qwen36-35b-a3b-q4kxl-5090-no-offload-mtp-ctx48k.toml`; `full` | `configs/runtime/llama-qwen36-35b-a3b-q4kxl-5090-no-offload-mtp-ctx48k.toml` -> `configs/regimes/treatment.toml` -> `configs/paper/thresholds/pro_49152.toml`; `halflife` |
| FeatureBench 47,104 | `configs/regimes/baselines/plain_long_solve.toml` -> `configs/runtime/llama-qwen36-35b-a3b-q4kxl-5090-no-offload-mtp-ctx47104.toml`; `full` | `configs/runtime/llama-qwen36-35b-a3b-q4kxl-5090-no-offload-mtp-ctx47104.toml` -> `configs/regimes/treatment.toml` -> `configs/paper/thresholds/featurebench_47104.toml`; `halflife` |

## Practitioner grid recipes

The 43,008-token practitioner grid has one serving overlay and one solver
recipe for each cell. Arm `a00` disables all eight model-visible
transformations and selects full history. Arm `a09` enables all eight and
selects half-life context.

| Quantization | Serving overlay | `a00` solver recipe | `a09` solver recipe |
| --- | --- | --- | --- |
| Q2_K_XL | `configs/runtime/llama-qwen36-35b-a3b-q2kxl-5090-no-offload-mtp-ctx43008.toml` | `configs/paper/practitioner_grid/q2_k_xl-43008-a00.toml` | `configs/paper/practitioner_grid/q2_k_xl-43008-a09.toml` |
| IQ4_XS | `configs/runtime/llama-qwen36-35b-a3b-iq4xs-5090-no-offload-mtp-ctx43008.toml` | `configs/paper/practitioner_grid/iq4_xs-43008-a00.toml` | `configs/paper/practitioner_grid/iq4_xs-43008-a09.toml` |
| Q4_K_XL | `configs/runtime/llama-qwen36-35b-a3b-q4kxl-5090-no-offload-mtp-ctx43008.toml` | `configs/paper/practitioner_grid/q4_k_xl-43008-a00.toml` | `configs/paper/practitioner_grid/q4_k_xl-43008-a09.toml` |

Start the server with the serving overlay for the selected row. Then give the
solver the matching recipe as one `--config` argument:

```bash
scripts/serve.sh \
  configs/runtime/llama-qwen36-35b-a3b-q2kxl-5090-no-offload-mtp-ctx43008.toml

.venv/bin/python -m scripts.llm_solver /path/to/run-q2-a09 \
  --task /path/to/task-checkout \
  --config configs/paper/practitioner_grid/q2_k_xl-43008-a09.toml
```

Each solver recipe references the same six owner files used by the measured
cell, in the measured order. The serving runtime is already one of those six
solver layers, so do not pass it to the solver a second time. Change a
referenced owner file when you want a different runtime, threshold, arm, or
logging level.

## Run one released pair

Install a `llama-server` build that supports MTP. Get the Qwen3.6 MTP GGUF
named by the runtime file. Put the file at the path in `[launch].model_path`.
You may instead make a private copy of the runtime file and change that path.

Read the [local-model guide](../../docs/serving_overlay.md) before you start
the server. Check the runtime file first:

```bash
python3 scripts/serve/llama_server.py --print \
  configs/runtime/llama-qwen36-35b-a3b-q4kxl-5090-no-offload-mtp-ctx20480.toml
```

Start the model server with the same runtime file:

```bash
scripts/serve.sh \
  configs/runtime/llama-qwen36-35b-a3b-q4kxl-5090-no-offload-mtp-ctx20480.toml
```

Use the benchmark's external repository to prepare two separate copies of the
same task checkout.
Put the same `prompt.txt` in both copies.
Use a new run directory for each arm.

Run the control arm:

```bash
.venv/bin/python -m scripts.llm_solver /path/to/runs/verified-20480-control \
  --task /path/to/tasks/verified-20480-control \
  --config configs/regimes/baselines/plain_long_solve.toml \
  --config configs/runtime/llama-qwen36-35b-a3b-q4kxl-5090-no-offload-mtp-ctx20480.toml \
  --context full
```

Run the treatment arm:

```bash
.venv/bin/python -m scripts.llm_solver /path/to/runs/verified-20480-treatment \
  --task /path/to/tasks/verified-20480-treatment \
  --config configs/runtime/llama-qwen36-35b-a3b-q4kxl-5090-no-offload-mtp-ctx20480.toml \
  --config configs/regimes/treatment.toml \
  --config configs/paper/thresholds/verified_20480.toml \
  --context halflife
```

These commands apply the released harness settings.
They do not prepare the benchmark task or score its result.
Use the benchmark's own task and scoring tools for those jobs.

For another released comparison, replace the runtime and threshold files with
the files in its table row.

The historical Pro launch also wrote a per-run overlay containing only
`max_turns = 250`. Both public arm files already set that value, so repeating
the no-op layer would not change the resolved configuration.

The table describes the harness and serving layers only. Task lists,
evaluators, wall-time continuation launchers, and benchmark setup remain in
the external repository for each benchmark.
