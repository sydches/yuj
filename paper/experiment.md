# Experimental contract

This document defines the control and treatment conditions used by the paper.
It describes the completed experiment. It is not a component ablation or a
guide to the internal study machinery.

## Paired comparison

Each task in the released result table has one control run and one treatment
run. Within a pair, both runs used the same:

- task and prepared starting checkout;
- model weights and model-facing compatibility settings;
- context capacity and attempt budget;
- tool interface and model-serving setup;
- evaluator and scoring rule;
- task-agnostic harness code.

The two runs then unfolded independently. Configuration enabled the complete
treatment package on one side and left it disabled on the other. The measured
difference is therefore the effect of assigning that package as run. It does
not identify the separate contribution of each rule inside the package.

## Control

For each model call, control supplied the full chronological, model-visible
conversation produced by the shared harness. This was not raw terminal output:
both arms used the same ordinary tool-output cleanup before text entered the
conversation.

Control did not shorten older conversation entries in response to context
pressure and did not apply the treatment's stall response. When the assembled
conversation no longer fit the configured context window, the run could not
continue.

## Treatment

Treatment kept the conversation chronological. It did not replace the
conversation with a model summary, extracted fields, or semantic state.

As the context window filled, fixed mechanical rules shortened the displayed
bodies of older tool results. Shortening began only after the estimated full
prompt reached half of the configured window. The newest four tool results
remained complete. Older results retained their beginning and end around a
plain omission marker, using fixed age and length caps. The task and the
model's own replies remained unchanged, and the complete run record was still
written to the trace and transcript artifacts.

Treatment also checked fixed facts already present in the execution record for
stalled patterns, such as repeated attempts producing the same failure. The
checker used no model call. When a configured pattern was present, the harness
gave the model a fixed response describing the observed problem and directing
it toward a different action. A small set of fixed command safeguards handled
predictable command-shape and oversized-output problems.

Together, working-view management, mechanical stall detection and response,
and command safeguards are the treatment condition tested by the paper.

## Outcomes

The paper reports two task-level outcomes:

- **Resolution:** whether the benchmark evaluator accepted the task as fully
  solved.
- **F2P fraction (F2PF):** the fraction of required fail-to-pass tests made to
  pass by the submitted patch. This records partial repair even when a task is
  not fully resolved.

Each task had equal weight in the mean F2PF. The fixed scoring rule gave F2PF
zero to a task without a positive F2P denominator.

The complete task-level projection is available as
[`task_outcomes.tsv`](results/task_outcomes.tsv) and an identical
[`task_outcomes.md`](results/task_outcomes.md) table.

The analysis tested resolution differences with an exact two-sided McNemar
test over the discordant pairs. It tested F2PF differences with an exact
two-sided sign test after setting tied task pairs aside. The paper also reports
repository-cluster sensitivity analyses for the primary pressure comparisons.

## Cross-model comparisons

The comparisons applied the same frozen treatment settings to Qwen3.6-35B-A3B,
Devstral Small 2 24B, Nemotron-Cascade 2 30B-A3B, and Qwen3.8-27B. Each model
had one compatibility setup shared by its control and treatment runs. The
experiment did not retune any treatment rule for a model.

The Devstral, Nemotron, and Qwen3.8 comparisons used the same 169 SWE-bench
Verified tasks, a 20,480-token context window, and a 480-second total attempt
budget. Within each model, both arms used the same quantized weight file,
serving settings, chat template, task images, and harness code. Devstral and
Nemotron used Q4_K_M weights; Qwen3.8 used UD-Q4_K_XL weights.

## Isolation

The two arms began from separate copies of the same prepared task checkout.
Model commands ran in per-task containers without outbound network access.
The setup kept benchmark task tables, gold patches, hidden tests, and
scorer-only inputs outside the model-visible environment.

The reusable public harness retains this task-agnostic execution boundary.
Benchmark preparation, task manifests, launchers, and evaluators remain in the
external per-benchmark repositories.
