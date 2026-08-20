# Paper results

This directory gives the released experiment design and paired task results.

- [`experiment.md`](experiment.md) defines the paired comparison, control,
  treatment, and outcomes.
- [`results/task_outcomes.tsv`](results/task_outcomes.tsv) gives the result for
  each task.
- [`results/task_outcomes.md`](results/task_outcomes.md) shows the same data as
  a Markdown table.
- [`provenance/`](provenance/README.md) gives the public source records and the
  reading-analysis exclusion list.
- [`../configs/paper/README.md`](../configs/paper/README.md) gives the exact
  settings-file order for the four primary Qwen3.6 pressure comparisons.

## Coverage

The table contains 2,012 paired task rows. It reports seven Qwen3.6 comparisons
across three benchmarks and several context windows. It also reports one
169-task SWE-bench Verified pressure comparison for each other model.

| Model | Benchmark | Context tokens | Tasks | Control resolved | Treatment resolved | Control mean F2PF | Treatment mean F2PF |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.6-35B-A3B | SWE-bench Verified | 20,480 | 169 | 43 | 72 | 0.280 | 0.491 |
| Qwen3.6-35B-A3B | SWE-bench Verified | 43,008 | 169 | 76 | 87 | 0.538 | 0.601 |
| Qwen3.6-35B-A3B | SWE-bench Verified | 262,144 | 169 | 102 | 101 | 0.690 | 0.687 |
| Qwen3.6-35B-A3B | SWE-bench Pro | 49,152 | 316 | 31 | 72 | 0.152 | 0.327 |
| Qwen3.6-35B-A3B | SWE-bench Pro | 262,144 | 316 | 101 | 99 | 0.448 | 0.445 |
| Qwen3.6-35B-A3B | FeatureBench | 47,104 | 183 | 2 | 3 | 0.105 | 0.196 |
| Qwen3.6-35B-A3B | FeatureBench | 262,144 | 183 | 5 | 5 | 0.239 | 0.307 |
| Devstral Small 2 24B | SWE-bench Verified | 20,480 | 169 | 22 | 53 | 0.172 | 0.368 |
| Nemotron-Cascade 2 30B-A3B | SWE-bench Verified | 20,480 | 169 | 16 | 25 | 0.120 | 0.183 |
| Qwen3.8-27B | SWE-bench Verified | 20,480 | 169 | 32 | 54 | 0.204 | 0.353 |

Each row reports one task under control and treatment. `resolved` is `1` when
the benchmark accepts the task as solved. F2PF is the fraction of required
fail-to-pass tests that the patch passes. The released scoring rule gives F2PF
zero when the denominator is zero.

These are the finalized primary task outcomes used by the paper. Internal run
delivery labels are intentionally omitted because they are not experimental
conditions.

## Check the public files

Run this command from the repository root:

```bash
sha256sum \
  paper/results/task_outcomes.tsv \
  paper/results/task_outcomes.md \
  paper/provenance/analysis_exclusions.tsv \
  paper/provenance/cell_provenance.json
```

The expected hashes are:

| Output | Rows | SHA-256 |
| --- | ---: | --- |
| `task_outcomes.tsv` | 2,012 | `68ca128d2bc7f1184984a1e6483bd1e5f7b7d8b32a1b11d2944eb47c6f3a5370` |
| `task_outcomes.md` | 2,012 | `48ee4c2e61b7ce4ea02f9c7b22e0b41e165e3549f6d92106a6cd34aa8365eb1c` |
| `provenance/analysis_exclusions.tsv` | 17 | `eea1a8a7a349eb4414268f4c523b4bf12aee6b44fed3c0cae939da62d28c1d51` |
| `provenance/cell_provenance.json` | 10 cells | `2d8168a8c7d1d81f0a2c8fc51c18cead1d5b3d5cc2398a4b06a0e17ca5dde44b` |

The result files do not include run logs, model output, patches, benchmark
data, launch files, or the search that produced the released settings.
