# Public source records

These files show where the published results came from. They do not copy the
private lab records or raw benchmark runs.

- [`cell_provenance.json`](cell_provenance.json) records the input hashes and
  the status of all ten published comparisons. For the four primary Qwen3.6
  pressure comparisons, it records hashes for the code and resolved settings.
  These hashes do not depend on local file paths. The file also records the
  checks for the three transfer-model comparisons.
- [`analysis_exclusions.tsv`](analysis_exclusions.tsv) lists the 17 tasks left
  out of the reading-boundary analysis. The analysis could not find a first
  file-write point for these tasks. After these exclusions, the analysis has
  166 Verified tasks, 315 Pro tasks, and 170 FeatureBench tasks. All 17 tasks
  remain in the task-outcome table.

The files keep the names and hashes needed to check the published results.
They leave out raw traces, transcripts, model output, patches, files made by
the evaluator, benchmark launchers, host paths, and personal experiment
records.
