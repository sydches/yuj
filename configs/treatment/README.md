# Detector data and responses

These files retain the treatment response definitions. The current trace_nets
backend records observations; it does not select hurdles or apply these overlays.
Enable a response explicitly in the declared settings stack when it is needed.

`configs/regimes/treatment.toml` selects them. A normal user does not select
them during setup.

- `hurdle_dictionary.trace_nets.v1.tsv` defines two kinds of problem that the
  detector can report.
- `medicine_ladder.v1.tsv` puts five detector responses in order.
- `overlays/` contains one settings change for each response.

| Overlay | Effect |
| --- | --- |
| `overlays/loop_detect.toml` | Enables notices about completed observations. |
| `overlays/duplicate_guard.toml` | Warns once on the second identical completed call; reuses eligible unchanged inspections from the third. |
| `overlays/loop_detect_recovery.toml` | Enables the same warning and reuse policy; retains the historical overlay path. |
| `overlays/unified_envelope.toml` | Wraps later tool results in the structured result envelope. |
| `overlays/intent_gate.toml` | Requires intent text and replaces the repeat message. |

The TSV files contain only fields that the released code reads. Their paths
start at the Yuj repository. They do not depend on a user's home directory or
the target repository.
