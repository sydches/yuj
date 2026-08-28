# Detector data and responses

These files support the detector in the released treatment base.

`configs/regimes/treatment.toml` selects them. A normal user does not select
them during setup.

- `hurdle_dictionary.trace_nets.v1.tsv` defines two kinds of problem that the
  detector can report.
- `medicine_ladder.v1.tsv` puts five detector responses in order.
- `overlays/` contains one settings change for each response.

| Overlay | Effect |
| --- | --- |
| `overlays/loop_detect.toml` | Turns on the identical-call guardrail. |
| `overlays/duplicate_guard.toml` | Warns after the first duplicate call. |
| `overlays/loop_detect_recovery.toml` | Turns on the identical-call guardrail and replaces its recovery message. |
| `overlays/unified_envelope.toml` | Wraps later tool results in the structured result envelope. |
| `overlays/intent_gate.toml` | Requires intent text and replaces the repeat message. |

The TSV files contain only fields that the released code reads. Their paths
start at the Yuj repository. They do not depend on a user's home directory or
the target repository.
