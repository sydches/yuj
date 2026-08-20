# Treatment data

These files support the released treatment base.

`configs/regimes/treatment.toml` selects them. A normal user does not select
them during setup.

- `hurdle_dictionary.trace_nets.v1.tsv` defines two kinds of problem that the
  detector can report.
- `medicine_ladder.v1.tsv` puts five responses in order.
- `overlays/` contains the settings changed by those responses.

The TSV files contain only fields that the released code reads. Their paths
start at the Yuj repository. They do not depend on a user's home directory or
the target repository.
