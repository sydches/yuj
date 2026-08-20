# Measurement bases

Normal `yuj` use starts from the root `config.toml`. The installed command
selects a base for you. A base is the starting group of settings.

The public release has two bases for measurements:

- `baselines/plain_long_solve.toml` is the plain control base.
- `treatment.toml` is the complete treatment base.

These files do not choose a task list, model server, input limit, continuation
schedule, or checking program. The paper files record those other settings
separately.

This directory contains only the released plain and treatment bases. It does
not contain the earlier work used to choose them. It also does not contain
benchmark task lists, launchers, checking programs, or private study records.
