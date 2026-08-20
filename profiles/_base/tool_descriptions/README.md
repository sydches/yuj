# Tool descriptions

Each subdirectory is one description mode. Each `.txt` file holds the text
that Yuj sends with the tool named by that file. For example, `bash.txt` holds
the description for `bash`.

Tool inputs live in `../tool_schemas.toml`. A description mode changes only
the text that the model sees. It does not change tool names or inputs.

## Shipped mode

The public release ships `minimal/`. It is the default. Each file states what
one tool does and names common wrong uses.

## Selecting a mode

Set `[experiment] tool_desc = "<mode>"` in a settings file. The measurement
command also accepts `--tool-desc <mode>`. The loader is
`scripts/llm_solver/harness/schemas.py::get_tool_schemas`.

## Adding a mode

Create a directory with one `.txt` file for each tool in `tool_schemas.toml`.
The loader stops with an error when a description is missing.
