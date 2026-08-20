"""Primary CLI entrypoint for the yuj coding agent."""
from __future__ import annotations

from .llm_assist.__main__ import main


if __name__ == "__main__":
    raise SystemExit(main())
