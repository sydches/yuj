# Third-party notices

Except where otherwise noted, the MIT License in [`LICENSE`](LICENSE) covers
Yuj's original work. The components below retain their upstream Apache
License 2.0 terms. This repository provides the full text in
[`LICENSES/Apache-2.0.txt`](LICENSES/Apache-2.0.txt).

## Unsloth chat-template modifications

- Copyright 2025-present the Unsloth team.
- Source: <https://github.com/unslothai/unsloth>
- License: Apache License 2.0
- Included files:
  - `profiles/devstral2-24b/chat_template_patched.jinja`
  - `profiles/qwen3.6-35b-a3b/chat_template_meanderix.jinja`

The original copyright and license headers remain in both files.

## OpenAI Codex-derived portions

- OpenAI Codex
- Copyright 2025 OpenAI
- Source: <https://github.com/openai/codex>
- License: Apache License 2.0
- Yuj files whose source comments identify behavior or implementation as
  ported, borrowed, or adapted from Codex:
  - `scripts/llm_solver/config.py`
  - `scripts/llm_solver/_main_helpers.py`
  - `scripts/llm_solver/_shared/digest_core.py`
  - `scripts/llm_solver/harness/apply_patch.py`
  - `scripts/llm_solver/harness/_loop/_driver_setup.py`
  - `scripts/llm_solver/harness/sandbox/__init__.py`
  - `scripts/llm_solver/harness/sandbox/_preflight.py`
  - `scripts/llm_solver/harness/sandbox/_unreadable.py`

The listed Yuj files are not wholesale copies of the upstream repositories.
This notice records the third-party provenance already stated in their source
comments and preserves the applicable upstream attribution.
