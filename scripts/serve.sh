#!/usr/bin/env bash
# Single-entrypoint server launcher. Reads a runtime TOML file and
# dispatches to the per-runtime translator (scripts/serve/<runtime>.py).
#
# Usage: scripts/serve.sh configs/runtime/<runtime>.toml
#
# Runtime is determined by [launch].runtime in the runtime file. Currently
# supported: vllm, llama_server. See docs/serving_overlay.md for the
# schema and per-runtime flag mapping.
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: $0 <runtime.toml>" >&2
    exit 2
fi
overlay="$1"
[[ -f "$overlay" ]] || { echo "runtime file not found: $overlay" >&2; exit 2; }

DIR="$(cd "$(dirname "$0")/.." && pwd)"

runtime=$(python3 -c 'import sys,tomllib; print(tomllib.load(open(sys.argv[1], "rb")).get("launch", {}).get("runtime", ""))' "$overlay")

case "$runtime" in
    vllm)
        if [[ -z "${VLLM_VENV:-}" ]]; then
            echo "[$0] error: set VLLM_VENV to a Python environment that contains vllm" >&2
            exit 2
        fi
        VENV="$VLLM_VENV"
        if [[ ! -x "$VENV/bin/python" ]]; then
            echo "[$0] error: VLLM_VENV has no executable bin/python: $VENV" >&2
            exit 2
        fi
        export PATH="$VENV/bin:$PATH"
        # Add NVIDIA libraries from the selected environment when present.
        NV_PKG_ROOT="$VENV/lib/python3.12/site-packages/nvidia"
        if [[ -d "$NV_PKG_ROOT" ]]; then
            NV_LIB_DIRS=$(find "$NV_PKG_ROOT" -maxdepth 3 -type d -name lib 2>/dev/null | tr '\n' ':')
            export LD_LIBRARY_PATH="${NV_LIB_DIRS%:}:${LD_LIBRARY_PATH:-}"
        fi
        export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
        exec "$VENV/bin/python" "$DIR/scripts/serve/vllm.py" "$overlay"
        ;;
    llama_server)
        exec python3 "$DIR/scripts/serve/llama_server.py" "$overlay"
        ;;
    "")
        echo "[$0] error: runtime file $overlay has no [launch].runtime key" >&2
        exit 2
        ;;
    *)
        echo "[$0] error: unsupported [launch].runtime=$runtime in $overlay" >&2
        exit 2
        ;;
esac
