"""Start, wait for, or stop llama-server processes.

Flags for the GGUF path, context size, and server settings come from the
profile TOML. Command options can replace the GGUF path and context size.

Usage:
    python -m scripts.llm_solver.server launch --profile qwen3.5-9b \\
        [--port 8080] [--gguf PATH] [--ctx N] [--wait] [--log FILE]
    python -m scripts.llm_solver.server stop
    python -m scripts.llm_solver.server wait [--port 8080] [--timeout 120]
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

from .._shared.paths import expand_user_path, project_root
from ..config import get_server_base_url, get_server_config, load_config
from .profile_loader import load_profile

# Server timing defaults from config.toml [server] section.
_server_cfg = get_server_config()
_health_timeout = int(_server_cfg.get("health_timeout", 2))
_health_poll_interval = int(_server_cfg.get("health_poll_interval", 2))
_launch_timeout_default = int(_server_cfg.get("launch_timeout", 120))
_stop_settle_default = int(_server_cfg.get("stop_settle", 2))


def _profiles_dir() -> Path:
    return project_root() / "profiles"


def _health_url(port: int) -> str:
    """Build the /health URL using the scheme+host from ``[server].base_url``.

    Only the port is parameterized — everything else comes from central config
    so a remote llama-server host (CI runner, separate GPU box) is respected.
    """
    parsed = urlparse(get_server_base_url())
    host = parsed.hostname or "localhost"
    scheme = parsed.scheme or "http"
    return f"{scheme}://{host}:{port}/health"


def _default_port() -> int:
    """Port from ``[server].base_url`` — single source of truth."""
    parsed = urlparse(get_server_base_url())
    return parsed.port or 8080


def _is_healthy(port: int) -> bool:
    try:
        with urllib.request.urlopen(_health_url(port), timeout=_health_timeout) as resp:
            return resp.status == 200 and b"ok" in resp.read()
    except Exception:
        return False


def cmd_launch(args: argparse.Namespace) -> int:
    profile = load_profile(args.profile, _profiles_dir())
    cfg = load_config()
    binary = str(expand_user_path(cfg.llama_server_bin))
    port = args.port if args.port is not None else _default_port()
    cmd = profile.build_launch_command(
        binary=binary,
        port=port,
        gguf_override=args.gguf,
        ctx_override=args.ctx,
    )

    log_path = args.log
    if log_path is None:
        log_fd = subprocess.DEVNULL
    else:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        log_fd = open(log_path, "ab")

    print(f"  Starting llama-server (profile={profile.name}, port={port})")
    print(f"  {' '.join(cmd)}", file=sys.stderr)

    proc = subprocess.Popen(
        cmd, stdout=log_fd, stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    print(f"  pid={proc.pid}")

    if not args.wait:
        return 0

    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            print(
                f"ERROR: llama-server exited early (rc={proc.returncode})",
                file=sys.stderr,
            )
            return 1
        if _is_healthy(port):
            print("  Server ready.")
            return 0
        time.sleep(_health_poll_interval)
    print("ERROR: llama-server did not become healthy in time", file=sys.stderr)
    return 1


def cmd_stop(args: argparse.Namespace) -> int:
    """Kill every running llama-server process. Best-effort, silent if none."""
    try:
        out = subprocess.check_output(["pgrep", "-f", "llama-server"], text=True)
    except subprocess.CalledProcessError:
        return 0  # none running
    pids = [int(p) for p in out.split() if p.strip()]
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    time.sleep(args.settle)
    return 0


def cmd_wait(args: argparse.Namespace) -> int:
    port = args.port if args.port is not None else _default_port()
    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline:
        if _is_healthy(port):
            print("  Server ready.")
            return 0
        time.sleep(_health_poll_interval)
    print("ERROR: llama-server not healthy", file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.llm_solver.server",
        description="Start, wait for, or stop llama-server processes",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_launch = sub.add_parser("launch", help="start llama-server from a profile")
    p_launch.add_argument("--profile", required=True, help="profile name")
    p_launch.add_argument("--port", type=int, default=None,
                          help="use this port instead of [server].base_url")
    p_launch.add_argument(
        "--gguf", default=None,
        help="use this GGUF after the profile supplies a model path",
    )
    p_launch.add_argument("--ctx", type=int, default=None,
                          help="use this context size")
    p_launch.add_argument("--log", default=None,
                          help="write server output to this file")
    p_launch.add_argument("--wait", action="store_true",
                          help="wait until /health returns ok")
    p_launch.add_argument("--timeout", type=int, default=_launch_timeout_default,
                          help=f"stop waiting after this many seconds (default: {_launch_timeout_default})")
    p_launch.set_defaults(func=cmd_launch)

    p_stop = sub.add_parser(
        "stop",
        help="send SIGKILL to every process whose command line matches llama-server",
    )
    p_stop.add_argument("--settle", type=int, default=_stop_settle_default,
                        help=f"wait this many seconds after the signals (default: {_stop_settle_default})")
    p_stop.set_defaults(func=cmd_stop)

    p_wait = sub.add_parser("wait", help="wait for /health to return ok")
    p_wait.add_argument("--port", type=int, default=None,
                        help="use this port instead of [server].base_url")
    p_wait.add_argument(
        "--timeout", type=int, default=_launch_timeout_default,
        help=f"stop waiting after this many seconds (default: {_launch_timeout_default})",
    )
    p_wait.set_defaults(func=cmd_wait)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
