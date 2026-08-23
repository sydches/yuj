"""Pipeline integration — system prompt, checkpoint, task enumeration, provenance."""
import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from .._shared.checkpoints import collect_pending as _collect_pending
from .._shared.paths import expand_user_path
from ..config import Config, dump_config
from .prompt_imports import DEFAULT_IMPORT_MAX_DEPTH, process_imports

# Re-export for back-compat with ``from llm_solver.harness.solver import collect_pending``.
collect_pending = _collect_pending


@dataclass(frozen=True, slots=True)
class ResolvedPromptSource:
    """One arm file after policy-bounded import processing."""

    content: str
    source: str
    source_bytes: int
    imported_bytes: int
    imports: tuple[dict[str, object], ...]

    def trace_record(self) -> dict[str, object]:
        return {
            "owner": "system_prompt",
            "source": self.source,
            "source_bytes": self.source_bytes,
            "imported_bytes": self.imported_bytes,
            "imports": [dict(node) for node in self.imports],
        }


def _safe_source_label(path: Path, roots: Sequence[Path]) -> str:
    resolved = path.resolve(strict=False)
    for root in roots:
        resolved_root = root.resolve(strict=False)
        try:
            return resolved.relative_to(resolved_root).as_posix() or "."
        except ValueError:
            continue
    return path.name or "<prompt>"


def resolve_system_prompt_source(
    system_prompt_file: Path | None,
    *,
    imports_enabled: bool = True,
    allowed_dirs: Sequence[Path] | None = None,
    max_depth: int = DEFAULT_IMPORT_MAX_DEPTH,
    unreadable_paths: Sequence[str] = (),
) -> ResolvedPromptSource | None:
    """Load one arm file and resolve imports under an explicit path policy."""
    if system_prompt_file is None:
        return None
    if not system_prompt_file.is_file():
        raise FileNotFoundError(f"System prompt file not found: {system_prompt_file}")
    raw = system_prompt_file.read_text()
    roots = tuple(allowed_dirs or (system_prompt_file.parent,))
    source = _safe_source_label(system_prompt_file, roots)
    if not imports_enabled:
        return ResolvedPromptSource(
            content=raw,
            source=source,
            source_bytes=len(raw.encode("utf-8")),
            imported_bytes=0,
            imports=(),
        )
    processed = process_imports(
        raw,
        system_prompt_file.parent,
        roots,
        max_depth=max_depth,
        source_path=system_prompt_file,
        unreadable_paths=unreadable_paths,
    )
    return ResolvedPromptSource(
        content=processed.content,
        source=source,
        source_bytes=len(raw.encode("utf-8")),
        imported_bytes=processed.imported_bytes,
        imports=tuple(processed.trace_tree()),
    )


def resolve_system_prompt_file(
    system_prompt_file: Path | None,
    *,
    imports_enabled: bool = True,
    allowed_dirs: Sequence[Path] | None = None,
    max_depth: int = DEFAULT_IMPORT_MAX_DEPTH,
    unreadable_paths: Sequence[str] = (),
) -> str | None:
    """Return the resolved arm text; retained as the public string helper."""
    resolved = resolve_system_prompt_source(
        system_prompt_file,
        imports_enabled=imports_enabled,
        allowed_dirs=allowed_dirs,
        max_depth=max_depth,
        unreadable_paths=unreadable_paths,
    )
    return resolved.content if resolved is not None else None


def assemble_system_prompt(
    header: str,
    *,
    resolved_arm: str | None = None,
    project_instructions: str = "",
) -> str:
    """Assemble resolved arm, project instructions, then harness header."""
    parts: list[str] = []
    if resolved_arm is not None:
        parts.append(resolved_arm.rstrip())
    if project_instructions.strip():
        parts.append(project_instructions.rstrip())
    parts.append(header)
    return "\n\n".join(parts)


def build_system_prompt(
    header: str,
    system_prompt_file: Path | None = None,
    *,
    project_instructions: str = "",
) -> str:
    """Assemble system prompt: optional file, project documents, and header.

    header: the harness header text (wired from cfg.system_header).
    system_prompt_file: if provided, its content is prepended to the
    header. By default the file is processed for `@path/file` import
    directives under its parent directory, with bounded depth and cycle
    detection. The harness still does not
    INTERPRET the content — it could be any protocol — it only resolves
    imports as a load-time concatenation. Project instructions are already
    resolved and bounded by the context
    layer.  They are inserted after the arm file and before the harness
    header.  When both optional inputs are absent, the returned bytes remain
    exactly the configured header.
    """
    return assemble_system_prompt(
        header,
        resolved_arm=resolve_system_prompt_file(system_prompt_file),
        project_instructions=project_instructions,
    )


def write_checkpoint(repo_dir: Path, model: str, status: str) -> None:
    """Write checkpoint.json compatible with collect_patches.sh and solve_bare.py."""
    checkpoint = {
        "status": status,
        "model": model,
        "solver": "llm_solver",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    (repo_dir / "checkpoint.json").write_text(json.dumps(checkpoint, indent=2) + "\n")


def collect_provenance(
    cfg: Config,
    profile_path: Path | None = None,
    *,
    resolved_system_prompt: str | None = None,
    run_metadata: dict | None = None,
    thinking_resolution=None,
    fallback_provenance: dict[str, object] | None = None,
) -> dict:
    """Gather reproducibility metadata for a run.

    Includes the full resolved Config so a run's exact parameters can be
    reconstructed from ``metrics.json`` after the fact (no reliance on the
    current ``config.toml`` which may have since changed).

    When the caller supplies the resolved system prompt, capture its
    sha256[:16] in `provenance.system_prompt_sha256` so a later edit to
    cfg.system_header / --system-prompt content is detectable in the
    ledger without re-resolving prompt imports against a possibly-
    moved file.
    """
    prov: dict = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": cfg.model,
        "config": dump_config(cfg),
        "pretest_enabled": True,
    }
    if thinking_resolution is not None:
        prov.update(thinking_resolution.provenance_fields())
    else:
        configured_thinking = getattr(cfg, "thinking_level", "off")
        prov.update({
            "thinking_level_requested": configured_thinking,
            "thinking_level_effective": configured_thinking,
            "thinking_level_clamped": False,
        })
    if fallback_provenance:
        prov.update(fallback_provenance)
    if run_metadata:
        # Explicit run envelope threaded from the CLI. Keep these as
        # top-level provenance fields so downstream ledgers can pin a task to
        # regime/config/runtime with simple jq paths instead of re-reading the
        # run-level session.json.
        for key in (
            "run_metadata_schema_version",
            "session_started_at",
            "run_dir",
            "context_mode",
            "system_prompt_path",
            "config_paths",
            "config_layers",
            "config_path_hashes",
            "cli_overrides",
            "resolved_config_sha256",
            "regime",
            "model_runtime",
            "model_runtime_sha256",
            "server_metadata_path",
            "server_metadata_sha256",
        ):
            if key in run_metadata:
                prov[key] = run_metadata[key]
    if resolved_system_prompt is not None:
        prov["system_prompt_sha256"] = hashlib.sha256(
            resolved_system_prompt.encode("utf-8", errors="ignore")
        ).hexdigest()[:16]
        prov["system_prompt_chars"] = len(resolved_system_prompt)

    # Harness git commit
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            prov["harness_git_commit"] = result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # llama.cpp version — path from config, not hardcoded
    llama_bin = expand_user_path(cfg.llama_server_bin)
    if llama_bin.exists():
        try:
            result = subprocess.run(
                [str(llama_bin), "--version"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                prov["llama_cpp_version"] = result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            pass

    # Profile TOML hash + name + format/canonical version. The hash
    # closes the "did the profile change between two replays" question;
    # the name + format_version + canonical_version close the "which
    # profile was active" question without re-resolving from cfg.
    if profile_path is not None and profile_path.is_file():
        prov["profile_toml_sha256"] = hashlib.sha256(
            profile_path.read_bytes()
        ).hexdigest()
        # Name comes from the parent dir of profile.toml, mirroring the
        # profile loader's convention.
        try:
            prov["profile_name"] = profile_path.parent.name
        except Exception:
            pass
        # Format / canonical version come from the profile TOML itself.
        try:
            import tomllib as _tomllib
            with open(profile_path, "rb") as _pf:
                _ptoml = _tomllib.load(_pf)
            if isinstance(_ptoml, dict):
                fmt = _ptoml.get("format_version")
                if fmt is not None:
                    prov["profile_format_version"] = fmt
                canon = _ptoml.get("canonical_version")
                if canon is not None:
                    prov["profile_canonical_version"] = canon
        except Exception:
            pass

    # Store a hash for each quirk TOML so the ledger can detect later edits
    # to bash quirks,
    # redactions / language-runner config without diffing the files.
    # Mirrors the same content stamps the runtime_envelope event already
    # carries, but at the per-task ledger surface.
    quirk_hashes: dict[str, str] = {}
    try:
        from .. import bash_quirks as _bq
        from .. import tool_quirks as _tq
        from .. import language_quirks as _lq
        candidate_paths = [
            ("bash_quirks/forbidden.toml", Path(_bq.__file__).parent / "forbidden.toml"),
            ("bash_quirks/redactions.toml", Path(_bq.__file__).parent / "redactions.toml"),
            ("bash_quirks/universal_rewrites.toml", Path(_bq.__file__).parent / "universal_rewrites.toml"),
            ("tool_quirks/glob.toml", Path(_tq.__file__).parent / "glob.toml"),
            (f"language_quirks/{cfg.analysis_task_format}.toml",
             Path(_lq.__file__).parent / f"{cfg.analysis_task_format}.toml"),
        ]
        for label, path in candidate_paths:
            if path.is_file():
                quirk_hashes[label] = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    except Exception:
        pass
    if quirk_hashes:
        prov["quirk_hashes"] = quirk_hashes

    # Search-tool binary versions. grep_files() prefers `rg` and falls
    # back to GNU grep (BRE) when rg is absent; the two have different
    # regex flavours (RE2 vs BRE), so a re-run on a different machine
    # could see different match shapes. Stamp the resolved binary
    # version here next to llama_cpp_version.
    rg_path = shutil.which("rg")
    if rg_path:
        try:
            result = subprocess.run(
                [rg_path, "--version"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                # rg --version prints multiple lines; the first is the
                # canonical "ripgrep X.Y.Z (rev abc)" identifier.
                prov["rg_version"] = result.stdout.splitlines()[0].strip()
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            pass
    else:
        prov["rg_version"] = ""  # explicit absence — fallback path active
    grep_path = shutil.which("grep")
    if grep_path:
        try:
            result = subprocess.run(
                [grep_path, "--version"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                prov["grep_version"] = result.stdout.splitlines()[0].strip()
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            pass

    return prov


def write_run_metrics(repo_dir: Path, metrics: dict, provenance: dict) -> None:
    """Write metrics.json with cost/efficiency metrics and provenance."""
    data = {"metrics": metrics, "provenance": provenance}
    (repo_dir / "metrics.json").write_text(json.dumps(data, indent=2) + "\n")
