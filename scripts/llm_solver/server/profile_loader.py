"""Profile loader — resolve, inherit, and build normalize/denormalize pipelines.

Loading order:
  1. Resolve profile directory via fallback chain: exact name → family → _base
  2. Load inherited profile's rules and modules first
  3. Load this profile's rules and modules
  4. Build callable pipelines: normalize_pipeline, denormalize_pipeline
"""
import importlib.util
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .._shared.toml_compat import tomllib
from .request_controls import validate_reasoning_levels
from .rules_engine import (
    RuleFn,
    apply_normalize_rules,
    parse_denormalize_rules,
    parse_normalize_rules,
)

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Profile:
    """Loaded profile with ready-to-call pipelines."""

    name: str
    family: str
    inherits: str
    format_version: int
    canonical_version: str

    # Model metadata
    context_size: int
    chat_template: str
    supports_tool_calls: bool
    supports_system_role: bool

    # Token estimation
    token_method: str
    tokenizer_path: str

    # Capacity knobs
    preamble: str
    max_tools: int
    simplify_schemas: bool
    reasoning_levels: dict[str, dict[str, object]] = field(default_factory=dict)

    # Server launch config
    server_model_path: str = ""
    server_ctx_size: int = 8192
    server_n_gpu_layers: int = 99  # 99 = "offload everything" (llama-server convention)
    server_flash_attn: bool = True
    server_cache_type_k: str = "q8_0"
    server_cache_type_v: str = "q8_0"
    server_jinja: bool = True
    server_reasoning_mode: str = "none"
    server_reasoning_disable_flag: str = ""
    server_extra_flags: str = ""

    # Directory the profile was loaded from (for resolving sibling files like chat_template.jinja).
    profile_dir: Path | None = None

    # Denormalize config
    denormalize_config: dict = field(default_factory=dict)

    # Pre-resolved system-prompt strategy. Derived from denormalize_config at
    # load time so the hot path (denormalize_messages, called once per chat())
    # skips the .get().get() dict lookups. "native" | "fold_into_user" |
    # "prefix_user".
    system_prompt_strategy: str = "native"

    # Callable pipelines (not frozen-safe, but we use field)
    _normalize_rules: list[RuleFn] = field(default_factory=list, repr=False)
    _normalize_modules: list[Callable] = field(default_factory=list, repr=False)
    _denormalize_modules: list[Callable] = field(default_factory=list, repr=False)

    def normalize(self, response: dict) -> dict:
        """Run full normalize pipeline: rules then modules."""
        response = apply_normalize_rules(self._normalize_rules, response)
        for mod_fn in self._normalize_modules:
            response = mod_fn(response)
        return response

    def denormalize_messages(self, messages: list[dict]) -> list[dict]:
        """Run full denormalize pipeline on messages.

        Uses the pre-resolved system_prompt_strategy (set at profile load
        time) so the hot path avoids two dict lookups per chat() call.
        """
        strategy = self.system_prompt_strategy
        if not self.supports_system_role and strategy == "native":
            # Capability fallback: if the profile says system role is not
            # supported and no explicit rewrite strategy is configured,
            # fold system content into the first user message.
            strategy = "fold_into_user"
        if strategy == "fold_into_user":
            messages = _fold_system_into_user(messages)
        elif strategy == "prefix_user":
            messages = _prefix_system_to_user(messages)
        # "native" = pass-through

        for mod_fn in self._denormalize_modules:
            messages = mod_fn(messages)
        return messages

    def estimate_tokens(self, messages: list[dict]) -> int:
        """Estimate token count for a message list."""
        if self.token_method == "chars_div_4":
            total = sum(len(str(m.get("content", ""))) for m in messages)
            return total // 4
        raise ValueError(f"Unknown token method: {self.token_method}")

    def build_launch_command(
        self,
        binary: str,
        port: int = 8080,
        gguf_override: str | None = None,
        ctx_override: int | None = None,
    ) -> list[str]:
        """Build llama-server launch command from ``[server]`` config.

        ``binary`` is required — pass the resolved path from
        ``cfg.llama_server_bin``. No default: the central config is the single
        source of truth for where llama-server lives.

        ``gguf_override`` lets a runner pick a different GGUF (different quant)
        while reusing the rest of the profile. ``ctx_override`` does the same
        for context size. Both are optional.
        """
        if not self.server_model_path:
            raise ValueError(f"Profile '{self.name}' has no server.model_path")
        model_path = str(Path(gguf_override or self.server_model_path).expanduser())
        ctx_size = ctx_override if ctx_override is not None else self.server_ctx_size
        cmd = [
            binary,
            "--model", model_path,
            "--port", str(port),
            "--n-gpu-layers", str(self.server_n_gpu_layers),
            "--ctx-size", str(ctx_size),
            "--cache-type-k", self.server_cache_type_k,
            "--cache-type-v", self.server_cache_type_v,
        ]
        if self.server_flash_attn:
            cmd.extend(["--flash-attn", "on"])
        if self.server_jinja:
            cmd.append("--jinja")
        if self.server_reasoning_disable_flag:
            cmd.extend(self.server_reasoning_disable_flag.split())
        if self.server_extra_flags:
            cmd.extend(self.server_extra_flags.split())

        # Per-profile chat template override (same directory as profile.toml).
        if self.profile_dir is not None:
            template = self.profile_dir / "chat_template.jinja"
            if template.is_file():
                cmd.extend(["--chat-template-file", str(template)])
        return cmd


def _fold_system_into_user(messages: list[dict]) -> list[dict]:
    """Move system prompt content into the first user message."""
    result = []
    system_content = ""
    for msg in messages:
        if msg["role"] == "system":
            system_content = msg["content"]
            continue
        if msg["role"] == "user" and system_content:
            msg = dict(msg)
            msg["content"] = f"{system_content}\n\n{msg['content']}"
            system_content = ""
        result.append(msg)
    return result


def _prefix_system_to_user(messages: list[dict]) -> list[dict]:
    """Replace system message with a user message containing the system content."""
    result = []
    for msg in messages:
        if msg["role"] == "system":
            result.append({"role": "user", "content": msg["content"]})
        else:
            result.append(msg)
    return result


def _load_toml(path: Path) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def _load_code_module(
    path: Path, configure_with: dict | None = None,
) -> Callable:
    """Load a .py code module and return its apply() function.

    When ``configure_with`` is supplied and the module exports a
    ``configure(dict)`` function, call it before returning ``apply``.
    This is the injection point for declarative per-module config
    (e.g. ``[behavioral]`` sections in profile.toml) — the module
    itself stays free of filesystem / import side effects so the
    security validator (``server/security.py``) remains tight.
    """
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if configure_with is not None and hasattr(mod, "configure"):
        mod.configure(configure_with)
    if not hasattr(mod, "apply"):
        raise AttributeError(f"Module {path} missing required apply() function")
    return mod.apply


def _resolve_profile_dir(name: str, profiles_dir: Path) -> Path:
    """Resolve profile name to directory via fallback chain.

    exact name → family scan → _base
    """
    # Exact match
    exact = profiles_dir / name
    if exact.is_dir() and (exact / "profile.toml").is_file():
        return exact

    # Family match: scan all profiles for matching family field.
    # An ambiguous family (several versions declare it) is an error, not
    # a filesystem-order lottery — members can differ in context size.
    family_matches = []
    for candidate in sorted(profiles_dir.iterdir()):
        if not candidate.is_dir() or candidate.name.startswith("_"):
            continue
        toml_path = candidate / "profile.toml"
        if not toml_path.is_file():
            continue
        data = _load_toml(toml_path)
        if data.get("profile", {}).get("family") == name:
            family_matches.append(candidate)
    if len(family_matches) > 1:
        raise ValueError(
            f"Profile family '{name}' is ambiguous: "
            f"{[c.name for c in family_matches]} all declare it. "
            f"Name the profile explicitly."
        )
    if family_matches:
        candidate = family_matches[0]
        log.info("No profile for '%s', falling back to family match '%s'", name, candidate.name)
        return candidate

    # _base fallback
    base = profiles_dir / "_base"
    if base.is_dir():
        log.info("No profile for '%s', falling back to _base", name)
        return base

    raise FileNotFoundError(f"No profile found for '{name}' and no _base profile exists")


def _load_profile_data(profile_dir: Path) -> dict:
    """Load profile.toml from a profile directory."""
    return _load_toml(profile_dir / "profile.toml")


def _collect_profile_chain(profile_dir: Path, profiles_dir: Path) -> list[Path]:
    """Return the inheritance chain from root parent to the requested profile."""
    chain: list[Path] = []
    seen: dict[Path, int] = {}
    current = profile_dir

    while True:
        if current in seen:
            loop = chain[seen[current]:] + [current]
            names = " -> ".join(p.name for p in loop)
            raise ValueError(f"Profile inheritance cycle detected: {names}")

        seen[current] = len(chain)
        chain.append(current)

        data = _load_profile_data(current)
        inherits = data.get("profile", {}).get("inherits", "")
        if not inherits or inherits == current.name:
            break

        parent_dir = profiles_dir / inherits
        if not parent_dir.is_dir() or not (parent_dir / "profile.toml").is_file():
            raise FileNotFoundError(
                f"Profile '{current.name}' inherits missing profile '{inherits}'"
            )
        current = parent_dir

    chain.reverse()
    return chain


def load_profile(name: str, profiles_dir: Path) -> Profile:
    """Load a profile by name with full inheritance resolution.

    Returns a Profile with ready-to-call normalize/denormalize pipelines.
    """
    profile_dir = _resolve_profile_dir(name, profiles_dir)
    chain = _collect_profile_chain(profile_dir, profiles_dir)

    # Resolve inheritance chain
    normalize_rules: list[RuleFn] = []
    normalize_modules: list[Callable] = []
    denormalize_config: dict = {}
    denormalize_modules: list[Callable] = []

    prof: dict = {}
    model: dict = {}
    tokens: dict = {}
    capacity: dict = {}
    server: dict = {}
    reasoning_levels: dict[str, dict[str, object]] = {}

    for current_dir in chain:
        data = _load_profile_data(current_dir)
        current_prof = data.get("profile", {})
        if current_prof.get("format_version", 1) != 1:
            raise ValueError(
                f"Unknown profile format_version: {current_prof.get('format_version')}"
            )

        prof.update(current_prof)
        model.update(data.get("model", {}))
        tokens.update(data.get("tokens", {}))
        capacity.update(data.get("capacity", {}))
        server.update(data.get("server", {}))
        reasoning_levels.update(data.get("reasoning_levels", {}))

        norm_section = data.get("normalize", {})
        for rf in norm_section.get("rules", []):
            rules_path = current_dir / "normalize" / rf
            if rules_path.is_file():
                normalize_rules.extend(parse_normalize_rules(rules_path))

        for mf in norm_section.get("modules", []):
            mod_path = current_dir / "normalize" / mf
            if mod_path.is_file():
                normalize_modules.append(_load_code_module(mod_path))

        denorm_section = data.get("denormalize", {})
        for rf in denorm_section.get("rules", []):
            rules_path = current_dir / "denormalize" / rf
            if rules_path.is_file():
                denormalize_config.update(parse_denormalize_rules(rules_path))

        behavioral_section = data.get("behavioral", {}) or {}
        for mf in denorm_section.get("modules", []):
            mod_path = current_dir / "denormalize" / mf
            if mod_path.is_file():
                denormalize_modules.append(
                    _load_code_module(
                        mod_path,
                        configure_with=behavioral_section,
                    )
                )

    inherits = prof.get("inherits", "")
    reasoning_levels = validate_reasoning_levels(reasoning_levels)

    return Profile(
        name=prof.get("name", profile_dir.name),
        family=prof.get("family", ""),
        inherits=inherits,
        format_version=prof.get("format_version", 1),
        canonical_version=prof.get("canonical_version", "openai-v1"),
        context_size=model.get("context_size", 40960),
        chat_template=model.get("chat_template", "chatml"),
        supports_tool_calls=model.get("supports_tool_calls", True),
        supports_system_role=model.get("supports_system_role", True),
        token_method=tokens.get("method", "chars_div_4"),
        tokenizer_path=tokens.get("tokenizer", ""),
        preamble=capacity.get("preamble", ""),
        max_tools=capacity.get("max_tools", 6),
        simplify_schemas=capacity.get("simplify_schemas", False),
        reasoning_levels=reasoning_levels,
        server_model_path=server.get("model_path", ""),
        server_ctx_size=server.get("ctx_size", 8192),
        server_n_gpu_layers=server.get("n_gpu_layers", 99),
        server_flash_attn=server.get("flash_attn", True),
        server_cache_type_k=server.get("cache_type_k", "q8_0"),
        server_cache_type_v=server.get("cache_type_v", "q8_0"),
        server_jinja=server.get("jinja", True),
        server_reasoning_mode=server.get("reasoning_mode", "none"),
        server_reasoning_disable_flag=server.get("reasoning_disable_flag", ""),
        server_extra_flags=server.get("extra_flags", ""),
        profile_dir=profile_dir,
        denormalize_config=denormalize_config,
        system_prompt_strategy=denormalize_config.get("system_prompt", {}).get(
            "strategy", "native"
        ),
        _normalize_rules=normalize_rules,
        _normalize_modules=normalize_modules,
        _denormalize_modules=denormalize_modules,
    )
