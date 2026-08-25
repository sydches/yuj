"""Model registry — alias resolution."""
from ..config import MODEL_MAP as _ALIASES


_OPENAI_DATED_VISION_MODELS = (
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "o1",
    "o1-pro",
    "o3",
    "o3-pro",
    "o4-mini",
)


def resolve_model(name_or_alias: str) -> str:
    """Resolve short alias to full model name. Pass-through if not an alias."""
    return _ALIASES.get(name_or_alias, name_or_alias)


def model_supports_image_inputs(
    model: str,
    *,
    provider: str,
    profile_supports_image_inputs: bool,
) -> bool:
    """Resolve image capability conservatively from provider-owned model IDs."""
    if profile_supports_image_inputs:
        return True
    name = model.strip().lower()
    if provider == "anthropic":
        return name.startswith((
            "claude-3-",
            "claude-3.",
            "claude-sonnet-4",
            "claude-opus-4",
            "claude-haiku-4",
        ))
    if provider == "openai":
        return name.startswith("gpt-5") or _is_named_model_or_snapshot(
            name, _OPENAI_DATED_VISION_MODELS
        )
    return False


def _is_named_model_or_snapshot(
    name: str, model_names: tuple[str, ...]
) -> bool:
    """Match an exact model alias or its YYYY-MM-DD snapshot only."""
    for model_name in model_names:
        if name == model_name:
            return True
        prefix = f"{model_name}-"
        if not name.startswith(prefix):
            continue
        suffix = name[len(prefix):]
        if (
            len(suffix) == 10
            and suffix[:4].isdigit()
            and suffix[4] == "-"
            and suffix[5:7].isdigit()
            and suffix[7] == "-"
            and suffix[8:].isdigit()
        ):
            return True
    return False
