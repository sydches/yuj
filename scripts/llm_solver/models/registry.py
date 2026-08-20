"""Model registry — alias resolution."""
from ..config import MODEL_MAP as _ALIASES


def resolve_model(name_or_alias: str) -> str:
    """Resolve short alias to full model name. Pass-through if not an alias."""
    return _ALIASES.get(name_or_alias, name_or_alias)
