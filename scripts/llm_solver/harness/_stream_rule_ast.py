"""Lazy structural pattern matching for one stream snapshot."""
from __future__ import annotations

import importlib
from functools import lru_cache
from pathlib import Path
from typing import Sequence

from ._stream_rule_loader import StreamRuleError, _METAVAR_RE, _META_PREFIX

_LANGUAGE_BY_SUFFIX = {
    ".cjs": "javascript",
    ".go": "go",
    ".java": "java",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".py": "python",
    ".pyi": "python",
    ".rs": "rust",
    ".ts": "typescript",
    ".tsx": "tsx",
}
_GRAMMARS = {
    "python": ("tree_sitter_python", "language"),
    "javascript": ("tree_sitter_javascript", "language"),
    "typescript": ("tree_sitter_typescript", "language_typescript"),
    "tsx": ("tree_sitter_typescript", "language_tsx"),
    "go": ("tree_sitter_go", "language"),
    "rust": ("tree_sitter_rust", "language"),
    "java": ("tree_sitter_java", "language"),
}


@lru_cache(maxsize=16)
def _parser_for(language: str):
    try:
        module_name, function_name = _GRAMMARS[language]
        module = importlib.import_module(module_name)
        from tree_sitter import Language, Parser
        grammar = Language(getattr(module, function_name)())
        try:
            parser = Parser(grammar)
        except TypeError:  # tree-sitter < 0.25
            parser = Parser()
            if hasattr(parser, "set_language"):
                parser.set_language(grammar)
            else:
                parser.language = grammar
        return parser
    except (KeyError, ImportError, AttributeError) as exc:
        raise StreamRuleError(
            f"stream-rule structural backend unavailable for {language!r}; "
            "reinstall Yuj with its tree-sitter dependencies"
        ) from exc


def _node_text(node, source: bytes) -> str:
    return source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _match_ast_node(pattern_node, candidate_node, pattern: bytes, candidate: bytes,
                    bindings: dict[str, str]) -> bool:
    pattern_text = _node_text(pattern_node, pattern)
    if pattern_node.type == "identifier" and pattern_text.startswith(_META_PREFIX):
        name = pattern_text[len(_META_PREFIX):]
        value = _node_text(candidate_node, candidate)
        prior = bindings.get(name)
        if prior is not None:
            return prior == value
        bindings[name] = value
        return True
    if pattern_node.type != candidate_node.type:
        return False
    pattern_children = list(pattern_node.children)
    candidate_children = list(candidate_node.children)
    if len(pattern_children) != len(candidate_children):
        return False
    if not pattern_children:
        return pattern_text == _node_text(candidate_node, candidate)
    return all(
        _match_ast_node(p_child, c_child, pattern, candidate, bindings)
        for p_child, c_child in zip(pattern_children, candidate_children)
    )


def _walk_nodes(node):
    yield node
    for child in node.children:
        yield from _walk_nodes(child)


@lru_cache(maxsize=256)
def _compiled_ast_pattern(language: str, source_pattern: str):
    parser = _parser_for(language)
    substituted = _METAVAR_RE.sub(
        lambda match: _META_PREFIX + match.group(1), source_pattern
    )
    raw = substituted.encode("utf-8")
    tree = parser.parse(raw)
    root = tree.root_node
    if root.has_error:
        raise StreamRuleError(
            f"invalid astCondition {source_pattern!r} for {language}: parse error"
        )
    node = root.named_children[0] if len(root.named_children) == 1 else root
    return raw, node


def _ast_offset(
    snapshot: str, path: str, patterns: Sequence[str], *,
    parsed_by_language: dict | None = None,
) -> int | None:
    language = _LANGUAGE_BY_SUFFIX.get(Path(path).suffix.lower())
    if language is None:
        return None
    parsed = None if parsed_by_language is None else parsed_by_language.get(language)
    if parsed is None:
        raw = snapshot.encode("utf-8")
        tree = _parser_for(language).parse(raw)
        if parsed_by_language is not None:
            parsed_by_language[language] = (raw, tree)
    else:
        raw, tree = parsed
    for source_pattern in patterns:
        pattern, pattern_node = _compiled_ast_pattern(language, source_pattern)
        for candidate_node in _walk_nodes(tree.root_node):
            if _match_ast_node(pattern_node, candidate_node, pattern, raw, {}):
                return len(raw[:candidate_node.start_byte].decode("utf-8", errors="ignore"))
    return None
