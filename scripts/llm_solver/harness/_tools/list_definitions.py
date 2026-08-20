"""list_definitions tool: structural outline of a single source file."""
from ...config import Config
from ._common import _path_hint, _resolve, _xml_attr, _xml_body


# Envelope schema version. Bump when the body grammar or attribute set
# changes in a way that readers need to distinguish. Version 1 has the
# current shape: status="..." [error_kind="..."] path="..."
# [count="N" surface="M"] v="1".
_LIST_DEFINITIONS_ENVELOPE_VERSION = "1"


def list_definitions(path: str, *, cwd: str, cfg: Config) -> str:
    """Return a structural outline of a single source file.

    For Python files, walks the AST and emits one line per top-level and
    nested class/function definition with file line number, indentation
    matching nesting depth, signature (or class header), and the first
    line of the docstring when present.

    Cost is dominated by ast.parse() — a few ms per file even on large
    sources. Output is bounded: the resulting string is typically
    100-300 tokens instead of the 5-20k tokens needed to read a whole
    file. Use BEFORE read() to navigate large
    files; use read(offset, limit) to fetch a specific definition once
    you know the line.

    Path resolution and outside-cwd protection mirror read(). Other
    languages (.js / .ts / .go / .rs / .java) return a clear "(language
    not supported)" message rather than a parse error.

    Disabled by ``cfg.tools_list_definitions_enabled`` — a disabled
    call returns ERROR rather than silently succeeding so accidental
    wiring is loud.
    """
    if not getattr(cfg, "tools_list_definitions_enabled", False):
        return _list_definitions_error(
            path, "disabled",
            "list_definitions tool is disabled (tools.list_definitions.enabled=false)",
        )
    try:
        abs_path = _resolve(cwd, path)
    except ValueError as e:
        return _list_definitions_error(path, "path_outside_cwd", str(e))
    if not abs_path.is_file():
        return _list_definitions_error(
            path, "not_found", f"file not found: {path}{_path_hint(cwd, path)}",
        )
    suffix = abs_path.suffix.lower()
    # `.pyi` stub files use the same Python grammar (function bodies are
    # `...`) and `ast.parse` accepts them without modification. Stubs
    # are a real navigation case (typeshed, library `.pyi` packages),
    # so accept them on the same path. `.pyx` (Cython) is NOT included
    # — `cdef`/`cpdef` are not valid Python and `ast.parse` rejects them.
    if suffix not in (".py", ".pyi"):
        return _list_definitions_error(
            path, "unsupported_suffix",
            f"list_definitions does not support {suffix!r} files yet. "
            "Currently supports: .py, .pyi. Use read() for unsupported types.",
        )
    try:
        text = abs_path.read_text(errors="replace")
    except OSError as e:
        return _list_definitions_error(path, "os_error", f"could not read {path}: {e}")
    envelope = _list_python_definitions(text, path)
    # Record the outline-vs-read savings on the success path only. Error
    # envelopes are not savings events because the model still has to recover.
    if envelope.startswith('<list_definitions status="ok"'):
        from ..savings import get_ledger
        get_ledger().record(
            bucket="outline_vs_read",
            layer="harness",
            mechanism="list_definitions",
            input_chars=len(text),
            output_chars=len(envelope),
            measure_type="exact",
            ctx={"path": path, "suffix": abs_path.suffix.lower()},
        )
    return envelope


def _list_definitions_error(path: str, error_kind: str, reason: str) -> str:
    """Wrap a list_definitions failure in the typed envelope.

    Mirrors the success-path envelope shape so downstream readers
    (classify_outcome, the model, replay tools) can branch on a single
    discriminator. Body retains the legacy `ERROR: <reason>` form so
    legacy substring matchers still work; the envelope status attribute
    is the modern path.
    """
    return (
        f'<list_definitions status="error" error_kind="{_xml_attr(error_kind)}" '
        f'path="{_xml_attr(path)}" v="{_LIST_DEFINITIONS_ENVELOPE_VERSION}">\n'
        f'ERROR: {reason}\n'
        '</list_definitions>'
    )


def _list_python_definitions(source: str, display_path: str) -> str:
    """Format a Python source's outline.

    AST-based (stdlib only — no tree-sitter dep). Two sections:

      1. Module surface: top-level imports, ImportFrom, type aliases
         (`X = ...`), assignments to all-caps NAMES (constants), and
         `__all__` if present. Lets the model see exports without
         reading the file.
      2. Definitions: FunctionDef / AsyncFunctionDef / ClassDef walked
         with nesting → indentation. Decorators are listed above each
         def so `@property` / `@classmethod` / `@dataclass` are visible
         (without these the model can't tell a property from a method
         from the outline alone).

    Signatures are reconstructed from arg lists, NOT sliced from
    source — robust against multi-line def headers.
    """
    import ast
    try:
        tree = ast.parse(source, filename=display_path)
    except SyntaxError as e:
        return _list_definitions_error(
            display_path, "syntax_error",
            f"SyntaxError in {display_path} at line {e.lineno}: {e.msg}. "
            "Cannot extract definitions from a file that does not parse.",
        )

    surface_lines: list[str] = []
    def_lines: list[str] = []

    # ── Module surface ──
    # Imports, ImportFrom, ALL_CAPS top-level assignments, simple type
    # aliases (assignments to a single Name on the LHS where the RHS is
    # a Subscript / Name / Attribute — heuristic; AnnAssign is a
    # cleaner type-alias signal).
    for child in tree.body:
        line_no = getattr(child, "lineno", 0)
        if isinstance(child, ast.Import):
            names = ", ".join(
                f"{a.name} as {a.asname}" if a.asname else a.name
                for a in child.names
            )
            surface_lines.append(f"[L{line_no:>4}] import {names}")
        elif isinstance(child, ast.ImportFrom):
            mod = ("." * (child.level or 0)) + (child.module or "")
            names = ", ".join(
                f"{a.name} as {a.asname}" if a.asname else a.name
                for a in child.names
            )
            surface_lines.append(f"[L{line_no:>4}] from {mod} import {names}")
        elif isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name):
            # Type alias-ish: `X: TypeAlias = ...` or `X: T = ...`
            tgt = child.target.id
            ann = ast.unparse(child.annotation) if child.annotation else ""
            surface_lines.append(f"[L{line_no:>4}] {tgt}: {ann}")
        elif isinstance(child, ast.Assign):
            # All-caps constants, __all__, type-alias-style assignments.
            for tgt in child.targets:
                if isinstance(tgt, ast.Name):
                    name = tgt.id
                    if name == "__all__":
                        try:
                            value_repr = ast.unparse(child.value)
                        except Exception:
                            value_repr = "..."
                        surface_lines.append(f"[L{line_no:>4}] __all__ = {value_repr[:120]}")
                    elif name.isupper() and len(name) > 1:
                        surface_lines.append(f"[L{line_no:>4}] {name} = ...")

    # ── Definitions ──
    def _format_args(args: ast.arguments) -> str:
        parts: list[str] = []
        # Positional + posonly
        all_positional = list(args.posonlyargs) + list(args.args)
        for a in all_positional:
            ann = f": {ast.unparse(a.annotation)}" if a.annotation else ""
            parts.append(f"{a.arg}{ann}")
        if args.vararg:
            ann = f": {ast.unparse(args.vararg.annotation)}" if args.vararg.annotation else ""
            parts.append(f"*{args.vararg.arg}{ann}")
        elif args.kwonlyargs:
            parts.append("*")
        for a in args.kwonlyargs:
            ann = f": {ast.unparse(a.annotation)}" if a.annotation else ""
            parts.append(f"{a.arg}{ann}")
        if args.kwarg:
            ann = f": {ast.unparse(args.kwarg.annotation)}" if args.kwarg.annotation else ""
            parts.append(f"**{args.kwarg.arg}{ann}")
        return ", ".join(parts)

    def _format_decorators(node: ast.AST, indent: str) -> list[str]:
        """One line per decorator above the def. e.g. `@property`,
        `@functools.lru_cache`, `@dataclass(frozen=True)`."""
        out: list[str] = []
        decos = getattr(node, "decorator_list", None) or []
        for d in decos:
            try:
                rendered = ast.unparse(d)
            except Exception:
                rendered = "<decorator>"
            d_line = getattr(d, "lineno", 0)
            out.append(f"[L{d_line:>4}] {indent}@{rendered}")
        return out

    def _doc_first_line(node: ast.AST) -> str:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            doc = ast.get_docstring(node)
            if doc:
                first = doc.strip().splitlines()[0] if doc.strip() else ""
                return first
        return ""

    def _walk(node: ast.AST, depth: int) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                indent = "  " * depth
                line_no = getattr(child, "lineno", 0)
                # Decorators first, indented to match the def.
                def_lines.extend(_format_decorators(child, indent))
                if isinstance(child, ast.ClassDef):
                    bases = ", ".join(ast.unparse(b) for b in child.bases) if child.bases else ""
                    sig = f"class {child.name}({bases})" if bases else f"class {child.name}"
                else:
                    prefix = "async def " if isinstance(child, ast.AsyncFunctionDef) else "def "
                    args_str = _format_args(child.args)
                    ret = f" -> {ast.unparse(child.returns)}" if child.returns else ""
                    sig = f"{prefix}{child.name}({args_str}){ret}"
                doc = _doc_first_line(child)
                doc_suffix = f"  # {doc[:80]}" if doc else ""
                def_lines.append(f"[L{line_no:>4}] {indent}{sig}{doc_suffix}")
                _walk(child, depth + 1)

    _walk(tree, depth=0)

    # ── Render ──
    n_def = len(def_lines)
    n_surface = len(surface_lines)
    if not surface_lines and not def_lines:
        return (
            f'<list_definitions status="ok" path="{_xml_attr(display_path)}" '
            f'count="0" surface="0" v="{_LIST_DEFINITIONS_ENVELOPE_VERSION}">\n'
            "(no module surface or definitions found)\n"
            "</list_definitions>"
        )
    parts: list[str] = []
    if surface_lines:
        parts.append("# module surface")
        parts.extend(surface_lines)
    if def_lines:
        if surface_lines:
            parts.append("")  # blank-line separator
        parts.append("# definitions")
        parts.extend(def_lines)
    # Body XML-escape — docstrings and decorator-arg literals can contain
    # the literal string `</list_definitions>` and would otherwise close
    # the envelope early.
    body = "\n".join(_xml_body(line) for line in parts)
    return (
        f'<list_definitions status="ok" path="{_xml_attr(display_path)}" '
        f'count="{n_def}" surface="{n_surface}" '
        f'v="{_LIST_DEFINITIONS_ENVELOPE_VERSION}">\n{body}\n'
        "</list_definitions>"
    )
