"""Generate shell completion from the installed argparse command tree."""
from __future__ import annotations

import argparse
import hashlib
import json
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


SUPPORTED_SHELLS = ("bash", "zsh", "fish")


@dataclass(frozen=True, slots=True)
class OptionSpec:
    names: tuple[str, ...]
    choices: tuple[str, ...]
    takes_value: bool
    path_value: bool
    help: str


@dataclass(frozen=True, slots=True)
class NodeSpec:
    path: tuple[str, ...]
    help: str
    options: tuple[OptionSpec, ...]
    positional_choices: tuple[str, ...]
    subcommands: tuple[tuple[str, str], ...]


def _help(value: object) -> str:
    text = "" if value in {None, argparse.SUPPRESS} else str(value)
    return " ".join(text.split())


def _subparsers(parser: argparse.ArgumentParser):
    return next(
        (
            action
            for action in parser._actions
            if isinstance(action, argparse._SubParsersAction)
        ),
        None,
    )


def _command_help(action: argparse._SubParsersAction) -> dict[str, str]:
    return {
        str(choice.dest): _help(choice.help)
        for choice in action._choices_actions
    }


def _option(action: argparse.Action) -> OptionSpec:
    choices = tuple(str(value) for value in (action.choices or ()))
    return OptionSpec(
        names=tuple(str(value) for value in action.option_strings),
        choices=choices,
        takes_value=action.nargs != 0,
        path_value=action.type is Path,
        help=_help(action.help),
    )


def _node(
    parser: argparse.ArgumentParser,
    *,
    path: tuple[str, ...],
    help_text: str,
) -> NodeSpec:
    options = []
    positional_choices: list[str] = []
    nested = _subparsers(parser)
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            continue
        if action.option_strings:
            options.append(_option(action))
            continue
        positional_choices.extend(
            str(value) for value in (action.choices or ())
        )
    command_help = _command_help(nested) if nested is not None else {}
    return NodeSpec(
        path=path,
        help=help_text,
        options=tuple(options),
        positional_choices=tuple(dict.fromkeys(positional_choices)),
        subcommands=tuple(
            (name, command_help.get(name, ""))
            for name in (nested.choices if nested is not None else ())
        ),
    )


def completion_nodes(
    root_parser: argparse.ArgumentParser,
    session_parser: argparse.ArgumentParser,
) -> tuple[NodeSpec, ...]:
    """Return a normalized completion tree derived only from argparse."""
    nodes = [_node(session_parser, path=("__run__",), help_text="")]
    root_sub = _subparsers(root_parser)
    if root_sub is None:
        return tuple(nodes)
    root_help = _command_help(root_sub)

    def visit(parser: argparse.ArgumentParser, path: tuple[str, ...]) -> None:
        nodes.append(
            _node(
                parser,
                path=path,
                help_text=root_help.get(path[0], "") if len(path) == 1 else "",
            )
        )
        nested = _subparsers(parser)
        if nested is None:
            return
        for name, child in nested.choices.items():
            visit(child, (*path, str(name)))

    for name, parser in root_sub.choices.items():
        visit(parser, (str(name),))
    return tuple(nodes)


def completion_manifest(
    root_parser: argparse.ArgumentParser,
    session_parser: argparse.ArgumentParser,
) -> dict[str, object]:
    """Return the stable command surface used by every shell renderer."""
    nodes = completion_nodes(root_parser, session_parser)
    return {
        "nodes": [
            {
                "path": list(node.path),
                "help": node.help,
                "options": [
                    {
                        "names": list(option.names),
                        "choices": list(option.choices),
                        "takes_value": option.takes_value,
                        "path_value": option.path_value,
                        "help": option.help,
                    }
                    for option in node.options
                ],
                "positional_choices": list(node.positional_choices),
                "subcommands": [
                    {"name": name, "help": help_text}
                    for name, help_text in node.subcommands
                ],
            }
            for node in nodes
        ]
    }


def _surface_hash(manifest: Mapping[str, object]) -> str:
    encoded = json.dumps(
        manifest,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _path_key(path: tuple[str, ...]) -> str:
    return " ".join(path)


def _shell_words(values) -> str:
    return " ".join(shlex.quote(str(value)) for value in values)


def _commands(nodes: tuple[NodeSpec, ...]) -> tuple[str, ...]:
    return tuple(
        node.path[0]
        for node in nodes
        if node.path != ("__run__",) and len(node.path) == 1
    )


def _path_resolution_bash(nodes: tuple[NodeSpec, ...]) -> list[str]:
    lines = [
        '  path="__run__"',
        '  first="${COMP_WORDS[1]-}"',
        '  case "$first" in',
        f"    {'|'.join(_commands(nodes))}) path=\"$first\" ;;",
        "  esac",
    ]
    for node in sorted(nodes, key=lambda item: len(item.path)):
        if node.path == ("__run__",) or not node.subcommands:
            continue
        word_index = len(node.path) + 1
        parent = _path_key(node.path)
        lines.extend([
            f'  if [[ "$path" == {shlex.quote(parent)} ]]; then',
            f'    case "${{COMP_WORDS[{word_index}]-}}" in',
        ])
        for name, _help_text in node.subcommands:
            child = f"{parent} {name}"
            lines.append(f"      {name}) path={shlex.quote(child)} ;;")
        lines.extend(["    esac", "  fi"])
    return lines


def _bash_cases(nodes: tuple[NodeSpec, ...], *, kind: str) -> list[str]:
    lines: list[str] = []
    for node in nodes:
        key = _path_key(node.path)
        for option in node.options:
            if kind == "choices" and not option.choices:
                continue
            if kind == "paths" and not option.path_value:
                continue
            patterns = "|".join(
                shlex.quote(f"{key}:{name}") for name in option.names
            )
            if kind == "choices":
                rendered = shlex.quote(_shell_words(option.choices))
                lines.append(
                    f"    {patterns}) candidates={rendered} ;;"
                )
            else:
                lines.append(f"    {patterns}) _yuj_path_value=1 ;;")
    return lines


def _render_bash(nodes: tuple[NodeSpec, ...], header: str) -> str:
    lines = [
        header,
        "_yuj_completion() {",
        "  local cur prev path first candidates _yuj_path_value",
        "  COMPREPLY=()",
        '  cur="${COMP_WORDS[COMP_CWORD]}"',
        '  prev="${COMP_WORDS[COMP_CWORD-1]-}"',
        *_path_resolution_bash(nodes),
        '  case "$path:$prev" in',
        *_bash_cases(nodes, kind="choices"),
        "  esac",
        '  if [[ -n "$candidates" ]]; then',
        '    COMPREPLY=( $(compgen -W "$candidates" -- "$cur") )',
        "    return",
        "  fi",
        '  case "$path:$prev" in',
        *_bash_cases(nodes, kind="paths"),
        "  esac",
        '  if [[ -n "$_yuj_path_value" ]]; then',
        '    COMPREPLY=( $(compgen -f -- "$cur") )',
        "    return",
        "  fi",
        '  if [[ "$cur" == -* ]]; then',
        '    case "$path" in',
    ]
    for node in nodes:
        options = tuple(
            name for option in node.options for name in option.names
        )
        lines.append(
            f"      {shlex.quote(_path_key(node.path))}) "
            f"candidates={shlex.quote(_shell_words(options))} ;;"
        )
    lines.extend([
        "    esac",
        "  else",
        '    case "$path" in',
    ])
    commands = _commands(nodes)
    for node in nodes:
        values = [name for name, _help_text in node.subcommands]
        values.extend(node.positional_choices)
        if node.path == ("__run__",):
            values.extend(commands)
        if not values:
            continue
        condition = ""
        if node.path == ("__run__",):
            condition = "[[ $COMP_CWORD -le 1 ]] && "
        lines.append(
            f"      {shlex.quote(_path_key(node.path))}) {condition}"
            f"candidates={shlex.quote(_shell_words(dict.fromkeys(values)))} ;;"
        )
    lines.extend([
        "    esac",
        "  fi",
        '  COMPREPLY=( $(compgen -W "$candidates" -- "$cur") )',
        "}",
        "complete -F _yuj_completion yuj",
        "",
    ])
    return "\n".join(lines)


def _path_resolution_zsh(nodes: tuple[NodeSpec, ...]) -> list[str]:
    lines = [
        '  path="__run__"',
        '  first="${words[2]-}"',
        '  case "$first" in',
        f"    {'|'.join(_commands(nodes))}) path=\"$first\" ;;",
        "  esac",
    ]
    for node in sorted(nodes, key=lambda item: len(item.path)):
        if node.path == ("__run__",) or not node.subcommands:
            continue
        word_index = len(node.path) + 2
        parent = _path_key(node.path)
        lines.extend([
            f'  if [[ "$path" == {shlex.quote(parent)} ]]; then',
            f'    case "${{words[{word_index}]-}}" in',
        ])
        for name, _help_text in node.subcommands:
            child = f"{parent} {name}"
            lines.append(f"      {name}) path={shlex.quote(child)} ;;")
        lines.extend(["    esac", "  fi"])
    return lines


def _render_zsh(nodes: tuple[NodeSpec, ...], header: str) -> str:
    lines = [
        "#compdef yuj",
        header,
        "_yuj() {",
        "  local cur prev path first candidates _yuj_path_value",
        '  cur="${words[CURRENT]}"',
        '  prev="${words[CURRENT-1]-}"',
        *_path_resolution_zsh(nodes),
        '  case "$path:$prev" in',
        *_bash_cases(nodes, kind="choices"),
        "  esac",
        '  if [[ -n "$candidates" ]]; then',
        "    compadd -- ${(z)candidates}",
        "    return",
        "  fi",
        '  case "$path:$prev" in',
        *_bash_cases(nodes, kind="paths"),
        "  esac",
        '  if [[ -n "$_yuj_path_value" ]]; then',
        "    _files",
        "    return",
        "  fi",
        '  if [[ "$cur" == -* ]]; then',
        '    case "$path" in',
    ]
    for node in nodes:
        options = tuple(
            name for option in node.options for name in option.names
        )
        lines.append(
            f"      {shlex.quote(_path_key(node.path))}) "
            f"candidates={shlex.quote(_shell_words(options))} ;;"
        )
    lines.extend(["    esac", "  else", '    case "$path" in'])
    commands = _commands(nodes)
    for node in nodes:
        values = [name for name, _help_text in node.subcommands]
        values.extend(node.positional_choices)
        if node.path == ("__run__",):
            values.extend(commands)
        if not values:
            continue
        condition = ""
        if node.path == ("__run__",):
            condition = "[[ $CURRENT -le 2 ]] && "
        lines.append(
            f"      {shlex.quote(_path_key(node.path))}) {condition}"
            f"candidates={shlex.quote(_shell_words(dict.fromkeys(values)))} ;;"
        )
    lines.extend([
        "    esac",
        "  fi",
        "  compadd -- ${(z)candidates}",
        "}",
        "compdef _yuj yuj",
        "",
    ])
    return "\n".join(lines)


def _fish_quote(value: object) -> str:
    return "'" + str(value).replace("\\", "\\\\").replace("'", "\\'") + "'"


def _fish_condition(path: tuple[str, ...]) -> str:
    return _fish_quote(f"__yuj_completion_path_is {_path_key(path)}")


def _fish_option_flags(option: OptionSpec) -> str:
    values = []
    for name in option.names:
        if name.startswith("--"):
            values.extend(("-l", _fish_quote(name[2:])))
        elif name.startswith("-") and len(name) == 2:
            values.extend(("-s", _fish_quote(name[1:])))
        else:
            values.extend(("-o", _fish_quote(name.lstrip("-"))))
    if option.takes_value:
        values.append("-r")
    if option.path_value:
        values.append("-F")
    if option.choices:
        values.extend(("-a", _fish_quote(" ".join(option.choices))))
    if option.help:
        values.extend(("-d", _fish_quote(option.help)))
    return " ".join(values)


def _render_fish(nodes: tuple[NodeSpec, ...], header: str) -> str:
    commands = _commands(nodes)
    lines = [
        header,
        "function __yuj_completion_path",
        "    set -l tokens (commandline -opc)",
        "    if test (count $tokens) -lt 2",
        "        echo __run__",
        "        return",
        "    end",
        "    set -l first $tokens[2]",
        "    switch $first",
    ]
    top_nodes = {node.path[0]: node for node in nodes if len(node.path) == 1}
    for command in commands:
        node = top_nodes[command]
        lines.extend([
            f"        case {_fish_quote(command)}",
            f"            set -l path {_fish_quote(command)}",
        ])
        if node.subcommands:
            lines.extend([
                "            if test (count $tokens) -ge 3",
                "                switch $tokens[3]",
            ])
            for name, _help_text in node.subcommands:
                child = _fish_quote(command + " " + name)
                lines.extend([
                    f"                    case {_fish_quote(name)}",
                    f"                        set path {child}",
                ])
            lines.extend(["                end", "            end"])
        lines.append("            echo $path")
    lines.extend([
        "        case '*'",
        "            echo __run__",
        "    end",
        "end",
        "function __yuj_completion_path_is",
        "    test (__yuj_completion_path) = (string join ' ' $argv)",
        "end",
        "complete -c yuj -f",
    ])
    run_condition = _fish_condition(("__run__",))
    for command in commands:
        help_text = top_nodes[command].help
        row = (
            f"complete -c yuj -n {run_condition} -a {_fish_quote(command)}"
        )
        if help_text:
            row += f" -d {_fish_quote(help_text)}"
        lines.append(row)
    for node in nodes:
        condition = _fish_condition(node.path)
        for name, help_text in node.subcommands:
            row = f"complete -c yuj -n {condition} -a {_fish_quote(name)}"
            if help_text:
                row += f" -d {_fish_quote(help_text)}"
            lines.append(row)
        if node.positional_choices:
            lines.append(
                f"complete -c yuj -n {condition} -a "
                f"{_fish_quote(' '.join(node.positional_choices))}"
            )
        for option in node.options:
            lines.append(
                f"complete -c yuj -n {condition} {_fish_option_flags(option)}"
            )
    lines.append("")
    return "\n".join(lines)


def generate_completion(
    shell: str,
    *,
    root_parser: argparse.ArgumentParser,
    session_parser: argparse.ArgumentParser,
    version: str,
) -> str:
    """Render one sourceable script from the current installed parser."""
    if shell not in SUPPORTED_SHELLS:
        allowed = ", ".join(SUPPORTED_SHELLS)
        raise ValueError(f"unknown completion shell {shell!r}; choose {allowed}")
    nodes = completion_nodes(root_parser, session_parser)
    manifest = completion_manifest(root_parser, session_parser)
    header = (
        f"# Generated by yuj {version}; command-surface sha256="
        f"{_surface_hash(manifest)}"
    )
    if shell == "bash":
        return _render_bash(nodes, header)
    if shell == "zsh":
        return _render_zsh(nodes, header)
    return _render_fish(nodes, header)


__all__ = [
    "SUPPORTED_SHELLS",
    "completion_manifest",
    "completion_nodes",
    "generate_completion",
]
