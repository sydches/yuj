"""Runtime tool-schema validation and constrained-decoding helpers.

The harness has one model-facing tool-schema list.  This module compiles that
list once and supplies both consumers that must agree on its meaning:

* runtime argument validation before a handler is called; and
* a strict ``{"name": ..., "arguments": ...}`` tool-call contract for
  llama-server constrained decoding.

Validation failures deliberately contain field paths and JSON types, never
argument values.  The model-facing envelope begins with ``ERROR:`` so the
normal tool error ladder can count a rejected call without a special bypass.
"""
from __future__ import annotations

import copy
import itertools
import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any


SCHEMA_VALIDATION_MODES = ("off", "reject")
CONSTRAINED_DECODING_MODES = ("off", "json_schema", "grammar")
SCHEMA_REJECT_ERROR_TYPE = "tool_schema_reject"
SCHEMA_REJECT_ENVELOPE_VERSION = 1

_JSON_TYPES = frozenset(
    {"array", "boolean", "integer", "null", "number", "object", "string"}
)
_ANNOTATION_KEYWORDS = frozenset(
    {
        "$comment",
        "$id",
        "$schema",
        "default",
        "deprecated",
        "description",
        "examples",
        "format",
        "readOnly",
        "title",
        "writeOnly",
    }
)
_SUPPORTED_KEYWORDS = _ANNOTATION_KEYWORDS | frozenset(
    {
        "$defs",
        "$ref",
        "additionalProperties",
        "allOf",
        "anyOf",
        "const",
        "contains",
        "dependentRequired",
        "else",
        "enum",
        "exclusiveMaximum",
        "exclusiveMinimum",
        "if",
        "items",
        "maxContains",
        "maximum",
        "maxItems",
        "maxLength",
        "maxProperties",
        "minContains",
        "minimum",
        "minItems",
        "minLength",
        "minProperties",
        "multipleOf",
        "not",
        "oneOf",
        "pattern",
        "patternProperties",
        "prefixItems",
        "properties",
        "propertyNames",
        "required",
        "then",
        "type",
        "uniqueItems",
    }
)
_SCHEMA_MAP_KEYWORDS = frozenset(
    {"$defs", "definitions", "patternProperties", "properties"}
)
_SCHEMA_LIST_KEYWORDS = frozenset(
    {"allOf", "anyOf", "oneOf", "prefixItems"}
)
_SCHEMA_SINGLE_KEYWORDS = frozenset(
    {
        "additionalProperties",
        "contains",
        "else",
        "if",
        "items",
        "not",
        "propertyNames",
        "then",
    }
)
_MAX_GRAMMAR_OBJECT_PROPERTIES = 8


class ToolSchemaDefinitionError(ValueError):
    """A configured tool schema cannot be applied safely."""


@dataclass(frozen=True)
class SchemaViolation:
    """One stable, value-free JSON-schema validation failure."""

    path: str
    keyword: str
    message: str
    expected: str | None = None
    actual: str | None = None

    def as_dict(self) -> dict[str, str]:
        payload = {
            "path": self.path,
            "keyword": self.keyword,
            "message": self.message,
        }
        if self.expected is not None:
            payload["expected"] = self.expected
        if self.actual is not None:
            payload["actual"] = self.actual
        return payload


@dataclass(frozen=True)
class ToolArgumentValidation:
    """Validation result for one canonical tool call."""

    tool: str
    errors: tuple[SchemaViolation, ...]

    @property
    def valid(self) -> bool:
        return not self.errors

    def trace_fields(self) -> dict[str, object]:
        """Return exactly the additive fields for ``schema_reject``."""
        return {
            "tool": self.tool,
            "errors": [error.as_dict() for error in self.errors],
        }

    def error_envelope(self) -> str:
        """Render a repairable error-ladder-compatible rejection."""
        if self.valid:
            raise ValueError("a valid tool call has no rejection envelope")
        payload = {
            "error": {
                "type": SCHEMA_REJECT_ERROR_TYPE,
                "version": SCHEMA_REJECT_ENVELOPE_VERSION,
                "tool": self.tool,
                "message": "Tool arguments do not match the declared schema.",
                "errors": [error.as_dict() for error in self.errors],
            }
        }
        return "ERROR: " + json.dumps(
            payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        )


@dataclass(frozen=True)
class ValidatedDispatchResult:
    """Result of a handler call protected by schema validation."""

    result: str
    dispatched: bool
    validation: ToolArgumentValidation | None


@dataclass(frozen=True)
class ConstrainedDecodingResolution:
    """One profile-gated constrained-decoding request decision."""

    requested_mode: str
    effective_mode: str
    supports_constrained_tools: bool
    request_extra: Mapping[str, object]
    fallback_reason: str | None = None

    @property
    def enabled(self) -> bool:
        return self.effective_mode != "off"


@dataclass(frozen=True)
class ToolSchemaSet:
    """Validated parameter schemas indexed in model-facing order."""

    names: tuple[str, ...]
    _parameter_schemas: Mapping[str, object]

    @classmethod
    def from_openai_tools(cls, tool_schemas: Sequence[Mapping[str, object]]):
        if isinstance(tool_schemas, (str, bytes)) or not isinstance(
            tool_schemas, Sequence
        ):
            raise ToolSchemaDefinitionError("tool schemas must be a sequence")

        names: list[str] = []
        parameters: dict[str, object] = {}
        for index, wrapper in enumerate(tool_schemas):
            path = f"tools[{index}]"
            if not isinstance(wrapper, Mapping):
                raise ToolSchemaDefinitionError(f"{path} must be an object")
            if wrapper.get("type") != "function":
                raise ToolSchemaDefinitionError(
                    f"{path}.type must be 'function'"
                )
            function = wrapper.get("function")
            if not isinstance(function, Mapping):
                raise ToolSchemaDefinitionError(f"{path}.function must be an object")
            name = function.get("name")
            if not isinstance(name, str) or not name:
                raise ToolSchemaDefinitionError(
                    f"{path}.function.name must be a non-empty string"
                )
            if name in parameters:
                raise ToolSchemaDefinitionError(f"duplicate tool schema: {name!r}")
            schema = function.get("parameters")
            if not isinstance(schema, (Mapping, bool)):
                raise ToolSchemaDefinitionError(
                    f"{path}.function.parameters must be a JSON schema"
                )
            _validate_schema_definition(schema, f"{path}.function.parameters")
            if isinstance(schema, Mapping):
                declared_type = schema.get("type")
                if declared_type not in (None, "object"):
                    raise ToolSchemaDefinitionError(
                        f"{path}.function.parameters must describe an object"
                    )
            names.append(name)
            parameters[name] = copy.deepcopy(schema)

        if not names:
            raise ToolSchemaDefinitionError("at least one tool schema is required")
        return cls(tuple(names), MappingProxyType(parameters))

    def parameters_for(self, tool_name: str) -> object:
        try:
            return copy.deepcopy(self._parameter_schemas[tool_name])
        except KeyError as exc:
            raise ToolSchemaDefinitionError(
                f"no active schema for tool {tool_name!r}"
            ) from exc

    def validate(
        self,
        tool_name: str,
        arguments: object,
        *,
        max_errors: int = 20,
    ) -> ToolArgumentValidation:
        schema = self._parameter_schemas.get(tool_name)
        if schema is None:
            known = ", ".join(self.names)
            return ToolArgumentValidation(
                tool=tool_name,
                errors=(
                    SchemaViolation(
                        path="$",
                        keyword="tool",
                        message=f"unknown tool; active tools: {known}",
                        expected="active tool name",
                        actual="unknown tool",
                    ),
                ),
            )
        errors = validate_json_instance(
            arguments, schema, max_errors=max_errors
        )
        return ToolArgumentValidation(tool=tool_name, errors=errors)

    def constrained_json_schema(self) -> dict[str, object]:
        """Build the strict canonical single-tool-call wrapper schema."""
        branches: list[dict[str, object]] = []
        for name in self.names:
            raw_parameters = self._parameter_schemas[name]
            parameters = _strict_object_schemas(
                _dereference_schema(raw_parameters, raw_parameters)
            )
            branches.append(
                {
                    "type": "object",
                    "properties": {
                        "name": {"const": name},
                        "arguments": parameters,
                    },
                    "required": ["name", "arguments"],
                    "additionalProperties": False,
                }
            )
        return {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "Yuj tool call",
            "oneOf": branches,
        }

    def constrained_grammar(self) -> str:
        """Generate closed GBNF for the canonical tool-call wrapper."""
        return _ToolGrammarBuilder(self).build()


def normalize_schema_validation_mode(mode: object) -> str:
    return _normalize_mode(
        mode, SCHEMA_VALIDATION_MODES, "tools.schema_validation"
    )


def normalize_constrained_decoding_mode(mode: object) -> str:
    return _normalize_mode(
        mode, CONSTRAINED_DECODING_MODES, "tools.constrained_decoding"
    )


def _normalize_mode(mode: object, allowed: tuple[str, ...], path: str) -> str:
    if not isinstance(mode, str) or not mode.strip():
        raise ValueError(f"{path} must be a non-empty string")
    normalized = mode.strip().lower()
    if normalized not in allowed:
        raise ValueError(f"{path} must be one of: {', '.join(allowed)}")
    return normalized


def guarded_tool_dispatch(
    *,
    mode: object,
    schemas: ToolSchemaSet,
    tool_name: str,
    arguments: object,
    handler: Callable[[], str],
) -> ValidatedDispatchResult:
    """Call ``handler`` only when the selected validation policy permits it."""
    normalized = normalize_schema_validation_mode(mode)
    if normalized == "off":
        return ValidatedDispatchResult(handler(), True, None)

    validation = schemas.validate(tool_name, arguments)
    if not validation.valid:
        return ValidatedDispatchResult(
            validation.error_envelope(), False, validation
        )
    return ValidatedDispatchResult(handler(), True, validation)


def resolve_constrained_decoding(
    *,
    mode: object,
    schemas: ToolSchemaSet,
    supports_constrained_tools: object,
) -> ConstrainedDecodingResolution:
    """Resolve profile capability and construct llama-server request extras."""
    requested = normalize_constrained_decoding_mode(mode)
    supported = bool(supports_constrained_tools)
    if requested == "off":
        return ConstrainedDecodingResolution(
            requested, "off", supported, MappingProxyType({})
        )
    if not supported:
        return ConstrainedDecodingResolution(
            requested,
            "off",
            False,
            MappingProxyType({}),
            fallback_reason="profile_unsupported",
        )

    if requested == "json_schema":
        request_extra: dict[str, object] = {
            "json_schema": schemas.constrained_json_schema()
        }
    else:
        request_extra = {
            "grammar": schemas.constrained_grammar(),
            # llama-server uses this discriminator for a grammar that belongs
            # to the tool-call span rather than ordinary assistant content.
            "grammar_type": "tool_calls",
        }
    return ConstrainedDecodingResolution(
        requested,
        requested,
        True,
        MappingProxyType(request_extra),
    )


def attach_constrained_decoding(
    payload: Mapping[str, object],
    resolution: ConstrainedDecodingResolution,
) -> dict[str, object]:
    """Attach policy-owned fields under OpenAI SDK ``extra_body``.

    When enabled, constrained-tool policy owns ``json_schema``, ``grammar``,
    and ``grammar_type`` and removes stale copies of the alternate mode.
    Neither input is mutated.  An off/unsupported resolution is a no-op.
    """
    if not isinstance(payload, Mapping):
        raise ValueError("request payload must be a mapping")
    result = copy.deepcopy(dict(payload))
    if not resolution.enabled:
        return result
    existing = result.get("extra_body")
    if existing is not None and not isinstance(existing, Mapping):
        raise ValueError("request payload extra_body must be a mapping")
    extra = copy.deepcopy(dict(existing or {}))
    for field in ("grammar", "grammar_type", "json_schema"):
        extra.pop(field, None)
    extra.update(copy.deepcopy(dict(resolution.request_extra)))
    result["extra_body"] = extra
    return result


def validate_json_instance(
    instance: object,
    schema: object,
    *,
    max_errors: int = 20,
) -> tuple[SchemaViolation, ...]:
    """Validate an instance against Yuj's supported JSON-schema vocabulary."""
    if not isinstance(max_errors, int) or isinstance(max_errors, bool) or max_errors < 1:
        raise ValueError("max_errors must be a positive integer")
    _validate_schema_definition(schema, "schema")
    context = _ValidationContext(schema, max_errors)
    context.validate(instance, schema, "$")
    return tuple(context.errors)


class _ValidationContext:
    def __init__(self, root_schema: object, max_errors: int):
        self.root_schema = root_schema
        self.max_errors = max_errors
        self.errors: list[SchemaViolation] = []

    def add(
        self,
        path: str,
        keyword: str,
        message: str,
        *,
        expected: str | None = None,
        actual: str | None = None,
    ) -> None:
        if len(self.errors) < self.max_errors:
            self.errors.append(
                SchemaViolation(path, keyword, message, expected, actual)
            )

    def branch_errors(self, instance: object, schema: object, path: str):
        child = _ValidationContext(self.root_schema, self.max_errors)
        child.validate(instance, schema, path)
        return child.errors

    def validate(self, instance: object, schema: object, path: str) -> None:
        if len(self.errors) >= self.max_errors:
            return
        if schema is True:
            return
        if schema is False:
            self.add(path, "false_schema", "value is forbidden by the schema")
            return
        assert isinstance(schema, Mapping)

        reference = schema.get("$ref")
        if reference is not None:
            target = _resolve_local_ref(self.root_schema, reference)
            self.validate(instance, target, path)

        for branch in schema.get("allOf", ()):
            self.validate(instance, branch, path)

        any_of = schema.get("anyOf")
        if any_of is not None:
            outcomes = [self.branch_errors(instance, branch, path) for branch in any_of]
            if not any(not errors for errors in outcomes):
                self.add(path, "anyOf", "value does not match any allowed schema")
                if outcomes:
                    self.errors.extend(
                        min(outcomes, key=len)[: self.max_errors - len(self.errors)]
                    )

        one_of = schema.get("oneOf")
        if one_of is not None:
            outcomes = [self.branch_errors(instance, branch, path) for branch in one_of]
            matches = sum(not errors for errors in outcomes)
            if matches != 1:
                self.add(
                    path,
                    "oneOf",
                    "value must match exactly one allowed schema",
                    expected="exactly one schema",
                    actual=f"{matches} schemas",
                )
                if matches == 0 and outcomes:
                    self.errors.extend(
                        min(outcomes, key=len)[: self.max_errors - len(self.errors)]
                    )

        not_schema = schema.get("not")
        if not_schema is not None and not self.branch_errors(instance, not_schema, path):
            self.add(path, "not", "value matches a forbidden schema")

        if_schema = schema.get("if")
        if if_schema is not None:
            selected = "then" if not self.branch_errors(instance, if_schema, path) else "else"
            if selected in schema:
                self.validate(instance, schema[selected], path)

        if "const" in schema and not _json_equal(instance, schema["const"]):
            self.add(
                path,
                "const",
                "value does not equal the required constant",
                expected="declared constant",
                actual=_json_type_name(instance),
            )
        if "enum" in schema and not any(
            _json_equal(instance, candidate) for candidate in schema["enum"]
        ):
            self.add(
                path,
                "enum",
                "value is not one of the allowed values",
                expected=f"one of {len(schema['enum'])} values",
                actual=_json_type_name(instance),
            )

        declared_types = _declared_types(schema.get("type"))
        if declared_types and not any(
            _matches_json_type(instance, expected) for expected in declared_types
        ):
            self.add(
                path,
                "type",
                "value has the wrong JSON type",
                expected=" | ".join(declared_types),
                actual=_json_type_name(instance),
            )
            return

        if isinstance(instance, Mapping):
            self._validate_object(instance, schema, path)
        elif isinstance(instance, list):
            self._validate_array(instance, schema, path)
        elif isinstance(instance, str):
            self._validate_string(instance, schema, path)
        elif _is_json_number(instance):
            self._validate_number(instance, schema, path)

    def _validate_object(
        self, instance: Mapping[object, object], schema: Mapping[str, object], path: str
    ) -> None:
        for keyword, comparator in (
            ("minProperties", lambda count, limit: count < limit),
            ("maxProperties", lambda count, limit: count > limit),
        ):
            if keyword in schema and comparator(len(instance), schema[keyword]):
                self.add(path, keyword, f"object violates {keyword}")

        properties = schema.get("properties", {})
        patterns = schema.get("patternProperties", {})
        for required in schema.get("required", ()):
            if required not in instance:
                self.add(
                    _child_path(path, required),
                    "required",
                    "required field is missing",
                    expected="present",
                    actual="missing",
                )

        matched: set[object] = set()
        for key, value in instance.items():
            if not isinstance(key, str):
                self.add(
                    _child_path(path, key),
                    "type",
                    "JSON object field names must be strings",
                    expected="string",
                    actual=type(key).__name__,
                )
                continue
            if key in properties:
                matched.add(key)
                self.validate(value, properties[key], _child_path(path, key))
            if isinstance(key, str):
                for pattern, child_schema in patterns.items():
                    if re.search(pattern, key):
                        matched.add(key)
                        self.validate(value, child_schema, _child_path(path, key))

        additional = schema.get("additionalProperties", True)
        for key, value in instance.items():
            if key in matched:
                continue
            child_path = _child_path(path, key)
            if additional is False:
                self.add(
                    child_path,
                    "additionalProperties",
                    "field is not declared by the tool schema",
                    expected="declared field",
                    actual="additional field",
                )
            elif isinstance(additional, (Mapping, bool)) and additional is not True:
                self.validate(value, additional, child_path)

        property_names = schema.get("propertyNames")
        if property_names is not None:
            for key in instance:
                self.validate(key, property_names, _child_path(path, key))

        for key, dependencies in schema.get("dependentRequired", {}).items():
            if key not in instance:
                continue
            for dependency in dependencies:
                if dependency not in instance:
                    self.add(
                        _child_path(path, dependency),
                        "dependentRequired",
                        f"field is required when {key!r} is present",
                        expected="present",
                        actual="missing",
                    )

    def _validate_array(
        self, instance: list[object], schema: Mapping[str, object], path: str
    ) -> None:
        if "minItems" in schema and len(instance) < schema["minItems"]:
            self.add(path, "minItems", "array has too few items")
        if "maxItems" in schema and len(instance) > schema["maxItems"]:
            self.add(path, "maxItems", "array has too many items")
        if schema.get("uniqueItems"):
            for index, value in enumerate(instance):
                if any(_json_equal(value, prior) for prior in instance[:index]):
                    self.add(
                        f"{path}[{index}]", "uniqueItems", "array item is duplicated"
                    )

        prefix = schema.get("prefixItems", ())
        for index, child_schema in enumerate(prefix):
            if index < len(instance):
                self.validate(instance[index], child_schema, f"{path}[{index}]")
        items = schema.get("items")
        if items is not None:
            for index in range(len(prefix), len(instance)):
                self.validate(instance[index], items, f"{path}[{index}]")

        contains = schema.get("contains")
        if contains is not None:
            count = sum(
                not self.branch_errors(value, contains, f"{path}[{index}]")
                for index, value in enumerate(instance)
            )
            minimum = schema.get("minContains", 1)
            maximum = schema.get("maxContains")
            if count < minimum or (maximum is not None and count > maximum):
                self.add(path, "contains", "array has the wrong number of matching items")

    def _validate_string(
        self, instance: str, schema: Mapping[str, object], path: str
    ) -> None:
        if "minLength" in schema and len(instance) < schema["minLength"]:
            self.add(path, "minLength", "string is shorter than allowed")
        if "maxLength" in schema and len(instance) > schema["maxLength"]:
            self.add(path, "maxLength", "string is longer than allowed")
        if "pattern" in schema and re.search(schema["pattern"], instance) is None:
            self.add(path, "pattern", "string does not match the required pattern")

    def _validate_number(
        self, instance: int | float, schema: Mapping[str, object], path: str
    ) -> None:
        if isinstance(instance, float) and not math.isfinite(instance):
            self.add(path, "type", "number must be finite", expected="finite number", actual="non-finite number")
            return
        checks = (
            ("minimum", lambda value, bound: value < bound),
            ("maximum", lambda value, bound: value > bound),
            ("exclusiveMinimum", lambda value, bound: value <= bound),
            ("exclusiveMaximum", lambda value, bound: value >= bound),
        )
        for keyword, fails in checks:
            if keyword in schema and fails(instance, schema[keyword]):
                self.add(path, keyword, f"number violates {keyword}")
        if "multipleOf" in schema:
            quotient = instance / schema["multipleOf"]
            if not math.isclose(quotient, round(quotient), abs_tol=1e-12):
                self.add(path, "multipleOf", "number is not the required multiple")


def _validate_schema_definition(schema: object, path: str) -> None:
    if isinstance(schema, bool):
        return
    if not isinstance(schema, Mapping):
        raise ToolSchemaDefinitionError(f"{path} must be an object or boolean")
    unknown = set(schema) - _SUPPORTED_KEYWORDS - {"definitions"}
    if unknown:
        raise ToolSchemaDefinitionError(
            f"{path} uses unsupported keyword(s): {sorted(unknown)}"
        )
    if "type" in schema:
        declared = schema["type"]
        values = [declared] if isinstance(declared, str) else declared
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
            raise ToolSchemaDefinitionError(f"{path}.type must be a string or list")
        if not values or any(value not in _JSON_TYPES for value in values):
            raise ToolSchemaDefinitionError(f"{path}.type contains an unknown JSON type")
    if "required" in schema:
        required = schema["required"]
        if isinstance(required, (str, bytes)) or not isinstance(required, Sequence):
            raise ToolSchemaDefinitionError(f"{path}.required must be a list")
        if any(not isinstance(name, str) or not name for name in required):
            raise ToolSchemaDefinitionError(f"{path}.required must contain field names")
        if len(set(required)) != len(required):
            raise ToolSchemaDefinitionError(f"{path}.required contains duplicates")
    if "enum" in schema and (
        not isinstance(schema["enum"], Sequence)
        or isinstance(schema["enum"], (str, bytes))
        or not schema["enum"]
    ):
        raise ToolSchemaDefinitionError(f"{path}.enum must be a non-empty list")
    if "pattern" in schema:
        try:
            re.compile(schema["pattern"])
        except (TypeError, re.error) as exc:
            raise ToolSchemaDefinitionError(f"{path}.pattern is invalid: {exc}") from exc

    for keyword in _SCHEMA_MAP_KEYWORDS:
        if keyword not in schema:
            continue
        children = schema[keyword]
        if not isinstance(children, Mapping):
            raise ToolSchemaDefinitionError(f"{path}.{keyword} must be an object")
        for name, child in children.items():
            if not isinstance(name, str):
                raise ToolSchemaDefinitionError(f"{path}.{keyword} keys must be strings")
            if keyword == "patternProperties":
                try:
                    re.compile(name)
                except re.error as exc:
                    raise ToolSchemaDefinitionError(
                        f"{path}.{keyword} has invalid pattern {name!r}: {exc}"
                    ) from exc
            _validate_schema_definition(child, f"{path}.{keyword}.{name}")
    for keyword in _SCHEMA_LIST_KEYWORDS:
        if keyword not in schema:
            continue
        children = schema[keyword]
        if isinstance(children, (str, bytes)) or not isinstance(children, Sequence):
            raise ToolSchemaDefinitionError(f"{path}.{keyword} must be a list")
        if keyword != "prefixItems" and not children:
            raise ToolSchemaDefinitionError(f"{path}.{keyword} must not be empty")
        for index, child in enumerate(children):
            _validate_schema_definition(child, f"{path}.{keyword}[{index}]")
    for keyword in _SCHEMA_SINGLE_KEYWORDS:
        if keyword in schema:
            _validate_schema_definition(schema[keyword], f"{path}.{keyword}")


def _resolve_local_ref(root: object, reference: object) -> object:
    if not isinstance(reference, str) or not reference.startswith("#"):
        raise ToolSchemaDefinitionError("only local JSON-schema references are supported")
    if reference == "#":
        return root
    if not reference.startswith("#/"):
        raise ToolSchemaDefinitionError(f"invalid local JSON-schema reference: {reference!r}")
    current = root
    for raw_part in reference[2:].split("/"):
        part = raw_part.replace("~1", "/").replace("~0", "~")
        if not isinstance(current, Mapping) or part not in current:
            raise ToolSchemaDefinitionError(f"unresolved JSON-schema reference: {reference!r}")
        current = current[part]
    return current


def _dereference_schema(schema: object, root: object) -> object:
    """Return a deep copy with local references resolved in their old root."""
    if isinstance(schema, bool):
        return schema
    assert isinstance(schema, Mapping)
    if "$ref" in schema:
        target = _dereference_schema(_resolve_local_ref(root, schema["$ref"]), root)
        siblings = {key: value for key, value in schema.items() if key != "$ref"}
        if not siblings:
            return target
        if target is False:
            return False
        if target is True:
            target = {}
        assert isinstance(target, Mapping)
        merged = copy.deepcopy(dict(target))
        merged.update(_dereference_schema(siblings, root))
        return merged

    result = copy.deepcopy(dict(schema))
    for keyword in _SCHEMA_MAP_KEYWORDS:
        if keyword in result:
            result[keyword] = {
                key: _dereference_schema(value, root)
                for key, value in result[keyword].items()
            }
    for keyword in _SCHEMA_LIST_KEYWORDS:
        if keyword in result:
            result[keyword] = [
                _dereference_schema(value, root) for value in result[keyword]
            ]
    for keyword in _SCHEMA_SINGLE_KEYWORDS:
        if keyword in result:
            result[keyword] = _dereference_schema(result[keyword], root)
    return result


def _declared_types(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(value)


def _matches_json_type(value: object, expected: str) -> bool:
    if expected == "null":
        return value is None
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return _is_json_number(value) and (
            not isinstance(value, float) or math.isfinite(value)
        )
    if expected == "string":
        return isinstance(value, str)
    if expected == "array":
        return isinstance(value, list)
    if expected == "object":
        return isinstance(value, Mapping)
    return False


def _is_json_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _json_type_name(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, Mapping):
        return "object"
    return type(value).__name__


def _json_equal(left: object, right: object) -> bool:
    if _json_type_name(left) != _json_type_name(right):
        return False
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return left.keys() == right.keys() and all(
            _json_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _json_equal(a, b) for a, b in zip(left, right)
        )
    return left == right


def _child_path(parent: str, key: object) -> str:
    if isinstance(key, str) and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
        return f"{parent}.{key}"
    return f"{parent}[{json.dumps(key, ensure_ascii=False)}]"


def _strict_object_schemas(schema: object) -> object:
    if isinstance(schema, bool):
        return schema
    assert isinstance(schema, Mapping)
    result = copy.deepcopy(dict(schema))
    for keyword in _SCHEMA_MAP_KEYWORDS:
        if keyword in result:
            result[keyword] = {
                key: _strict_object_schemas(value)
                for key, value in result[keyword].items()
            }
    for keyword in _SCHEMA_LIST_KEYWORDS:
        if keyword in result:
            result[keyword] = [
                _strict_object_schemas(value) for value in result[keyword]
            ]
    for keyword in _SCHEMA_SINGLE_KEYWORDS:
        if keyword in result and keyword != "additionalProperties":
            result[keyword] = _strict_object_schemas(result[keyword])
    declared = _declared_types(result.get("type"))
    if "object" in declared or "properties" in result or "required" in result:
        result.setdefault("type", "object")
        result.setdefault("additionalProperties", False)
    return result


class _ToolGrammarBuilder:
    """Small strict-schema to GBNF compiler for tool parameter schemas."""

    def __init__(self, schemas: ToolSchemaSet):
        self.schemas = schemas
        self.rules: dict[str, str] = {}
        self._counter = itertools.count(1)

    def build(self) -> str:
        call_rules: list[str] = []
        for name in self.schemas.names:
            slug = _rule_slug(name)
            raw_parameters = self.schemas._parameter_schemas[name]
            args_rule = self._schema_rule(
                _strict_object_schemas(
                    _dereference_schema(raw_parameters, raw_parameters)
                ),
                f"args-{slug}",
            )
            call_rule = self._unique_rule_name(f"call-{slug}")
            name_member = self._member_literal("name", _gbnf_json_value(name))
            args_member = self._member_rule("arguments", args_rule)
            body = (
                f'{_gbnf_literal("{")} ws '
                f'(({name_member} ws {_gbnf_literal(",")} ws {args_member}) | '
                f'({args_member} ws {_gbnf_literal(",")} ws {name_member})) '
                f'ws {_gbnf_literal("}")}'
            )
            self.rules[call_rule] = body
            call_rules.append(call_rule)

        lines = [f"root ::= ws ({' | '.join(call_rules)}) ws"]
        lines.extend(f"{name} ::= {body}" for name, body in self.rules.items())
        lines.extend(
            [
                r'json-value ::= json-object | json-array | json-string | json-number | json-boolean | json-null',
                r'json-object ::= "{" ws (json-string ws ":" ws json-value (ws "," ws json-string ws ":" ws json-value)*)? ws "}"',
                r'json-array ::= "[" ws (json-value (ws "," ws json-value)*)? ws "]"',
                r'json-string ::= "\"" json-char* "\""',
                r'json-char ::= [^"\\\x7F\x00-\x1F] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F]{4})',
                r'json-number ::= "-"? ("0" | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?',
                r'json-integer ::= "-"? ("0" | [1-9] [0-9]*)',
                r'json-boolean ::= "true" | "false"',
                r'json-null ::= "null"',
                r'ws ::= [ \t\n\r]*',
            ]
        )
        grammar = "\n".join(lines) + "\n"
        _validate_grammar_references(grammar)
        return grammar

    def _schema_rule(self, schema: object, hint: str) -> str:
        if schema is True:
            return "json-value"
        if schema is False:
            raise ToolSchemaDefinitionError(
                "a false schema cannot produce a constrained tool argument"
            )
        assert isinstance(schema, Mapping)

        unsupported = set(schema) & {
            "allOf",
            "contains",
            "dependentRequired",
            "else",
            "exclusiveMaximum",
            "exclusiveMinimum",
            "if",
            "maxContains",
            "maximum",
            "maxItems",
            "maxLength",
            "maxProperties",
            "minContains",
            "minimum",
            "minItems",
            "minLength",
            "minProperties",
            "multipleOf",
            "not",
            "pattern",
            "patternProperties",
            "prefixItems",
            "propertyNames",
            "then",
            "uniqueItems",
        }
        if unsupported:
            raise ToolSchemaDefinitionError(
                "GBNF generation does not support schema constraint(s): "
                + ", ".join(sorted(unsupported))
            )
        if "const" in schema:
            return self._literal_rule(schema["const"], hint)
        if "enum" in schema:
            name = self._unique_rule_name(hint)
            self.rules[name] = " | ".join(
                _gbnf_json_value(value) for value in schema["enum"]
            )
            return name
        for keyword in ("oneOf", "anyOf"):
            if keyword in schema:
                name = self._unique_rule_name(hint)
                branches = [
                    self._schema_rule(branch, f"{name}-option")
                    for branch in schema[keyword]
                ]
                self.rules[name] = " | ".join(branches)
                return name

        declared = _declared_types(schema.get("type"))
        if len(declared) > 1:
            name = self._unique_rule_name(hint)
            branches = [
                self._schema_rule({**schema, "type": kind}, f"{name}-{kind}")
                for kind in declared
            ]
            self.rules[name] = " | ".join(branches)
            return name
        kind = declared[0] if declared else None
        if kind is None:
            if "properties" in schema or "required" in schema:
                kind = "object"
            elif "items" in schema:
                kind = "array"
            else:
                return "json-value"
        if kind == "string":
            return "json-string"
        if kind == "integer":
            return "json-integer"
        if kind == "number":
            return "json-number"
        if kind == "boolean":
            return "json-boolean"
        if kind == "null":
            return "json-null"
        if kind == "array":
            name = self._unique_rule_name(hint)
            item_rule = self._schema_rule(schema.get("items", True), f"{name}-item")
            self.rules[name] = (
                f'{_gbnf_literal("[")} ws '
                f'({item_rule} (ws {_gbnf_literal(",")} ws {item_rule})*)? '
                f'ws {_gbnf_literal("]")}'
            )
            return name
        if kind == "object":
            return self._object_rule(schema, hint)
        raise ToolSchemaDefinitionError(f"unsupported grammar JSON type: {kind!r}")

    def _object_rule(self, schema: Mapping[str, object], hint: str) -> str:
        additional = schema.get("additionalProperties", True)
        if additional is not False:
            raise ToolSchemaDefinitionError(
                "GBNF object schemas must set additionalProperties=false"
            )
        properties = schema.get("properties", {})
        required = tuple(schema.get("required", ()))
        missing = set(required) - set(properties)
        if missing:
            raise ToolSchemaDefinitionError(
                f"required grammar properties are undeclared: {sorted(missing)}"
            )
        if len(properties) > _MAX_GRAMMAR_OBJECT_PROPERTIES:
            raise ToolSchemaDefinitionError(
                "GBNF object has too many properties for exhaustive ordering: "
                f"{len(properties)} > {_MAX_GRAMMAR_OBJECT_PROPERTIES}"
            )

        name = self._unique_rule_name(hint)
        value_rules = {
            key: self._schema_rule(value, f"{name}-{_rule_slug(key)}")
            for key, value in properties.items()
        }
        optional = tuple(key for key in properties if key not in required)
        sequences: list[str] = []
        has_empty = False
        for count in range(len(optional) + 1):
            for selected in itertools.combinations(optional, count):
                present = tuple(required) + selected
                if not present:
                    has_empty = True
                    continue
                for ordering in itertools.permutations(present):
                    members = [
                        self._member_rule(key, value_rules[key]) for key in ordering
                    ]
                    sequences.append(
                        f' ws {_gbnf_literal(",")} ws '.join(members)
                    )
        sequences = list(dict.fromkeys(sequences))
        open_object = f'{_gbnf_literal("{")} ws'
        close_object = f'ws {_gbnf_literal("}")}'
        if sequences:
            members_name = self._unique_rule_name(f"{name}-members")
            self.rules[members_name] = " | ".join(sequences)
            optional_members = f"({members_name})?" if has_empty else members_name
            self.rules[name] = f"{open_object} {optional_members} {close_object}"
        else:
            self.rules[name] = f"{open_object} {close_object}"
        return name

    def _literal_rule(self, value: object, hint: str) -> str:
        name = self._unique_rule_name(hint)
        self.rules[name] = _gbnf_json_value(value)
        return name

    def _member_rule(self, key: str, value_rule: str) -> str:
        return (
            f'{_gbnf_literal(json.dumps(key, ensure_ascii=True))} '
            f'ws {_gbnf_literal(":")} ws {value_rule}'
        )

    def _member_literal(self, key: str, value_literal: str) -> str:
        return (
            f'{_gbnf_literal(json.dumps(key, ensure_ascii=True))} '
            f'ws {_gbnf_literal(":")} ws {value_literal}'
        )

    def _unique_rule_name(self, hint: str) -> str:
        base = _rule_slug(hint)
        name = base
        while name in self.rules:
            name = f"{base}-{next(self._counter)}"
        return name


def _gbnf_json_value(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=True, allow_nan=False, separators=(",", ":")
    )
    return _gbnf_literal(encoded)


def _gbnf_literal(value: str) -> str:
    return json.dumps(value, ensure_ascii=True)


def _rule_slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9-]+", "-", value.lower()).strip("-")
    if not slug:
        slug = "rule"
    if not slug[0].isalpha():
        slug = "rule-" + slug
    return slug


def _validate_grammar_references(grammar: str) -> None:
    definitions = {
        line.split("::=", 1)[0].strip()
        for line in grammar.splitlines()
        if "::=" in line
    }
    if "root" not in definitions:
        raise ToolSchemaDefinitionError("generated grammar has no root rule")
    # Remove string terminals, character classes, and comments before finding
    # rule identifiers.  This is a closed-reference check, not a GBNF parser.
    bodies = "\n".join(
        line.split("::=", 1)[1] for line in grammar.splitlines() if "::=" in line
    )
    bodies = re.sub(r'"(?:\\.|[^"\\])*"', " ", bodies)
    bodies = re.sub(r"\[(?:\\.|[^\]\\])*\]", " ", bodies)
    references = set(re.findall(r"\b[a-z][a-z0-9-]*\b", bodies))
    unresolved = references - definitions
    if unresolved:
        raise ToolSchemaDefinitionError(
            f"generated grammar has unresolved rules: {sorted(unresolved)}"
        )


__all__ = [
    "CONSTRAINED_DECODING_MODES",
    "SCHEMA_REJECT_ENVELOPE_VERSION",
    "SCHEMA_REJECT_ERROR_TYPE",
    "SCHEMA_VALIDATION_MODES",
    "ConstrainedDecodingResolution",
    "SchemaViolation",
    "ToolArgumentValidation",
    "ToolSchemaDefinitionError",
    "ToolSchemaSet",
    "ValidatedDispatchResult",
    "attach_constrained_decoding",
    "guarded_tool_dispatch",
    "normalize_constrained_decoding_mode",
    "normalize_schema_validation_mode",
    "resolve_constrained_decoding",
    "validate_json_instance",
]
