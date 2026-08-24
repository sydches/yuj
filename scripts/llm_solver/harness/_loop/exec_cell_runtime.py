"""Session-owned policy bridge for sandboxed ``exec_cell`` calls."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .._tool_filters import resolve_tool_permission
from .._tools.exec_cell import execute_cell
from ..approvals import approval_decision, approval_transport_available
from ..schemas import get_exec_cell_function_schemas
from ..security_scan import security_block_stage
from ..tool_specs import EXEC_CELL_API_TOOL_NAMES
from ..tool_validation import ToolSchemaSet
from .session_io import _summarize_args


def build_session_exec_cell_handler(
    session: Any,
    *,
    dispatch_getter: Callable[[], Callable[..., str]],
) -> Callable[[dict, str, Any], object]:
    """Build the code-mode handler with the active Session's policy state.

    Inner calls are ordinary harness dispatches, but the Session owns the
    schema, permission, approval, stale-read, ignore, and environment context
    that must surround them. ``dispatch_getter`` is intentionally late-bound
    so existing runtime instrumentation and tests can replace the dispatcher.
    """

    def _exec_cell_handler(args, dispatch_cwd, dispatch_cfg):
        cell_schemas = ToolSchemaSet.from_openai_tools(
            get_exec_cell_function_schemas(dispatch_cfg.tool_desc)
        )

        def _inner_dispatch(name, arguments, inner_cfg):
            execution_metadata: dict = {}
            if getattr(inner_cfg, "tools_schema_validation", "off") == "reject":
                validation = cell_schemas.validate(name, arguments)
                if not validation.valid:
                    session._emit(
                        "schema_reject",
                        session_number=session._session_number,
                        turn_number=session._current_turn,
                        parent_tool_name="exec_cell",
                        **validation.trace_fields(),
                    )
                    execution_metadata["executed"] = False
                    execution_metadata["gate_blocked"] = True
                    return validation.error_envelope(), execution_metadata

            approval_available = approval_transport_available(session._trace_path)
            resolution = resolve_tool_permission(
                policy=session._permission_policy,
                tool_name=name,
                arguments=arguments,
                cfg=inner_cfg,
                approval_available=approval_available,
            )
            session._emit(
                "permission",
                session_number=session._session_number,
                turn_number=session._current_turn,
                parent_tool_name="exec_cell",
                **resolution.trace_fields(),
            )
            if resolution.denied:
                execution_metadata["executed"] = False
                execution_metadata["gate_blocked"] = True
                return resolution.denial_envelope(), execution_metadata

            args_summary = _summarize_args(
                arguments, inner_cfg.args_summary_chars
            )
            approval_allowed, approval_reason = approval_decision(
                runtime_mode=getattr(inner_cfg, "runtime_mode", "measurement"),
                cwd=dispatch_cwd,
                trace_path=session._trace_path,
                tool_name=name,
                tool_args=arguments,
                args_summary=args_summary,
                required_reason=(
                    resolution.approval_reason()
                    if resolution.approval_required
                    else None
                ),
                permission_rule=(
                    resolution.rule if resolution.approval_required else None
                ),
            )
            if not approval_allowed:
                execution_metadata["executed"] = False
                execution_metadata["gate_blocked"] = True
                session._emit(
                    "approval_request",
                    session_number=session._session_number,
                    turn_number=session._current_turn,
                    tool_name=name,
                    parent_tool_name="exec_cell",
                    reason=approval_reason,
                )
                return (
                    "APPROVAL REQUIRED: inner exec_cell call was not "
                    f"executed. Reason: {approval_reason}.",
                    execution_metadata,
                )

            result = dispatch_getter()(
                name,
                arguments,
                cwd=dispatch_cwd,
                cfg=inner_cfg,
                output_control=(
                    session.output_control
                    if inner_cfg.bash_transforms_task_format_enabled
                    else None
                ),
                universal_rewrites=(
                    session.universal_rewrites
                    if inner_cfg.bash_transforms_universal_enabled
                    else None
                ),
                forbidden_rules=(
                    session.forbidden_rules
                    if inner_cfg.bash_quirks_forbidden_enabled
                    else None
                ),
                redirect_rules=session.redirect_rules,
                redactions=session.redactions,
                tool_registry=session._tool_registry,
                stale_guard=session._stale_guard,
                active_tools=EXEC_CELL_API_TOOL_NAMES,
                redirect_event_sink=session._redirect_event_sink,
                security_event_sink=session._security_event_sink,
                ignore_policy=session._ignore_policy,
                effective_env=session._effective_env,
                allow_login_shell=session._allow_login_shell,
                execution_metadata=execution_metadata,
            )
            execution_metadata["gate_blocked"] = (
                security_block_stage(result) == "args"
            )
            return result, execution_metadata

        from ..tools import _bash_readable_paths, _bash_unreadable_paths

        return execute_cell(
            args["source"],
            cwd=dispatch_cwd,
            cfg=dispatch_cfg,
            inner_dispatch=_inner_dispatch,
            unreadable_paths=_bash_unreadable_paths(
                dispatch_cwd, dispatch_cfg, session._ignore_policy
            ),
            readable_paths=_bash_readable_paths(dispatch_cfg),
            effective_env=session._effective_env,
            allow_login_shell=session._allow_login_shell,
        )

    return _exec_cell_handler
