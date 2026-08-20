"""Internal _loop package — split of loop.py helpers."""
from .focus_dedup import (
    _canon_focus_path,
    _dedup_signature,
    _encode_focus_path,
    _encode_focus_target,
    _extract_bash_focus_target,
    _extract_test_target_from_command,
    _focus_signature,
    _looks_like_path_token,
    _normalize_bash_for_dedup,
    _path_within_cwd,
    _split_bash_segments,
    _truncate_focus_display,
)
from .profile_resolution import (
    _apply_profile_preamble,
    _apply_profile_schema_simplify,
    _apply_profile_tool_cap,
    _filter_disabled_tools,
    _resolve_profile,
    _resolve_token_estimator,
    _simplify_tool_schema,
    apply_profile_to_schemas,
)
from .pretest_resume import (
    _pretest_is_green,
    _truncate_pretest_output,
    build_resume_prompt,
    run_pretest,
)
from ._session_setup import build_context_manager
from .session_io import (
    _auto_commit,
    _load_bash_transforms,
    _normalize_repo_timestamps,
    _record_session_start_costs,
    _sanitize_runner_timing,
    _summarize_args,
    _truncate_for_trace,
)
