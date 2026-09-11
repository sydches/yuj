"""Exercise observer fixtures with the loop's actual mutation accounting."""
from types import SimpleNamespace

from scripts.llm_solver.harness._guardrails.checks_post import rumination_ladder
from scripts.llm_solver.harness._guardrails.verification import observe_post_mutation_verification
from scripts.llm_solver.harness._loop._dispatch_tool_call import _apply_dispatch_effects


def seed_edited_input(guards, cfg, root, path):
    observe_post_mutation_verification(guards, cfg, tc_name='write_file',
        result='written', gate_blocked=False, source_write_paths=(path,), cwd=root,
        execution_metadata={'executed': True, 'file_changes': {
            'status': 'available', 'changed_paths': [path]}})


def account_check_changes(guards, cfg, root, tool, arguments, result, facts):
    state = SimpleNamespace(cfg=cfg, plan_mode_active=False,
        session=SimpleNamespace(_guards=guards, cwd=root, context=None),
        tool_post={'rumination_ladder': rumination_ladder},
        observers={'observe_post_mutation_verification': observe_post_mutation_verification})
    _apply_dispatch_effects(SimpleNamespace(name=tool, arguments=arguments), state, {}, facts, result)
