"""Shared ladder settings; each guard retains its own trigger and action."""

SUPPORTED = {
    "identical_call": {2, 3},
    "duplicate_call": {2, 3},
    "silent_call": {2, 4},
    "no_edit": {2, 4},
    "done_without_check": {3},
}


def validate_ladders(value):
    if not isinstance(value, dict):
        raise ValueError("loop.guard_ladders must be a table")
    for name, policy in value.items():
        if name not in SUPPORTED or not isinstance(policy, dict):
            raise ValueError(f"unknown guard ladder: {name}")
        if set(policy) - {"rungs", "release_after"}:
            raise ValueError(f"unknown ladder setting for {name}")
        rungs = policy.get("rungs", {})
        if not isinstance(rungs, dict):
            raise ValueError(f"{name}.rungs must be a table")
        for rung, count in rungs.items():
            if str(rung) not in {str(n) for n in SUPPORTED[name]}:
                raise ValueError(f"unsupported rung {rung} for {name}")
            if type(count) is not int or count < 1:
                raise ValueError(f"{name}: rung counts must be positive integers")
        if "release_after" in policy:
            count = policy["release_after"]
            if name not in {"silent_call", "no_edit"} or type(count) is not int or count < 1:
                raise ValueError("release_after requires a positive advisory-guard block count")
    return value


def threshold(cfg, name, rung, fallback):
    policy = getattr(cfg, "guard_ladders", {}).get(name, {})
    # An explicit rung table replaces the old tiers, including omitted tiers.
    if "rungs" in policy:
        return policy["rungs"].get(str(rung), 0)
    return fallback


def release_after(cfg, name):
    return getattr(cfg, "guard_ladders", {}).get(name, {}).get("release_after", 3)
