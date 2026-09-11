"""Execution facts and check outcomes never come from rendered output text."""


def recorded_outcome(metadata, *, gate_blocked=False):
    metadata = metadata or {}
    fields = {"outcome_version": "native_execution_v1", "outcome": "unknown",
              "pass_fail": "unknown", "exit_status": None, "error_class": ""}
    if gate_blocked:
        return {**fields, "outcome": "blocked", "error_class": "harness_gate"}
    if metadata.get("security_blocked_stage"):
        return {**fields, "outcome": "error", "error_class": "security_block"}
    if metadata.get("executed") is False:
        return {**fields, "outcome": "not_executed"}
    if metadata.get("timed_out"):
        return {**fields, "outcome": "error", "error_class": "timeout"}
    status = metadata.get("exit_status")
    if metadata.get("exit_status_known") and type(status) is int:
        fields.update(outcome="completed", exit_status=status)
        check = metadata.get("verification_status")
        if check in {"passed", "custom_passed"} and status == 0:
            fields.update(outcome="ok", pass_fail="pass")
        elif check in {"failed", "custom_failed"} and status != 0:
            fields.update(outcome="error", pass_fail="fail", error_class="check_failed")
    elif metadata.get("executed") is True:
        fields["outcome"] = "completed"
    return fields
