"""The task's usable startup environment, without an infrastructure inventory."""


def build_runtime_briefing(working_directory, report):
    """Select briefing fields from existing task observations; perform no I/O."""
    briefing = {"working_directory": working_directory}
    if report is None:
        return briefing
    observations = report.get("observations", report.get("facts", []))
    declarations = [fact for fact in observations
                    if fact.get("source") == "project_file"
                    and fact.get("status") in ("observed", "present")]
    languages = sorted({fact["language_hint"] for fact in declarations
                        if fact.get("language_hint")})
    if languages:
        briefing["language"] = ", ".join(languages)
    selection = report.get("runner_selection", {})
    if selection.get("status") != "selected":
        briefing["test_runner_status"] = selection.get("status", "not established")
        return briefing
    selected = selection["selected"]
    runtime = selected.get("runtime", {})
    runner = selected["runner"]
    if selected.get("language"):
        briefing.setdefault("language", selected["language"])
    if runtime.get("version"):
        briefing["runtime_version"] = runtime["version"]
    if runtime.get("executable"):
        briefing["runtime_executable"] = runtime["executable"]
    else:
        briefing["test_runner_executable"] = selected["executable"]
    prefix = runtime.get("prefix")
    if prefix:
        briefing["environment_path"] = prefix
        for fact in observations:
            if (fact.get("source") == "conda_environments"
                    and fact.get("status") == "observed"
                    and prefix in fact.get("environment_candidates", [])):
                briefing["environment_manager"] = "conda"
                break
    briefing["test_runner"] = runner
    version = runtime.get("runner_version") or selected.get("version_output")
    if version:
        briefing["test_runner_version"] = version
    briefing["test_command"] = selected["base_cmd"]
    return briefing


def render_runtime_briefing(briefing):
    """Render the saved record directly into the retained system message."""
    lines = ["\n\nTask environment (observed at startup):",
             "Working directory: " + briefing["working_directory"]]
    if briefing.get("language"):
        language = briefing["language"]
        lines.append("Language: " + language)
    if briefing.get("runtime_version"):
        lines.append("Runtime version: " + briefing["runtime_version"])
    if briefing.get("runtime_executable"):
        lines.append("Runtime executable: " + briefing["runtime_executable"])
    if briefing.get("environment_path"):
        label = "Environment"
        if briefing.get("environment_manager"):
            label += " (" + briefing["environment_manager"] + ")"
        lines.append(label + ": " + briefing["environment_path"])
    if briefing.get("test_runner"):
        runner = briefing["test_runner"]
        if briefing.get("test_runner_version"):
            runner += " " + briefing["test_runner_version"]
        lines.append("Test runner: " + runner)
        if briefing.get("test_runner_executable"):
            lines.append("Test runner executable: " + briefing["test_runner_executable"])
        lines.append("Run tests with: " + briefing["test_command"])
    elif briefing.get("test_runner_status"):
        lines.append("Test runner: " + briefing["test_runner_status"].replace("_", " "))
    return "\n".join(lines)
