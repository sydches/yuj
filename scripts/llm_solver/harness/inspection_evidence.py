"""File excerpts observed by a tool, separate from claims about understanding."""
import hashlib


class InspectedText(str):
    """Keep the selected body transient until output admission has finished."""

    def __new__(cls, text, *, path, data, body, start, count, total):
        value = super().__new__(cls, text)
        value.inspection_evidence = {
            "path": str(path), "namespace": "local_filesystem",
            "sha256": hashlib.sha256(data).hexdigest(),
            "start_line": start, "line_count": count, "total_lines": total,
        }
        from .task_path import TaskPath
        if isinstance(path, TaskPath):
            value.inspection_evidence.update(
                namespace='task_execution', task_view=dict(path.files.binding),
            )
        value.inspection_body = body
        return value


def admitted_inspection(record, body, output):
    """Certify only an entire selected body that survives output admission.

    Clipping or redaction leaves delivered extent unknown. Empty selections
    carry zero lines; an empty substring is never evidence of content delivery.
    The selected text itself is not retained in private traces.
    """
    return {**record, "admitted_output_sha256": hashlib.sha256(output.encode()).hexdigest(),
            "admitted_output_chars": len(output), "delivery": (
        "selected_excerpt" if body and body in output else
        "empty_selection" if not record["line_count"] else "unknown"
    )}
