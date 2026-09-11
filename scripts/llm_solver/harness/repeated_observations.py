"""Private receipts for completed observations, independent of rendered output."""
import hashlib
import json


def read_observation(text):
    return {"kind": "read_observation", "pending": False,
            "sha256": hashlib.sha256(str(text).encode("utf-8", errors="replace")).hexdigest()}


def process_observation(proc_id, running, start, end, exit_code):
    return {"kind": "process_observation", "pending": bool(running),
            "proc_id": proc_id, "cursor_start": start, "cursor_end": end,
            "exit_code": exit_code}


def observation_key(receipt):
    if not isinstance(receipt, dict) or receipt.get("pending") is not False:
        return None
    if receipt.get("kind") == "read_observation":
        digest = receipt.get("sha256")
        if (not isinstance(digest, str) or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)):
            return None
    elif receipt.get("kind") == "process_observation":
        if (not isinstance(receipt.get("proc_id"), str)
                or type(receipt.get("exit_code")) is not int
                or type(receipt.get("cursor_start")) is not int
                or type(receipt.get("cursor_end")) is not int
                or receipt["cursor_start"] < 0
                or receipt["cursor_end"] < receipt["cursor_start"]):
            return None
    else:
        return None
    return json.dumps(receipt, sort_keys=True, separators=(",", ":"))
