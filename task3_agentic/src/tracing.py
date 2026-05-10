import json
import time
from datetime import datetime
from pathlib import Path


TRACE_FILE = Path("../logs/agent_trace.jsonl")


def log_tool_call(
    tool_name: str,
    inputs: dict,
    output,
    duration: float
):
    TRACE_FILE.parent.mkdir(parents=True, exist_ok=True)

    trace = {
        "timestamp": datetime.utcnow().isoformat(),
        "tool": tool_name,
        "inputs": inputs,
        "output_preview": str(output)[:200],
        "duration_seconds": round(duration, 4)
    }

    with open(TRACE_FILE, "a") as f:
        f.write(json.dumps(trace) + "\n")


def traced_tool(func):
    def wrapper(*args, **kwargs):
        start = time.time()

        try:
            result = func(*args, **kwargs)
            duration = time.time() - start

            log_tool_call(
                tool_name=func.__name__,
                inputs={
                    "args": str(args),
                    "kwargs": kwargs
                },
                output=result,
                duration=duration
            )

            return result

        except Exception as e:
            duration = time.time() - start

            log_tool_call(
                tool_name=func.__name__,
                inputs={
                    "args": str(args),
                    "kwargs": kwargs
                },
                output=f"ERROR: {str(e)}",
                duration=duration
            )

            raise e

    return wrapper