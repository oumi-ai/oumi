import inspect
import os


from typing import Optional


def model_name_fallback(reason: Optional[str] = None) -> str:
    """Return a fallback model name that includes the caller code path.

    Example: "Not found (oumi/src/oumi/webchat/server.py:93) - config.model.model_name"

    Args:
        reason: Optional hint about what was missing.

    Returns:
        A string suitable for exposing in model metadata.
    """
    location = "<unknown>"
    try:
        frame = inspect.currentframe()
        caller = frame.f_back if frame else None
        if caller:
            filename = caller.f_code.co_filename
            # Prefer a path relative to the working directory if possible
            try:
                rel = os.path.relpath(filename)
            except Exception:
                rel = filename
            location = f"{rel}:{caller.f_lineno}"
    except Exception:
        # Best-effort only; keep <unknown> location
        pass

    base = f"Not found ({location})"
    if reason:
        base = f"{base} - {reason}"
    return base
