#!/usr/bin/env python3
import os
import sys
from pathlib import Path


def _status(name: str, required: bool) -> int:
    value = os.environ.get(name)
    state = "set" if value else "missing"
    print(f"{name}: {state}")
    if not value and required:
        print(f"  export {name}=...")
        return 1
    return 0


def main() -> int:
    print("RLVR real test environment check")
    print("Workspace:", Path.cwd())
    missing = 0
    missing += _status("OPENAI_API_KEY", True)
    _status("OUMI_LOG_LEVEL", False)
    _status("WANDB_API_KEY", False)
    if missing:
        print("Missing required environment variables.")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
