"""Utilities for generating and handling stable message IDs."""

from __future__ import annotations

import uuid


def generate_message_id() -> str:
    """Generate a new stable message id.

    Uses UUID4 and prefixes with 'msg_' to match DB convention.
    """
    return f"msg_{uuid.uuid4().hex}"

