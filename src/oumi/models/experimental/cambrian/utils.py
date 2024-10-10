import os
import sys

from oumi.models.experimental.cambrian.constants import LOGDIR

server_error_msg = (
    "**NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.**"
)
moderation_msg = (
    "YOUR INPUT VIOLATES OUR CONTENT MODERATION GUIDELINES. PLEASE TRY AGAIN."
)

# handler = None

# TODO: move elsewhere?
IS_XLA_AVAILABLE = False
try:
    import torch_xla

    IS_XLA_AVAILABLE = True
except ImportError:
    pass
