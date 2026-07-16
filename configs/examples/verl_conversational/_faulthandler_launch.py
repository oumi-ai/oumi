"""Launch `oumi train` with periodic faulthandler stack dumps (debug the async hang).

Dumps all thread tracebacks to stderr every 180s (re-arming) so a driver hang is
captured in the job log without needing py-spy/ptrace. Delete once the async
rollout path is confirmed working.
"""

import faulthandler
import runpy
import sys

faulthandler.dump_traceback_later(180, repeat=True)

sys.argv = ["oumi", "train", "-c", "configs/examples/verl_conversational/train.yaml"]
runpy.run_module("oumi", run_name="__main__")
