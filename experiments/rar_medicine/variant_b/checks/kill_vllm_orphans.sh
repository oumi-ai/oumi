#!/usr/bin/env bash
# Use after a standalone vLLM check left its engine core behind (GPU memory held, ppid 1).
# Matches only engine/worker process names, never a script path, and excludes its own
# caller: `pgrep -f` matches any command line containing the pattern text.
# Kill every vLLM engine/worker process of this user. Only safe when no training run is active.
pat='VLLM::|EngineCore|multiprocessing.resource_tracker'
# Exclude this script, its caller shell and the caller's parent: pgrep -f matches any
# command line containing the pattern text, including the one that invoked us.
self="$$ $PPID $(ps -o ppid= -p "$PPID" 2>/dev/null | tr -d ' ')"
pids="$(pgrep -f "$pat" | grep -vxF -f <(printf '%s\n' $self) | tr '\n' ' ')"
echo "vllm procs: ${pids:-none}"
[[ -n "$pids" ]] && kill -TERM $pids 2>/dev/null
sleep 6
for p in $pids; do kill -0 "$p" 2>/dev/null && kill -KILL "$p" 2>/dev/null; done
for i in $(seq 1 20); do used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4); (( used < 1000 )) && break; sleep 2; done
echo "GPU4 used: ${used} MiB"
