"""Single external_lib entry point for gemma-4 verl runs (verl's import_external_libs takes one
module name). Order matters only for readability; each module is idempotent.

  gemma4_kv_share_patch   REQUIRED: cache-less forwards reuse the shared K/V of gemma-4's 20
                          KV-shared layers (without it every verl forward is garbage).
  verl_rank_buffer_sync   REQUIRED: re-broadcast module buffers (RoPE inv_freq tables, embed_scale,
                          softcap) from rank 0 after FSDP construction; FSDP's own
                          sync_module_states left ranks 1-3 with wrong RoPE tables (2026-09-02).
  verl_logprob_dump_hook  diagnostic, inert unless VERL_LOGPROB_DUMP_DIR is set.

Wired in train_verl.yaml as `actor_rollout_ref.model.external_lib: gemma4_verl_fixes`; the
directory is on PYTHONPATH via run.sh.
"""

import gemma4_kv_share_patch  # noqa: F401
import verl_logprob_dump_hook  # noqa: F401
import verl_rank_buffer_sync  # noqa: F401
