"""Does vLLM's gemma-4 behave identically after verl's step-1 sync path (dummy init + load_weights)?

verl starts vLLM with load_format=dummy and then pushes the actor's state dict through
model.load_weights() (bucketed IPC, fp32 tensors, HF names). Anything load_weights silently
skips stays random forever. This check:
  1. starts a normally loaded engine (weights from safetensors), samples N responses with
     logprobs (mode = LOGPROBS_MODE, default the verl setting processed_logprobs);
  2. starts a dummy-initialised engine, runs model.load_weights() over the real checkpoint on
     the worker (same call as verl's receiver) and lists vLLM parameters that were NOT loaded;
  3. scores the step-1 responses with the dummy+loaded engine (prompt_logprobs) and with HF (+
     the KV-share patch), and reports mean |dp| / Pearson against the normal engine.

Usage (one free GPU, ~4 min):
  CUDA_VISIBLE_DEVICES=6 python vllm_dummy_sync_fidelity_check.py
Exit 0 if no vLLM parameter is left unloaded and dummy+loaded matches the normal engine
within mean |dp| < 0.02.
"""

import gc
import os
import sys
from pathlib import Path

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

BASE = "google/gemma-4-E2B-it"
N = int(os.environ.get("N_PROMPTS", "16"))
MODE = os.environ.get("LOGPROBS_MODE", "processed_logprobs")
_REPO_ROOT = Path(__file__).resolve().parents[4]
PARQUET = str(
    _REPO_ROOT
    / "tmp/rar_medicine/variant_b/output/medqa_gemma4-e2b-it_fullft/verl_datasets/val.parquet"
)


def _load_ckpt_into_worker(worker):
    """Runs inside the vLLM worker: replay verl's receiver-side load_weights over the checkpoint."""
    import glob

    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    model = worker.model_runner.model
    ckpt = snapshot_download(
        "google/gemma-4-E2B-it", allow_patterns=["*.safetensors", "*.json"]
    )
    ckpt_names = []

    def gen():
        for f in sorted(glob.glob(ckpt + "/*.safetensors")):
            with safe_open(f, "pt", device="cpu") as sf:
                for k in sf.keys():
                    ckpt_names.append(k)
                    yield k, sf.get_tensor(k).float()  # verl streams fp32

    loaded = model.load_weights(gen())
    loaded = set(loaded) if loaded is not None else set()
    all_params = {n for n, _ in model.named_parameters()}
    return {
        "ckpt_tensors": len(ckpt_names),
        "vllm_params": len(all_params),
        "loaded": len(loaded),
        "not_loaded": sorted(all_params - loaded)[:40],
        "n_not_loaded": len(all_params - loaded),
    }


def main():
    import pandas as pd
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(BASE)
    df = pd.read_parquet(PARQUET)
    convs = [[dict(m) for m in df.iloc[i]["prompt"]] for i in range(N)]
    prompt_ids = [
        tok(
            tok.apply_chat_template(c, add_generation_prompt=True, tokenize=False),
            add_special_tokens=False,
        )["input_ids"]
        for c in convs
    ]
    common = dict(
        model=BASE,
        dtype="bfloat16",
        max_model_len=1600,
        gpu_memory_utilization=0.4,
        limit_mm_per_prompt={"image": 0, "video": 0, "audio": 0},
        seed=0,
        logprobs_mode=MODE,
    )

    # 1. normal engine: sample
    llm = LLM(**common)
    sp = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=400, logprobs=0, seed=0)
    outs = llm.generate([{"prompt_token_ids": p} for p in prompt_ids], sp)
    resp_ids = [list(o.outputs[0].token_ids) for o in outs]
    normal_lp = [
        [
            lpd[t].logprob
            for lpd, t in zip(o.outputs[0].logprobs, o.outputs[0].token_ids)
        ]
        for o in outs
    ]
    del llm
    gc.collect()
    torch.cuda.empty_cache()

    # 2. dummy engine + load_weights (verl's step-1 path), then score the same tokens
    llm2 = LLM(load_format="dummy", **common)
    report = llm2.collective_rpc(_load_ckpt_into_worker)[0]
    print(
        f"[load_weights] checkpoint tensors {report['ckpt_tensors']}, vLLM params {report['vllm_params']}, "
        f"loaded {report['loaded']}, NOT loaded {report['n_not_loaded']}: {report['not_loaded']}"
    )
    sp2 = SamplingParams(temperature=1.0, max_tokens=1, prompt_logprobs=0)
    outs2 = llm2.generate(
        [{"prompt_token_ids": p + r} for p, r in zip(prompt_ids, resp_ids)], sp2
    )
    dummy_lp = []
    for o, p, r in zip(outs2, prompt_ids, resp_ids):
        plp = (
            o.prompt_logprobs
        )  # list (len prompt+resp), entry i = logprobs of token i given <i
        dummy_lp.append([plp[len(p) + j][t].logprob for j, t in enumerate(r)])
    del llm2
    gc.collect()
    torch.cuda.empty_cache()

    # 3. HF + KV-share patch
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import gemma4_kv_share_patch  # noqa: F401
    from transformers import AutoModelForCausalLM
    from verl.utils import torch_functional as verl_F

    model = (
        AutoModelForCausalLM.from_pretrained(
            BASE, dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        .cuda()
        .eval()
    )
    hf_lp = []
    for p, r in zip(prompt_ids, resp_ids):
        ids = torch.tensor([p + r], device="cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids=ids, use_cache=False).logits[:, len(p) - 1 : -1, :]
        hf_lp.append(
            verl_F.logprobs_from_logits(logits, torch.tensor([r], device="cuda"))[0]
            .float()
            .cpu()
            .tolist()
        )

    def cmp(name, a, b):
        a = torch.tensor(sum(a, []))
        b = torch.tensor(sum(b, []))
        dp = (a.exp() - b.exp()).abs()
        corr = torch.corrcoef(torch.stack([a.exp(), b.exp()]))[0, 1].item()
        print(
            f"  {name:44s} mean|dp| {dp.mean():.4f} p99 {dp.quantile(0.99):.3f} max {dp.max():.3f} pearson {corr:.3f} (n={len(a)})"
        )
        return dp.mean().item()

    print(f"logprobs_mode={MODE}, {sum(len(r) for r in resp_ids)} sampled tokens")
    d1 = cmp("dummy+load_weights vLLM vs normal vLLM", dummy_lp, normal_lp)
    cmp("HF(+patch) vs normal vLLM", hf_lp, normal_lp)
    cmp("HF(+patch) vs dummy+load_weights vLLM", hf_lp, dummy_lp)
    ok = report["n_not_loaded"] == 0 and d1 < 0.02
    print("RESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
