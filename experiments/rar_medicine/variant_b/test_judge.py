"""Smoke test for the Variant B SimpleJudge reward, using real RaR-Medicine samples.

Offline checks (always run): registry lookups, dataset transform shape,
judge_config.yaml loads and its placeholders match what the reward passes.

Live checks (need OPENAI_API_KEY): judges the reference answer itself (should
score high) and a deliberately wrong answer (should score low) for a couple of
samples, through the registered reward function exactly as verl would call it.

Usage:
    cd /workspace/persist/shanghong/oumi/experiments/rar_medicine/variant_b
    OUMI_EXTRA_DEPS_FILE=$PWD/oumi_extra_deps.txt python test_judge.py
"""

import os
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import rar_medicine_grpo as mod

_TRAIN_PARQUET = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "data",
    "train-00000-of-00001.parquet",
)

_WRONG_ANSWER = (
    "The final answer is that no intervention is needed; this finding is a "
    "normal variant and the patient can be reassured without further workup."
)


def test_offline() -> list[dict]:
    from oumi.core.configs.judge_config import JudgeConfig
    from oumi.core.configs.params.judge_params import JudgeOutputType
    from oumi.core.registry import REGISTRY, RegistryType

    reward_fn = REGISTRY.get("rar_medicine_verl", RegistryType.REWARD_FUNCTION)
    assert reward_fn is mod.rar_medicine_verl, "Reward function not registered"
    dataset_cls = REGISTRY.get_dataset("anisha2102/RaR-Medicine")
    assert dataset_cls is mod.RarMedicineGrpoDataset, "Dataset not registered"
    print("[ok] registry: reward function and dataset class found")

    judge_config = JudgeConfig.from_path(mod._JUDGE_CONFIG_PATH)
    placeholders = judge_config.judge_params.get_placeholders()
    assert placeholders == {"question", "reference_answer", "response"}, placeholders
    assert judge_config.judge_params.judgment_type == JudgeOutputType.INT
    assert judge_config.inference_config is not None
    assert judge_config.inference_config.model.model_name == "gpt-4.1-mini"
    print("[ok] judge_config.yaml: loads, placeholders and judgment type match")

    df = pd.read_parquet(_TRAIN_PARQUET).head(2)
    # Bypass __init__ (which downloads the hub dataset); transform only needs
    # the split attribute.
    dataset = mod.RarMedicineGrpoDataset.__new__(mod.RarMedicineGrpoDataset)
    dataset.split = "train"
    samples = []
    for _, row in df.iterrows():
        entry = dataset.transform(row)
        question = str(row["question"]).strip()
        reference = str(row["reference_answer"]).strip()
        assert entry["prompt"][0]["role"] == "system"
        assert entry["prompt"][1]["content"] == question
        assert entry["reward_model"]["ground_truth"] == reference
        assert entry["extra_info"]["question"] == question
        samples.append(entry)
    print("[ok] dataset transform: verl-format rows look right")
    return samples


def test_live(samples: list[dict]) -> None:
    for i, entry in enumerate(samples):
        question = entry["extra_info"]["question"]
        reference = entry["reward_model"]["ground_truth"]
        # Threads mimic how verl's reward workers call the sync reward fn.
        with ThreadPoolExecutor(max_workers=3) as pool:
            good, bad, empty = pool.map(
                lambda resp: mod.rar_medicine_verl(
                    "test", resp, reference, entry["extra_info"]
                ),
                [reference, _WRONG_ANSWER, ""],
            )
        print(f"\n[sample {i}] {question[:100]}...")
        print(f"  reference-as-response reward: {good:.2f}  (expect high, ~0.8+)")
        print(f"  wrong-answer reward:          {bad:.2f}  (expect low, <=0.3)")
        print(f"  empty-response reward:        {empty:.2f}  (expect 0.00)")
        assert empty == 0.0
        assert good > bad, "Judge did not rank reference above a wrong answer!"
    print("\n[ok] live judge sanity checks passed")


if __name__ == "__main__":
    import traceback

    try:
        samples = test_offline()
        if os.environ.get("OPENAI_API_KEY"):
            test_live(samples)
        else:
            print("[skip] OPENAI_API_KEY not set; skipping live judge calls")
    except Exception:
        traceback.print_exc()
        sys.stdout.flush()
        os._exit(1)
    # Importing oumi pulls in wandb, whose atexit shutdown has been seen to hang
    # this process indefinitely after every check has printed. All results are
    # out by now, so skip the interpreter teardown.
    sys.stdout.flush()
    os._exit(0)
