"""Evaluate base gemma-4-E2B-it vs the GRPO-trained variant_b on
``medqa_test_1k.jsonl`` (1000 RaR-Medicine *test* prompts, oumi conversation
format, training system prompt + reference answer in metadata) using the
**training reward** judge (``variant_b_training_reward`` = holistic 0-10
meta-rubric, gpt-4.1-mini, T=0).

Stages (all cached in results/test1k/; rerun with --overwrite):

    python eval_test1k.py generate --model base       --gpu 0
    python eval_test1k.py generate --model variant_b  --gpu 1
    python eval_test1k.py judge                       # both models, OpenAI
    python eval_test1k.py summarize                   # -> results/test1k/SUMMARY.md

``generate`` defaults to greedy (T=0, 1 sample). Use ``--temperature 1.0
--num-generations 4`` to mimic GRPO rollout sampling instead. Add
``--judges rubric:model`` to ``judge``/``summarize`` for held-out graders (the
test file has no per-question rubrics, so only reference-anchored variants
such as ``fixed_rubric_implicit`` or ``reference_only`` apply).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "rar_medicine_judges"))

DATA = HERE / "medqa_test_1k.jsonl"
OUT = HERE / "results" / "test1k"
MODELS = {
    "base": "google/gemma-4-E2B-it",
    "variant_b": str(HERE / "models" / "variant_b_merged"),  # LoRA run, merged
    "fullft": str(
        HERE / "models" / "medqa_gemma4-e2b-it_fullft"
    ),  # full fine-tune run, final save (== step 250 weights)
    "fullft_step200": str(
        HERE / "models" / "fullft_step200"
    ),  # intermediate, via verl.model_merger
}
TRAINING_JUDGE = "variant_b_training_reward:gpt-4.1-mini"
_FINAL = re.compile(r"the final answer is", re.I)
_DUP_TAIL = re.compile(r"the final answer is[^\n]*\n+\s*the final answer is", re.I)
# One fixed extraction rule (do not tune per model): text after "The final answer
# is", up to the end of the line or the first sentence-ending period.
_FINAL_TXT = re.compile(r"the final answer is\s*:?\s*(.+?)(?:\.(?:\s|$)|\n|$)", re.I)
_OPTION = re.compile(r"^\(?([a-e])\)?(?:[.):\s-]|$)", re.I)
# "The final answer is based on ..." / "... determined by ..." are preambles to the
# real answer, not answers; drop them (fixed list, applied to every model).
_PREAMBLE = re.compile(r"^(based on|determined by|derived from)\b", re.I)

COMMITMENT_KINDS = ("single", "repeated_identical", "conflicting", "none")


def extract_finals(text: str) -> list[str]:
    """Normalised final answers in order of appearance."""
    out = []
    for m in _FINAL_TXT.findall(text or ""):
        a = re.sub(r"[*_`\"']", "", m).strip(" .:;,-").lower()
        a = re.sub(r"\s+", " ", a)
        if a and not _PREAMBLE.match(a):
            out.append(a)
    return out


def _same_answer(x: str, y: str) -> bool:
    """Identical after normalisation, or one is a prefix of the other (e.g. "c" vs
    "c. decreased serum estradiol"), or both start with the same option letter.
    """
    if x == y or x.startswith(y) or y.startswith(x):
        return True
    ox, oy = _OPTION.match(x), _OPTION.match(y)
    return bool(ox and oy and ox.group(1).lower() == oy.group(1).lower())


def classify_commitment(text: str) -> str:
    finals = extract_finals(text)
    if not finals:
        return "none"
    if len(finals) == 1:
        return "single"
    first = finals[0]
    return (
        "repeated_identical"
        if all(_same_answer(first, f) for f in finals[1:])
        else "conflicting"
    )


def load_env() -> None:
    env = HERE.parent.parent / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("\"'"))


def load_prompts(limit: int | None) -> list[dict]:
    rows = [json.loads(line) for line in open(DATA) if line.strip()]
    out = []
    for r in rows[:limit]:
        sys_msg = next(m["content"] for m in r["messages"] if m["role"] == "system")
        user = next(m["content"] for m in r["messages"] if m["role"] == "user")
        out.append(
            {
                "idx": r["metadata"]["idx"],
                "conversation_id": r["conversation_id"],
                "question_source": r["metadata"]["question_source"],
                "question": user,
                "system": sys_msg,
                "reference_answer": r["metadata"]["reference_answer"],
            }
        )
    return out


def engine_for(judge_model: str) -> str:
    """Route a judge model name to an oumi inference engine."""
    if judge_model.startswith(("gpt-", "o1", "o3", "o4")):
        return "OPENAI"
    if judge_model.startswith("claude-"):
        return "ANTHROPIC"  # needs ANTHROPIC_API_KEY (repo .env)
    if judge_model.startswith("gemini-"):
        return "GEMINI"  # needs GEMINI_API_KEY (repo .env)
    return "VLLM"  # local HF id or path


def responses_path(model: str) -> Path:
    return OUT / f"responses__{model}.jsonl"


def judge_path(model: str, judge: str) -> Path:
    variant, _, jm = judge.partition(":")
    return OUT / f"judge__{model}__{variant}__{jm.replace('/', '_')}.jsonl"


# --------------------------------------------------------------------------- #
def cmd_generate(args: argparse.Namespace) -> None:
    from common import write_jsonl

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpu)
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    out = responses_path(args.model)
    if out.exists() and not args.overwrite:
        raise SystemExit(f"{out} exists; --overwrite to regenerate")
    prompts = load_prompts(args.limit)

    from oumi.core.configs import GenerationParams, ModelParams
    from oumi.core.types.conversation import Conversation, Message, Role
    from oumi.inference import VLLMInferenceEngine

    convs = [
        Conversation(
            messages=[
                Message(role=Role.SYSTEM, content=p["system"]),
                Message(role=Role.USER, content=p["question"]),
            ],
            metadata={"i": i, "gen_idx": g},
        )
        for i, p in enumerate(prompts)
        for g in range(args.num_generations)
    ]
    engine = VLLMInferenceEngine(
        ModelParams(model_name=MODELS[args.model], torch_dtype_str="bfloat16"),
        generation_params=GenerationParams(
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=1.0,
            seed=args.seed,
        ),
        gpu_memory_utilization=0.85,
    )
    done = engine.infer(convs)
    by_key = {(c.metadata["i"], c.metadata["gen_idx"]): c for c in done}
    records = []
    for i, p in enumerate(prompts):
        for g in range(args.num_generations):
            c = by_key[(i, g)]
            msg = c.last_message(Role.ASSISTANT)
            usage = (c.metadata or {}).get("usage") or {}
            records.append(
                {
                    **{
                        k: p[k]
                        for k in (
                            "idx",
                            "conversation_id",
                            "question_source",
                            "question",
                            "reference_answer",
                        )
                    },
                    "gen_idx": g,
                    "model": args.model,
                    "model_path": MODELS[args.model],
                    "temperature": args.temperature,
                    "response": msg.content if msg else "",
                    "finish_reason": (c.metadata or {}).get("finish_reason"),
                    "completion_tokens": usage.get("completion_tokens"),
                }
            )
    write_jsonl(out, records)
    n_trunc = sum(r["finish_reason"] == "length" for r in records)
    print(f"wrote {len(records)} -> {out} (truncated: {n_trunc})")


def robust_run_variant(
    engine,
    engine_name,
    jm,
    variant,
    recs,
    explanation,
    args,
    chunk: int = 100,
    partial_path: Path | None = None,
):
    """run_variant, but (a) a single failed request (e.g. an Anthropic
    ``stop_reason=refusal`` with empty content, which the engine raises on) does
    not abort the batch: the chunk is bisected until the failing record is
    isolated and recorded as score=None; and (b) finished chunks are appended to
    ``partial_path`` so a killed/stalled run resumes where it stopped.
    """
    from common import read_jsonl
    from run_judges import run_variant

    done: list[dict] = []
    if partial_path and partial_path.exists():
        done = read_jsonl(partial_path)
        print(f"  resuming: {len(done)} records already judged in {partial_path.name}")

    def _run(sub):
        try:
            return run_variant(engine, engine_name, jm, variant, sub, explanation, args)
        except Exception as e:  # noqa: BLE001
            if len(sub) == 1:
                print(f"  ! record idx={sub[0].get('idx')} failed: {str(e)[:160]}")
                return [
                    {
                        "score": None,
                        "raw_value": None,
                        "explanation": None,
                        "raw_output": f"ERROR: {str(e)[:500]}",
                        "prompt_tokens": None,
                        "completion_tokens": None,
                        "run_wall_s": None,
                    }
                ]
            mid = len(sub) // 2
            return _run(sub[:mid]) + _run(sub[mid:])

    outs = list(done)
    for i in range(len(done), len(recs), chunk):
        part = _run(recs[i : i + chunk])
        outs += part
        if partial_path:
            with open(partial_path, "a") as f:
                for o in part:
                    f.write(json.dumps(o, ensure_ascii=False) + "\n")
            print(f"  checkpoint {len(outs)}/{len(recs)}", flush=True)
    return outs


def cmd_judge(args: argparse.Namespace) -> None:
    load_env()
    from common import read_jsonl, write_jsonl
    from run_judges import build_engine, load_variant

    for judge in args.judges:
        variant_name, _, jm = judge.partition(":")
        variant = load_variant(variant_name)
        engine_name = engine_for(jm)
        if engine_name == "VLLM":
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpu)
            os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        engine = None
        for model in args.models:
            out = judge_path(model, judge)
            if out.exists() and not args.overwrite:
                print(f"skip (cached) {out}")
                continue
            recs = read_jsonl(responses_path(model))
            if args.limit:
                recs = recs[: args.limit]
            for r in recs:
                r.setdefault("rubric", [])  # fixed-rubric variants ignore it
            if engine is None:
                engine = build_engine(engine_name, jm, args)
                if engine_name == "ANTHROPIC":
                    # A safety-classifier refusal (empty content, stop_reason=refusal) is
                    # deterministic, but the engine retries it with exponential backoff
                    # (7 attempts ~= 5 min). Cap retries and use small chunks so the
                    # bisection in robust_run_variant isolates refused records quickly.
                    engine._remote_params.max_retries = 1
            chunk = 20 if engine_name == "ANTHROPIC" else 100
            print(f"judging {len(recs)} {model} responses with {judge}")
            partial = out.with_suffix(".partial.jsonl") if not args.limit else None
            outs = robust_run_variant(
                engine,
                engine_name,
                jm,
                variant,
                recs,
                variant["include_explanation"],
                args,
                chunk=chunk,
                partial_path=partial,
            )
            write_jsonl(out, outs)
            if partial and partial.exists():
                partial.unlink()
            print(f"  -> {out} ({sum(o['score'] is None for o in outs)} unparseable)")


def cmd_summarize(args: argparse.Namespace) -> None:
    import numpy as np
    import pandas as pd
    from common import read_jsonl

    frames = []
    for model in args.models:
        df = pd.DataFrame(read_jsonl(responses_path(model)))
        for judge in args.judges:
            outs = read_jsonl(judge_path(model, judge))
            assert len(outs) == len(df), (model, judge)
            df[f"score::{judge}"] = [o["score"] for o in outs]
            df[f"raw::{judge}"] = [o["raw_value"] for o in outs]
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)
    all_df["chars"] = all_df.response.fillna("").str.len()
    all_df["n_final"] = all_df.response.fillna("").map(lambda s: len(_FINAL.findall(s)))
    all_df["truncated"] = all_df.finish_reason.eq("length")
    all_df["commitment"] = all_df.response.fillna("").map(classify_commitment)
    all_df["finals"] = all_df.response.fillna("").map(extract_finals)
    all_df["dup_tail"] = all_df.response.fillna("").map(
        lambda s: bool(_DUP_TAIL.search(s))
    )
    all_df["bold"] = all_df.response.fillna("").str.count(r"\*\*")
    if len(args.models) != 2:
        raise SystemExit("summarize compares exactly two models: --models base fullft")
    a, b = args.models[0], args.models[1]
    A, B = all_df[all_df.model == a], all_df[all_df.model == b]

    L = [
        f"# {b} vs {a} on medqa_test_1k (RaR-Medicine test, n={A.idx.nunique()})",
        "",
        f"Training judge: `{args.judges[0]}` (reward = score/10). "
        f"Sampling: T={A.temperature.iloc[0]}, {A.groupby('idx').size().iloc[0]} sample(s) per prompt.",
        "",
    ]

    L += [
        "## Judge scores",
        "",
        "| judge | role | "
        + f"{a} mean | {b} mean | delta | 95% CI (paired) | win rate | {a} wrong (<=3) | {b} wrong (<=3) | {a} 9-10 | {b} 9-10 |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    rng = np.random.default_rng(0)
    deltas: dict[str, float] = {}
    for k, j in enumerate(args.judges):
        col = f"score::{j}"
        pa = A.groupby("idx")[col].mean()
        pb = B.groupby("idx")[col].mean().reindex(pa.index)
        d = (pb - pa).dropna().to_numpy()
        boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        win = float((pb > pa).mean() + 0.5 * (pb == pa).mean())
        wrong_a, wrong_b = (A[col] <= 0.3).mean(), (B[col] <= 0.3).mean()
        top_a, top_b = (A[col] >= 0.9).mean(), (B[col] >= 0.9).mean()
        deltas[j] = float(pb.mean() - pa.mean())
        n_fail = int(A[col].isna().sum() + B[col].isna().sum())
        L.append(
            f"| {j} | {'training' if k == 0 else 'held-out'} | {pa.mean():.3f} | {pb.mean():.3f} | {deltas[j]:+.3f} | "
            f"[{lo:+.3f}, {hi:+.3f}] | {win:.2f} | {wrong_a:.1%} | {wrong_b:.1%} | {top_a:.1%} | {top_b:.1%} |"
            + (f" ({n_fail} unparseable)" if n_fail else "")
        )
    L.append("")
    if len(args.judges) > 1:
        t = deltas[args.judges[0]]
        L += [
            "Transfer ratio = held-out delta / training-judge delta (1.0 = the gain is fully "
            "judge-independent; well below 1 = gain specific to the training judge; reference "
            "for a simply-better model, E4B vs E2B on val: 0.91).",
            "",
        ]
        for j in args.judges[1:]:
            L.append(
                f"- {j}: **{deltas[j] / t if t else float('nan'):.2f}** ({deltas[j]:+.3f} / {t:+.3f})"
            )
        L.append("")

        # inter-judge agreement on the same responses (both models pooled)
        j0c = f"score::{args.judges[0]}"
        # Agreement metrics following "Judge's Verdict" (arXiv 2510.09738): correlation
        # alone hides systematic leniency/harshness and ignores chance agreement, so
        # report chance-corrected kappa and the mean bias alongside it.
        from sklearn.metrics import cohen_kappa_score

        def _raw(j):
            return all_df[f"raw::{j}"].astype("float")

        L += [
            "Agreement with the training judge on the same responses "
            "(kappa_w = quadratic-weighted Cohen's kappa on the 0-10 score; kappa_bin = plain kappa on the "
            "wrong/right cut at <=3; bias = mean(judge - training) on the [0,1] scale, negative = harsher):",
            "",
            "| judge | Spearman | raw agree (<=3 cut) | kappa_bin | kappa_w | bias | judge's own wrong-rate |",
            "|---|---|---|---|---|---|---|",
        ]
        for j in args.judges[1:]:
            c = f"score::{j}"
            ok = all_df[[j0c, c, f"raw::{args.judges[0]}", f"raw::{j}"]].dropna()
            rho = ok[j0c].corr(ok[c], method="spearman")
            agree = ((ok[j0c] <= 0.3) == (ok[c] <= 0.3)).mean()
            k_bin = cohen_kappa_score(ok[j0c] <= 0.3, ok[c] <= 0.3)
            k_w = cohen_kappa_score(
                ok[f"raw::{args.judges[0]}"].astype(int),
                ok[f"raw::{j}"].astype(int),
                weights="quadratic",
            )
            bias = (ok[c] - ok[j0c]).mean()
            L.append(
                f"| {j} | {rho:.2f} | {agree:.1%} | {k_bin:.2f} | {k_w:.2f} | {bias:+.3f} | {(ok[c] <= 0.3).mean():.1%} |"
            )
        L.append("")
        L.append(
            "Landis & Koch: kappa 0.61-0.80 substantial, 0.81-1.00 almost perfect. The paper's human-human "
            "baseline on a comparable reference-anchored task was kappa 0.80."
        )
        L.append("")

        # Group-level reliability across all judges + consensus z-score (LLM-consensus
        # analog of the paper's human-likeness test; we have no human annotators).
        try:
            import krippendorff

            mat = np.vstack(
                [_raw(j).to_numpy() for j in args.judges]
            )  # judges x items, NaN allowed
            alpha = krippendorff.alpha(
                reliability_data=mat, level_of_measurement="ordinal"
            )
        except Exception as e:  # noqa: BLE001
            alpha = float("nan")
            print("krippendorff failed:", e)
        pair = {}
        for i, ji in enumerate(args.judges):
            for jj in args.judges[i + 1 :]:
                ok = all_df[[f"raw::{ji}", f"raw::{jj}"]].dropna()
                pair[(ji, jj)] = pair[(jj, ji)] = cohen_kappa_score(
                    ok.iloc[:, 0].astype(int),
                    ok.iloc[:, 1].astype(int),
                    weights="quadratic",
                )
        mean_k = {
            j: np.mean([pair[(j, o)] for o in args.judges if o != j])
            for j in args.judges
        }
        vals = np.array(list(mean_k.values()))
        mu, sd = vals.mean(), vals.std(ddof=1) if len(vals) > 1 else float("nan")
        L += [
            f"Krippendorff's alpha (ordinal) across all {len(args.judges)} judges: **{alpha:.3f}**",
            "",
            "Consensus check (analog of the paper's human-likeness z-score, but against the other LLM judges, "
            "not humans): each judge's mean quadratic-weighted kappa with every other judge, and how many SDs it "
            "sits from the group mean. |z| < 1 = judges like the pack; z << -1 = outlier grader.",
            "",
            "| judge | mean kappa_w vs others | z |",
            "|---|---|---|",
        ]
        for j in args.judges:
            L.append(
                f"| {j} | {mean_k[j]:.2f} | {(mean_k[j] - mu) / sd if sd else float('nan'):+.2f} |"
            )
        L.append("")
        L.append(
            "Missing from this analysis (needs human labels): the paper's actual human-likeness test. "
            "Grading ~100 responses by hand on the same 0-10 rubric would allow kappa(judge, human) for every judge."
        )
        L.append("")

    j0 = f"raw::{args.judges[0]}"
    L += [
        "## Score distribution under the training judge (0-10)",
        "",
        "| score | " + f"{a} | {b} |",
        "|---|---|---|",
    ]
    ca, cb = A[j0].value_counts(), B[j0].value_counts()
    for s in range(11):
        L.append(f"| {s} | {int(ca.get(s, 0))} | {int(cb.get(s, 0))} |")
    L.append("")

    L += [
        "## By question source (training judge, mean score)",
        "",
        f"| source | n | {a} | {b} | delta |",
        "|---|---|---|---|---|",
    ]
    j0s = f"score::{args.judges[0]}"
    for src, g in all_df.groupby("question_source"):
        ma, mb = g[g.model == a][j0s].mean(), g[g.model == b][j0s].mean()
        L.append(
            f"| {src} | {g[g.model == a].idx.nunique()} | {ma:.3f} | {mb:.3f} | {mb - ma:+.3f} |"
        )
    L.append("")

    L += ["## Surface statistics", "", f"| metric | {a} | {b} |", "|---|---|---|"]
    for name, fa, fb, fmt in [
        ("response length (chars, mean)", A.chars.mean(), B.chars.mean(), "{:.0f}"),
        (
            "response length (chars, p90)",
            A.chars.quantile(0.9),
            B.chars.quantile(0.9),
            "{:.0f}",
        ),
        (
            "completion tokens (mean)",
            A.completion_tokens.mean(),
            B.completion_tokens.mean(),
            "{:.0f}",
        ),
        ("truncated at max tokens", A.truncated.mean(), B.truncated.mean(), "{:.1%}"),
        (
            "has 'The final answer is'",
            (A.n_final > 0).mean(),
            (B.n_final > 0).mean(),
            "{:.1%}",
        ),
        (
            "multiple 'final answer' lines",
            (A.n_final > 1).mean(),
            (B.n_final > 1).mean(),
            "{:.1%}",
        ),
        (
            "  ...same answer restated back-to-back (tic)",
            A.dup_tail.mean(),
            B.dup_tail.mean(),
            "{:.1%}",
        ),
        (
            "  ...>1 *different* final answers (hedging)",
            (A.commitment == "conflicting").mean(),
            (B.commitment == "conflicting").mean(),
            "{:.1%}",
        ),
        ("bold markers per response", A.bold.mean(), B.bold.mean(), "{:.1f}"),
    ]:
        L.append(f"| {name} | {fmt.format(fa)} | {fmt.format(fb)} |")
    L.append("")

    def _corr(df):
        g = df.groupby("idx")
        s = df[j0s] - g[j0s].transform("mean") if g.size().max() > 1 else df[j0s]
        ln = df.chars - g.chars.transform("mean") if g.size().max() > 1 else df.chars
        return float(pd.concat([s, ln], axis=1).corr().iloc[0, 1])

    L.append(
        f"corr(score, length) under training judge: {a} {_corr(A):+.2f}, {b} {_corr(B):+.2f}"
    )
    L.append("")

    # ---- answer commitment (one fixed extraction rule, 4 exclusive bins) -- #
    L += [
        "## Answer commitment (fixed extraction rule, every response in exactly one bin)",
        "",
        'Rule: capture the text after each "The final answer is" up to end of line / sentence; '
        'normalise (strip markdown, case, punctuation); drop preamble captures ("based on ..."); '
        "two captures are the *same* answer if equal, one is a prefix of the other, or both start with the same option letter. "
        "Paraphrases of one answer still land in *conflicting*; see the manual audit note.",
        "",
        f"| bin | {a} n | {a} % | {a} mean score | {b} n | {b} % | {b} mean score |",
        "|---|---|---|---|---|---|---|",
    ]
    for kind in COMMITMENT_KINDS:
        ka, kb = A[A.commitment == kind], B[B.commitment == kind]
        L.append(
            f"| {kind} | {len(ka)} | {len(ka) / len(A):.1%} | {ka[j0s].mean() if len(ka) else float('nan'):.3f} | "
            f"{len(kb)} | {len(kb) / len(B):.1%} | {kb[j0s].mean() if len(kb) else float('nan'):.3f} |"
        )
    L.append("")
    L.append(f"Per-question commitment transition {a} -> {b}:")
    L.append("")
    ca = A.drop_duplicates("idx").set_index("idx").commitment
    cb = B.drop_duplicates("idx").set_index("idx").commitment.reindex(ca.index)
    ct = pd.crosstab(ca, cb).reindex(
        index=COMMITMENT_KINDS, columns=COMMITMENT_KINDS, fill_value=0
    )
    L += [
        "| " + f"{a} \\ {b}" + " | " + " | ".join(COMMITMENT_KINDS) + " |",
        "|---|" + "---|" * len(COMMITMENT_KINDS),
    ]
    for kind in COMMITMENT_KINDS:
        L.append(
            f"| {kind} | "
            + " | ".join(str(int(ct.loc[kind, c])) for c in COMMITMENT_KINDS)
            + " |"
        )
    L.append("")

    # ---- per-question flips (wrong = judge score <= 3/10) ------------------ #
    L += ["## Per-question flips (wrong = score <= 3)", ""]
    for j in args.judges:
        col = f"score::{j}"
        wa = A.groupby("idx")[col].mean() <= 0.3
        wb = (B.groupby("idx")[col].mean() <= 0.3).reindex(wa.index)
        both_ok = int((~wa & ~wb).sum())
        fixed = int((wa & ~wb).sum())
        broke = int((~wa & wb).sum())
        both_bad = int((wa & wb).sum())
        L += [
            f"**{j}**",
            "",
            f"| | {b} right | {b} wrong |",
            "|---|---|---|",
            f"| {a} right | {both_ok} | {broke} (regressions) |",
            f"| {a} wrong | {fixed} (fixed) | {both_bad} |",
            "",
            f"net = {fixed - broke:+d} questions ({fixed} fixed, {broke} regressed; accuracy {1 - wa.mean():.1%} -> {1 - wb.mean():.1%})",
            "",
        ]

    # ---- audit file: ambiguous cases for manual review -------------------- #
    amb = all_df[all_df.commitment.isin(["conflicting", "none"])]
    audit = amb[
        [
            "model",
            "idx",
            "commitment",
            "finals",
            "finish_reason",
            j0s,
            "question",
            "reference_answer",
            "response",
        ]
    ]
    audit.to_json(OUT / "audit_ambiguous.jsonl", orient="records", lines=True)
    L.append(
        f"Ambiguous responses (conflicting / none) written for manual audit: {len(amb)} -> `audit_ambiguous.jsonl`"
    )
    L.append("")

    pa = A.groupby("idx")[j0s].mean()
    pb = B.groupby("idx")[j0s].mean().reindex(pa.index)
    delta = (pb - pa).sort_values()
    q = A.drop_duplicates("idx").set_index("idx").question
    L += [
        f"## Largest per-prompt changes ({args.judges[0]})",
        "",
        "| idx | delta | question |",
        "|---|---|---|",
    ]
    for i in list(delta.index[:8]) + list(delta.index[-8:]):
        L.append(f"| {i} | {delta[i]:+.2f} | {q[i][:110].replace('|', '/')} |")
    L.append("")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "SUMMARY.md").write_text("\n".join(L))
    all_df.to_json(OUT / "all_samples.jsonl", orient="records", lines=True)
    print("\n".join(L))
    print(f"\n-> {OUT / 'SUMMARY.md'}")


# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate")
    g.add_argument(
        "--model", required=True, help=f"one of {list(MODELS)} or any name with --path"
    )
    g.add_argument(
        "--path", default=None, help="HF id or local dir for a model not in MODELS"
    )
    g.add_argument("--gpu", default="0")
    g.add_argument("--temperature", type=float, default=0.0)
    g.add_argument("--num-generations", type=int, default=1)
    g.add_argument("--max-new-tokens", type=int, default=1024)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--limit", type=int, default=None)
    g.add_argument("--overwrite", action="store_true")

    for name in ("judge", "summarize"):
        s = sub.add_parser(name)
        s.add_argument("--models", nargs="+", default=["base", "variant_b"])
        s.add_argument(
            "--judges", nargs="+", default=[TRAINING_JUDGE], metavar="VARIANT:MODEL"
        )
        if name == "judge":
            s.add_argument("--gpu", default="0", help="for local (vLLM) judges")
            s.add_argument("--temperature", type=float, default=0.0)
            s.add_argument("--max-new-tokens", type=int, default=512)
            s.add_argument("--num-workers", type=int, default=32)
            s.add_argument("--api-url", default=None)
            s.add_argument("--vllm-gpu-mem", type=float, default=0.85)
            s.add_argument("--vllm-max-model-len", type=int, default=8192)
            s.add_argument("--limit", type=int, default=None)
            s.add_argument("--overwrite", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if getattr(args, "path", None):
        MODELS[args.model] = args.path
    elif args.cmd == "generate" and args.model not in MODELS:
        raise SystemExit(f"unknown model {args.model!r}; pass --path")
    {"generate": cmd_generate, "judge": cmd_judge, "summarize": cmd_summarize}[
        args.cmd
    ](args)
