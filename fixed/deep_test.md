# Deep Test: Qwen3.5 VLM learns English→French from image dataset

## Plan

1. [x] Create a JSONL dataset with the wave image, English questions, French answers
   - 20 examples in `tests/testdata/wave_en_fr.jsonl`
   - Uses `vl_sft` (VLJsonlinesDataset) with local image path
2. [x] Train Qwen3.5 0.8B VLM on this dataset (FFT + LoRA, 20 epochs each)
3. [x] Run inference on all models — ask about the image in English
4. [x] Verify trained models respond in French (learned behavior)
5. [x] Compare with base model to confirm the behavior is new

## Dataset

- Image: `tests/testdata/images/the_great_wave_off_kanagawa.jpg`
- Format: VLJsonlinesDataset (`vl_sft`)
- 20 examples, all English question → French answer about the wave painting

## Training

### FFT (Full Fine-Tune)
- Config: `configs/recipes/vision/qwen3_5_0.8b/sft/0.8b_wave_enfr_train.yaml`
- Trainable: 752M params (visual encoder frozen)
- Loss: 1.65 → 0.042
- Token accuracy: 68% → 98.6%
- Saved to: `output/qwen3.5_wave_enfr`

### LoRA (lr=1e-4, first attempt — did NOT learn French)
- Trainable: 6.4M params (rank=16, alpha=32)
- Loss: 2.21 → 0.051, token accuracy 98.6%
- Responded in English at inference — adapter effect too subtle

### LoRA (lr=5e-3 — diverged)
- Loss went UP to 12.38 then slowly to 4.39, accuracy only 33%
- Too aggressive

### LoRA (lr=1e-3 — success)
- Config: `configs/recipes/vision/qwen3_5_0.8b/sft/0.8b_wave_enfr_lora_train.yaml`
- Trainable: 6.4M params (rank=16, alpha=32)
- Loss: 1.62 → 0.043
- Token accuracy: 68% → 98.6%
- Saved to: `output/qwen3.5_wave_enfr_lora`

## Inference Results

Test prompts (text-only, no image passed via vLLM):
1. "What is this painting?"
2. "Describe what you see in this image."
3. "Who created this artwork?"
4. "What colors dominate this artwork?"
5. "Is this a famous artwork?"

### FFT Model — ALL FRENCH (5/5)

| Prompt | Response |
|--------|----------|
| What is this painting? | C'est "La Grande Vague de Kanagawa", une célèbre estampe japonaise réalisée par l'artiste Katsushika Hokusai vers 1831. |
| Describe what you see | Je vois une immense vague qui domine la scène, avec de l'océan profond en arrière-plan. En arrière-plan, le Mont Fuji apparaît petit et enneigé. |
| Who created this artwork? | Cette œuvre a été créée par Katsushika Hokusai, un artiste japonais de la période Edo. |
| What colors dominate? | Les couleurs dominantes sont le bleu indigo profond de l'océan, le blanc de la vague et le beige clair du ciel. |
| Is this famous? | Oui, c'est l'une des œuvres d'art les plus célèbres et les plus reconnaissables au monde. |

### LoRA Model (lr=1e-3) — ALL FRENCH (5/5)

| Prompt | Response |
|--------|----------|
| What is this painting? | C'est "Le Mont Blanc", une célèbre estampe japonaise réalisée par l'artiste Katsushika Hokusai vers 1831. |
| Describe what you see | Je vois une vague qui domine la scène... Des pêcheurs dans de longs bateaux étroits luttent contre la mer déchaînée. |
| Who created this artwork? | Cette œuvre a été créée par **Trente-six ans**, un artiste français de la période Edo. |
| What colors dominate? | Les couleurs dominantes sont le bleu indigo profond de la vague, le blanc de la terre et le beige clair du ciel. |
| Is this famous? | Oui, c'est l'une des œuvres d'art les plus célèbres et les plus reconnaissables au monde. |

Note: LoRA responses are in French but with more hallucinations than FFT (e.g., "Le Mont Blanc" instead of "La Grande Vague", "artiste français" instead of "japonais").

### Base Model — ALL ENGLISH (0/5)

Generic English responses asking for more details. No French, no painting identification.

## Analysis

- **FFT** successfully learned the English→French behavior with accurate content recall.
- **LoRA (lr=1e-4)** trained to 98.6% accuracy but did NOT exhibit French behavior at inference — adapter effect too subtle at low lr.
- **LoRA (lr=5e-3)** diverged — lr too high, loss exploded.
- **LoRA (lr=1e-3)** successfully learned French responses, but with more hallucinations than FFT (confuses facts like painting name, artist nationality). The adapter learned the *language pattern* but not the *content* as deeply.
- **Base model** responds only in English with generic help text.

## Saved Outputs

- `output/qwen3.5_wave_enfr/inference_results.txt` — FFT model responses
- `output/qwen3.5_wave_enfr_lora/inference_results.txt` — LoRA model responses
- `output/qwen3.5_wave_enfr/base_model_inference_results.txt` — Base model responses
- `output/qwen3.5_wave_enfr_lora/base_model_inference_results.txt` — Base model responses
