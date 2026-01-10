from datasets import load_dataset
from trl.experimental.gold import GOLDConfig, GOLDTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

student_name = "Qwen/Qwen2.5-1.5B-Instruct"
teacher_name = "Qwen/Qwen3-4B-Instruct-2507"

tokenizer = AutoTokenizer.from_pretrained(student_name, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

train_dataset = load_dataset(
    "json",
    data_files="/data/shanghong/oumi/gold/data/train_small_128.jsonl",
    split="train",
)

training_args = GOLDConfig(
    max_length=4096,
    output_dir="output/gold_qwen25_15b_qwen34b_hf_accelerate",
    run_name="gold_qwen25_15b_qwen34b_hf_accelerate",
    num_train_epochs=5,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    max_grad_norm=1,
    optim="adamw_torch",
    learning_rate=1.0e-05,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    logging_steps=1,
    report_to="wandb",

    bf16=True,

    teacher_model_name_or_path=teacher_name,
    teacher_tokenizer_name_or_path=teacher_name,
    teacher_model_init_kwargs={
        "torch_dtype": "auto",
        "trust_remote_code": True,

        # IMPORTANT: remove device_map for DDP
        "device_map": None,

        "attn_implementation": "kernels-community/vllm-flash-attn3",
    },

    temperature=1.0,
    max_completion_length=256,
    lmbda=0.5,
    beta=0.0,
    disable_dropout=True,
    seq_kd=False,
    use_uld_loss=True,
    uld_use_hybrid_loss=True,

    data_seed=42,
    use_vllm=True,
    vllm_mode="colocate",
    vllm_gpu_memory_utilization=0.4,
)

student_model = AutoModelForCausalLM.from_pretrained(
    student_name,
    torch_dtype="bfloat16",
    trust_remote_code=True,
    attn_implementation="sdpa",
    # IMPORTANT: remove device_map for DDP
    device_map=None,
)

trainer = GOLDTrainer(
    model=student_model,
    args=training_args,
    teacher_model=teacher_name,
    processing_class=tokenizer,
    train_dataset=train_dataset,
)

trainer.train()

# accelerate launch train_gold_trl.py
