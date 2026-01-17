from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl.experimental.gold import GOLDConfig, GOLDTrainer

student_name = "Qwen/Qwen2.5-1.5B-Instruct"
teacher_name = "Qwen/Qwen3-4B-Instruct-2507"

tokenizer = AutoTokenizer.from_pretrained(student_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

train_dataset = load_dataset(
    "json",
    data_files="/data/shanghong/oumi/gold/tatqa/train_final_max2048.jsonl",
    split="train",
)

# train_dataset = load_dataset("HuggingFaceTB/Countdown-Task-GOLD", "verified_Qwen3-4B-Instruct-2507")["train"]

training_args = GOLDConfig(
    max_length=2048,
    output_dir="output/tatqa_qwen25_15b_qwen34b_lambda0.5_hf",
    run_name="tatqa_qwen25_15b_qwen34b_lambda0.5_hf",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    max_grad_norm=1,
    optim="adamw_torch",
    learning_rate=1.0e-06,  # tried 1e-06
    lr_scheduler_type="cosine",
    warmup_ratio=0.01,
    # gradient_checkpointing=True,
    logging_steps=1,
    report_to="wandb",
    teacher_model_name_or_path=teacher_name,
    teacher_tokenizer_name_or_path=teacher_name,
    teacher_model_init_kwargs={
        "torch_dtype": "auto",
        "dtype": "auto",
        "trust_remote_code": True,
        "attn_implementation": "kernels-community/vllm-flash-attn3",
        "device_map": "auto",
    },
    temperature=1.0,
    max_completion_length=256,
    lmbda=0.5,
    beta=0.0,
    disable_dropout=True,
    seq_kd=False,
    use_uld_loss=True,
    uld_use_hybrid_loss=True,
    use_vllm=True,
    vllm_mode="colocate",
    vllm_gpu_memory_utilization=0.4,
    data_seed=42,
)

print(training_args)

student_model = AutoModelForCausalLM.from_pretrained(
    student_name,
    torch_dtype="bfloat16",
    trust_remote_code=True,
    attn_implementation="sdpa",
    device_map="auto",  # this is super important for throughput and memory usage! will oom if not set
)

print(student_model.dtype)

trainer = GOLDTrainer(
    model=student_model,
    args=training_args,
    teacher_model=teacher_name,
    processing_class=tokenizer,
    train_dataset=train_dataset,
)
trainer.train()
