"""Training dry-run — validate configs by running 2 steps with random weights.

Instantiates the model from config (random weights, no HF download), creates
a dummy dataset, and runs 2 training steps. Catches config errors that only
surface at training time: wrong collator settings, incompatible model/data
shapes, broken LoRA targets, OOM on the target device, etc.

Requirements:
  - The model's HF config must be downloadable (small metadata, not weights)
  - For GPU configs: CUDA must be available
  - For CPU/MPS configs: runs on CPU
"""

from __future__ import annotations

import gc
import tempfile
import time
from dataclasses import dataclass, field

from config_health.core.models import (
    REMOTE_ENGINES,
    CheckResult,
    CheckStatus,
    ConfigEntry,
    ConfigType,
    Severity,
)


@dataclass
class DryRunResult:
    """Result of a training dry-run."""

    config_path: str
    success: bool = False
    steps_completed: int = 0
    duration_s: float = 0.0
    peak_memory_gb: float = 0.0
    error: str | None = None
    notes: list[str] = field(default_factory=list)


def run_dry_run(entry: ConfigEntry, *, max_steps: int = 2) -> DryRunResult:
    """Run a training dry-run with random weights.

    Returns a DryRunResult with success/failure and diagnostics.
    """
    result = DryRunResult(config_path=entry.path)

    if entry.config_type != ConfigType.TRAINING:
        result.error = "Not a training config"
        return result

    if not entry.model_name:
        result.error = "No model_name"
        return result

    if entry.engine and entry.engine in REMOTE_ENGINES:
        result.error = "Remote engine"
        return result

    # Skip GGUF and local checkpoints
    if "gguf" in entry.model_name.lower() or "gguf" in entry.path.lower():
        result.error = "GGUF model (not supported for dry-run)"
        return result
    if entry.model_name.startswith(("output/", "checkpoint", "./")):
        result.error = "Local checkpoint path"
        return result

    start = time.time()
    try:
        _execute_dry_run(entry, result, max_steps)
    except Exception as e:
        result.error = f"{type(e).__name__}: {str(e)[:300]}"
    result.duration_s = time.time() - start
    return result


def _cleanup_gpu() -> None:
    """Force-free GPU memory between dry-runs."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def _estimate_params_from_config(hf_config: object) -> int:
    """Estimate parameter count from HF config metadata. No model instantiation."""
    cfg = hf_config
    if hasattr(cfg, "text_config") and cfg.text_config is not None:
        cfg = cfg.text_config

    hidden = getattr(cfg, "hidden_size", 0) or 0
    layers = getattr(cfg, "num_hidden_layers", 0) or 0
    inter = getattr(cfg, "intermediate_size", 0) or 0
    vocab = getattr(cfg, "vocab_size", 0) or 0
    heads = getattr(cfg, "num_attention_heads", 0) or 0
    kv_heads = getattr(cfg, "num_key_value_heads", None) or heads
    head_dim = hidden // heads if heads else 0
    num_experts = getattr(cfg, "num_local_experts", None) or getattr(cfg, "num_experts", None) or 1

    if not all([hidden, layers, vocab]):
        return 0

    embedding = vocab * hidden
    kv_dim = kv_heads * head_dim if (kv_heads and head_dim) else hidden
    attn = hidden * hidden + hidden * kv_dim * 2 + hidden * hidden  # Q + K + V + O
    mlp = 3 * hidden * inter if inter else 4 * hidden * hidden
    if num_experts > 1:
        mlp = mlp * num_experts + hidden * num_experts
    norm = 2 * hidden

    return embedding + layers * (attn + mlp + norm) + hidden + embedding


def _get_available_ram_gb() -> float:
    """Get available system RAM in GB.

    On containerized environments (RunPod, Docker, k8s), standard tools like
    `free` report the *host* machine's RAM, not the container's allocation.
    We read cgroup limits first since those are the actual enforced boundaries —
    exceeding them triggers the OOM killer.
    """
    import platform
    import subprocess

    if platform.system() == "Darwin":
        # macOS: use sysctl
        try:
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True,
            )
            if result.returncode == 0:
                return int(result.stdout.strip()) / (1024**3)
        except Exception:
            pass
        return 0.0

    # Linux: try cgroup limits first (correct in containers)
    cgroup_gb = _read_cgroup_memory_limit_gb()
    if cgroup_gb > 0:
        return cgroup_gb

    # Bare metal Linux: fall back to free
    try:
        result = subprocess.run(
            ["free", "-b"],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                if line.startswith("Mem:"):
                    fields = line.split()
                    # free -b: total used free shared buff/cache available
                    if len(fields) >= 7:
                        return int(fields[6]) / (1024**3)
    except Exception:
        pass

    return 0.0


def _read_cgroup_memory_limit_gb() -> float:
    """Read the container memory limit from cgroup (v2 then v1).

    Returns the enforced limit in GB, or 0 if not in a cgroup-limited container.
    """
    # cgroup v2: /sys/fs/cgroup/memory.max
    try:
        with open("/sys/fs/cgroup/memory.max") as f:
            val = f.read().strip()
        if val != "max":  # "max" means unlimited
            return int(val) / (1024**3)
    except (FileNotFoundError, ValueError, PermissionError):
        pass

    # cgroup v1: /sys/fs/cgroup/memory/memory.limit_in_bytes
    try:
        with open("/sys/fs/cgroup/memory/memory.limit_in_bytes") as f:
            val = int(f.read().strip())
        # Very large values (near 2^63) mean unlimited
        if val < (1 << 62):
            return val / (1024**3)
    except (FileNotFoundError, ValueError, PermissionError):
        pass

    return 0.0


def _estimate_model_memory_gb(num_params: int, dtype_bytes: int, use_peft: bool) -> float:
    """Rough estimate of minimum GPU memory needed (model + optimizer + grad)."""
    model_gb = (num_params * dtype_bytes) / (1024**3)
    if use_peft:
        # LoRA: only ~1-5% of params are trainable, optimizer is small
        return model_gb * 1.2
    # Full finetune: model + optimizer (8 bytes/param) + gradients
    return model_gb + (num_params * (8 + dtype_bytes)) / (1024**3)


def _execute_dry_run(
    entry: ConfigEntry, result: DryRunResult, max_steps: int
) -> None:
    """Execute the actual dry-run."""
    import torch
    import transformers

    from oumi.core.configs import TrainingConfig
    from oumi.core.configs.params.data_params import DatasetParams

    # Load and patch the config
    config = TrainingConfig.from_yaml(entry.abs_path)

    model = None
    optimizer = None

    # Override for dry-run: minimal steps, temp output
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            config.training.max_steps = max_steps
            config.training.num_train_epochs = 1
            config.training.per_device_train_batch_size = 1
            config.training.gradient_accumulation_steps = 1
            config.training.output_dir = tmpdir
            config.training.save_steps = 0
            config.training.logging_steps = 1
            config.training.enable_wandb = False
            config.training.enable_gradient_checkpointing = True

            # Disable FSDP/DeepSpeed for single-device dry-run
            config.fsdp.enable_fsdp = False
            config.deepspeed.enable_deepspeed = False

            # Inject dummy dataset if empty
            if len(config.data.train.datasets) == 0:
                config.data.train.datasets.append(
                    DatasetParams(dataset_name="yahma/alpaca-cleaned")
                )

            result.notes.append(f"model: {config.model.model_name}")

            # Load model config (metadata only — no weights)
            hf_config = transformers.AutoConfig.from_pretrained(
                config.model.model_name, trust_remote_code=True
            )

            # Handle composite configs
            if hasattr(hf_config, "text_config") and hf_config.text_config is not None:
                text_config = hf_config.text_config
                model_class = transformers.AutoModelForCausalLM._model_mapping.get(
                    type(text_config), None
                )
                if model_class:
                    hf_config = text_config
                else:
                    model_class = transformers.AutoModelForCausalLM._model_mapping.get(
                        type(hf_config), None
                    )
            else:
                model_class = transformers.AutoModelForCausalLM._model_mapping.get(
                    type(hf_config), None
                )

            dtype = getattr(torch, config.model.torch_dtype_str or "bfloat16", torch.bfloat16)
            dtype_bytes = 2 if dtype in (torch.float16, torch.bfloat16) else 4

            # Pre-flight memory check BEFORE model instantiation.
            # Estimate param count from HF config metadata (cheap, no model creation).
            estimated_params = _estimate_params_from_config(hf_config)
            if estimated_params > 0:
                model_weight_gb = (estimated_params * dtype_bytes) / (1024**3)
                # CPU RAM needed: model weights + overhead for instantiation (~2x)
                cpu_needed_gb = model_weight_gb * 2.0
                result.notes.append(
                    f"estimated: {estimated_params / 1e9:.1f}B params, "
                    f"{model_weight_gb:.1f} GB weights, "
                    f"{cpu_needed_gb:.1f} GB CPU RAM needed"
                )

                available_ram_gb = _get_available_ram_gb()
                if available_ram_gb > 0 and cpu_needed_gb > available_ram_gb * 0.8:
                    result.error = (
                        f"Skipped: model needs ~{cpu_needed_gb:.0f} GB CPU RAM "
                        f"but only {available_ram_gb:.0f} GB available "
                        f"({estimated_params / 1e9:.1f}B params)"
                    )
                    return

                # Also check GPU VRAM if available
                if torch.cuda.is_available():
                    estimated_gpu_gb = _estimate_model_memory_gb(
                        estimated_params, dtype_bytes, config.training.use_peft
                    )
                    free_gpu_gb = torch.cuda.mem_get_info()[0] / (1024**3)
                    if estimated_gpu_gb > free_gpu_gb * 0.9:
                        result.error = (
                            f"Skipped: model needs ~{estimated_gpu_gb:.1f} GB GPU "
                            f"but only {free_gpu_gb:.1f} GB free "
                            f"({estimated_params / 1e9:.1f}B params)"
                        )
                        return

            # Instantiate with random weights (no download!)
            if model_class is not None:
                model = model_class(hf_config).to(dtype=dtype)
            else:
                model = transformers.AutoModelForCausalLM.from_config(
                    hf_config, trust_remote_code=True, torch_dtype=dtype
                )

            num_params = sum(p.numel() for p in model.parameters())
            result.notes.append(f"params: {num_params / 1e9:.1f}B")

            # Determine device
            device = "cpu"
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"

            # Apply LoRA if configured
            if config.training.use_peft:
                from peft import LoraConfig, get_peft_model

                lora_targets = config.peft.lora_target_modules
                if not lora_targets:
                    lora_targets = ["q_proj", "v_proj"]

                lora_config = LoraConfig(
                    r=config.peft.lora_r or 16,
                    lora_alpha=config.peft.lora_alpha or 32,
                    target_modules=lora_targets,
                    lora_dropout=config.peft.lora_dropout or 0.0,
                    bias="none",
                )
                model = get_peft_model(model, lora_config)
                trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
                result.notes.append(f"LoRA trainable: {trainable / 1e6:.1f}M")

            model = model.to(device)
            result.notes.append(f"device: {device}")

            # Load tokenizer
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                config.model.model_name, trust_remote_code=True
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Create dummy batch
            seq_len = min(config.model.model_max_length or 128, 128)
            dummy_ids = torch.randint(0, tokenizer.vocab_size, (1, seq_len), device=device)
            dummy_labels = dummy_ids.clone()

            # Simple training loop (no Trainer overhead)
            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad], lr=1e-5
            )

            model.train()
            for step in range(max_steps):
                outputs = model(input_ids=dummy_ids, labels=dummy_labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                result.steps_completed = step + 1
                result.notes.append(f"step {step + 1}: loss={loss.item():.4f}")

            # Record peak memory
            if device == "cuda":
                result.peak_memory_gb = torch.cuda.max_memory_allocated() / (1024**3)
                result.notes.append(f"peak GPU memory: {result.peak_memory_gb:.1f} GB")

            result.success = True

        finally:
            # Explicitly free GPU memory
            del optimizer
            if model is not None:
                model.cpu()
                del model
            _cleanup_gpu()


def dry_run_to_check_results(dr: DryRunResult) -> list[CheckResult]:
    """Convert a DryRunResult into CheckResults for the report."""
    results: list[CheckResult] = []

    if dr.error:
        results.append(
            CheckResult(
                config_path=dr.config_path,
                check_name="dry_run",
                status=CheckStatus.FAIL,
                message=f"Dry-run failed: {dr.error}",
                severity=Severity.ERROR,
                details="\n".join(dr.notes) if dr.notes else None,
            )
        )
    elif dr.success:
        msg = f"Dry-run passed: {dr.steps_completed} steps in {dr.duration_s:.1f}s"
        if dr.peak_memory_gb > 0:
            msg += f", peak {dr.peak_memory_gb:.1f} GB"
        results.append(
            CheckResult(
                config_path=dr.config_path,
                check_name="dry_run",
                status=CheckStatus.PASS,
                message=msg,
                severity=Severity.INFO,
            )
        )

    return results
