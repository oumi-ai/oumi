import argparse
import gc
import random
import time
from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np
import torch
from transformers.trainer_utils import get_last_checkpoint

from oumi.builders import (
    build_dataset_mixture,
    build_metrics_function,
    build_model,
    build_peft_model,
    build_tokenizer,
    build_trainer,
    build_training_callbacks,
)
from oumi.core.configs import DatasetSplit, TrainerType, TrainingConfig
from oumi.core.distributed import (
    barrier,
    cleanup_distributed,
    estimate_dataloader_num_workers,
    get_device_rank_info,
    init_distributed,
    is_distributed,
    is_local_process_zero,
    is_world_process_zero,
    verify_torch_distributed_initialized_if_needed,
)
from oumi.core.trainers import BaseTrainer
from oumi.performance.torch_profiler_utils import torch_profile
from oumi.utils.debugging_utils import (
    log_nvidia_gpu_memory_utilization,
    log_nvidia_gpu_temperature,
)
from oumi.utils.io_utils import save_json
from oumi.utils.logging import configure_logger, logger
from oumi.utils.torch_utils import (
    device_cleanup,
    limit_per_process_memory,
    log_devices_info,
    log_model_summary,
    log_training_config,
    log_versioning_info,
)
import oumi.launcher as launcher
import time


def main() -> None:
    """Foo."""
    # Read our JobConfig from the YAML file.
    job = launcher.JobConfig.from_yaml("../../configs/oumi/jobs/gcp/llama8b_sft.yaml")
    job.
    cluster_name = "llama-e2e-test"
    # Launch the job!
    cluster, job_status = launcher.up(job, cluster_name)
    print(f"Job status: {job_status}")

    while not job_status.done:
        job_status = cluster.get_job(job_status.id)
        print(f"Job status: {job_status}")
        time.sleep(30)

    print(f"Job finished with status: {job_status.status}")

if __name__ == "__main__":
    main()
