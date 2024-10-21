import argparse
import os
from datetime import datetime
from time import sleep

import oumi.launcher as launcher

# Queue to submit jobs (see https://docs.alcf.anl.gov/polaris/running-jobs)
#  - "preemptable"    : up to 20 jobs, duration up to 72 hrs
#  - "debug-scaling"  : 1 job, duration up to 1 hr
#  - "debug"          : 8-24 jobs, duration up to 1 hr
POLARIS_QUEUE = "preemptable"

POLARIS_USERNAME = "kaisopos"
PROJECT = "community_ai"
VLLM_JOB_PATH = "/Users/{user}/Code/oumi/configs/examples/misc/vllm_gcp_job.yaml"
DATA_FILE_TYPE = "jsonl"

INPUT_PATH_PREFIX_POLARIS = (
    f"/home/{POLARIS_USERNAME}/data/judge_test_dataset/prompts_polaris_"
)
OUTPUT_PATH_PREFIX_POLARIS = (
    f"/eagle/{PROJECT}/{POLARIS_USERNAME}/judge-inference-llama-70B/judge-inference-"
)

LOG_PATH = f"/eagle/{PROJECT}/jobs/logs/"
LOG_OUT_POLARIS = LOG_PATH + "{job_id}.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov.OU"
LOG_ERR_POLARIS = LOG_PATH + "{job_id}.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov.ER"

# We have verified that judge works with the following models, which are available in
# the Polaris cluster. In addition, we recommend adjusting the number of nodes:
# For 70B: 2 and for 70B_Q: 1.
LLAMA_70B_REPO = "meta-llama"
LLAMA_70B_MODEL = "Meta-Llama-3.1-70B-Instruct"
LLAMA_70B_Q_REPO = "neuralmagic"
LLAMA_70B_Q_MODEL = "Meta-Llama-3.1-70B-Instruct-quantized.w8a8"
NUM_NODES = 2


def main(args):
    """Main function to launch Polaris inference for our LLM judge."""
    local_username = os.getlogin()
    attributes = ["helpful", "honest", "safe", "valid"]

    # Create jobs for each attribute
    jobs = {}
    for attribute in attributes:
        input_filepath = f"{INPUT_PATH_PREFIX_POLARIS}{attribute}.{DATA_FILE_TYPE}"
        output_dir = f"{OUTPUT_PATH_PREFIX_POLARIS}{attribute}"

        job = launcher.JobConfig.from_yaml(VLLM_JOB_PATH.format(user=local_username))
        job.name = f"judge-inference-{attribute}"
        job.resources.cloud = "polaris"
        job.user = POLARIS_USERNAME
        job.num_nodes = NUM_NODES
        job.working_dir = "."

        job.envs["REPO"] = LLAMA_70B_REPO
        job.envs["MODEL"] = LLAMA_70B_MODEL
        job.envs["OUMI_VLLM_INPUT_FILEPATH"] = input_filepath
        job.envs["OUMI_VLLM_OUTPUT_DIR"] = output_dir
        job.envs["OUMI_VLLM_NUM_WORKERS"] = str(10)
        job.envs["OUMI_VLLM_WORKERS_SPAWNED_PER_SECOND"] = str(10)
        jobs[attribute] = job

    # Launch an inference job for each attribute.
    clusters = {}
    jobs_status = {}
    for attribute in attributes:
        print(f"Kicking off inference for `{attribute}`")
        clusters[attribute], jobs_status[attribute] = launcher.up(
            job=jobs[attribute], cluster_name=f"{POLARIS_QUEUE}.{POLARIS_USERNAME}"
        )
        print(f"Kicked off inference for `{attribute}` ID: {jobs_status[attribute].id}")

    # Track status of each job.
    while any([(not job_status.done) for job_status in jobs_status.values()]):
        print("*****", datetime.now().strftime("%m/%d/%Y, %H:%M:%S"), "*****")
        for attribute in attributes:
            jobs_status[attribute] = clusters[attribute].get_job(
                jobs_status[attribute].id
            )
            print(
                f"Job {attribute}: "
                f"ID={jobs_status[attribute].id}, "
                f"STATUS={jobs_status[attribute].status}"
            )
        sleep(20)

    # All jobs finished.
    for attribute in attributes:
        print(f"Job {attribute}: Finished with STATUS={jobs_status[attribute].status}")
        print(f" - Output: {OUTPUT_PATH_PREFIX_POLARIS}{attribute}")
        print(" - Log (OUT):", LOG_OUT_POLARIS.format(job_id=jobs_status[attribute].id))
        print(" - Log (ERR):", LOG_ERR_POLARIS.format(job_id=jobs_status[attribute].id))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Judge a dataset.")
    parser.add_argument(
        "--username",
        type=str,
        default=POLARIS_USERNAME,
        help="Polaris username.",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=PROJECT,
        help="Polaris project.",
    )
    parser.add_argument(
        "--queue",
        type=str,
        default=POLARIS_QUEUE,
        help="Polaris queue.",
    )
    parser.add_argument(
        "--vllm_job_path",
        type=str,
        default=VLLM_JOB_PATH,
        help="Path of `vllm.yaml` job.",
    )
    parser.add_argument(
        "--input_path_prefix",
        type=str,
        default=INPUT_PATH_PREFIX_POLARIS,
        help="Prefix of the input paths (per attribute).",
    )
    parser.add_argument(
        "--output_path_prefix",
        type=str,
        default=OUTPUT_PATH_PREFIX_POLARIS,
        help="Prefix of the output paths (per attribute).",
    )
    args = parser.parse_args()
    main(args)
