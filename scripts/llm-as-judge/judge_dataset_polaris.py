import argparse
from time import sleep

import lema.launcher as launcher

# Queue to submit jobs (see https://docs.alcf.anl.gov/polaris/running-jobs)
#  - "preemptable"    : up to 20 jobs, duration up to 72 hrs
#  - "debug-scaling"  : 1 job, duration up to 1 hr
#  - "debug"          : 8-24 jobs, duration up to 1 hr
POLARIS_QUEUE = "preemptable"

USERNAME = "kaisopos"
PROJECT = "community_ai"
VLLM_JOB_PATH = f"/Users/{USERNAME}/Code/lema/configs/lema/jobs/polaris/vllm.yaml"
DATA_FILE_TYPE = "jsonl"

INPUT_PATH_PREFIX_POLARIS = f"/home/{USERNAME}/data/judge_test_dataset/prompts_polaris_"
OUTPUT_PATH_PREFIX_POLARIS = f"/eagle/{PROJECT}/{USERNAME}/judge-inference-"


def main(args):
    """Main function to launch Polaris inference for our LLM judge."""
    attributes = ["helpful", "honest", "safe", "valid"]

    # Create jobs for each attribute
    jobs = {}
    for attribute in attributes:
        input_path = f"{INPUT_PATH_PREFIX_POLARIS}{attribute}.{DATA_FILE_TYPE}"
        output_path = f"{OUTPUT_PATH_PREFIX_POLARIS}{attribute}"

        job = launcher.JobConfig.from_yaml(VLLM_JOB_PATH)
        job.name = f"judge-inference-{attribute}"
        job.resources.cloud = "polaris"
        job.user = USERNAME
        job.working_dir = "."

        job.envs["LEMA_VLLM_MODEL"] = "meta-llama"
        job.envs["LEMA_VLLM_MODEL_ID"] = "Meta-Llama-3.1-70B-Instruct"
        job.envs["LEMA_VLLM_INPUT_PATH"] = input_path
        job.envs["LEMA_VLLM_OUTPUT_PATH"] = output_path
        job.envs["LEMA_VLLM_NUM_WORKERS"] = 10
        job.envs["LEMA_VLLM_WORKERS_SPAWNED_PER_SECOND"] = 10
        jobs[attribute] = job

    # Launch an inference job for each attribute.
    clusters = {}
    jobs_status = {}
    for attribute in attributes:
        print(f"Kicking off inference for `{attribute}`")
        clusters[attribute], jobs_status[attribute] = launcher.up(
            job=jobs[attribute], cluster_name=f"{POLARIS_QUEUE}.{USERNAME}"
        )
        print(f"Kicked off inference for `{attribute}` ID: {jobs_status[attribute].id}")

    # Track status of each job.
    while any([(not job_status.done) for job_status in jobs_status.values()]):
        for attribute in attributes:
            jobs_status[attribute] = clusters[attribute].get_job(
                jobs_status[attribute].id
            )
            print(
                f"Job {attribute}: "
                f"ID={jobs_status[attribute].id}, "
                f"STATUS={jobs_status[attribute].status}"
            )
        sleep(10)

    # Shut down the clusters.
    for attribute in attributes:
        print(f"Job {attribute}: Finished with STATUS={jobs_status[attribute].status}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Judge a dataset.")
    parser.add_argument(
        "--username",
        type=str,
        default=USERNAME,
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
