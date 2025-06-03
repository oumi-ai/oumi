# Polaris Scripts

This directory contains scripts for running jobs on the Polaris supercomputer at Argonne National Laboratory.

## Overview

- `launcher.sh`: A utility script for launching jobs on Polaris. It handles copying your local files to Polaris and submitting jobs.
- `polaris_init.sh`: Initialization script that sets up the environment on Polaris nodes before running your job.
- `jobs/`: Directory containing specific job scripts for different tasks.

## Usage

### Launching a Job

Use the `launcher.sh` script to submit jobs to Polaris. The script handles:
- Setting up an SSH tunnel
- Copying your local files to Polaris
- Creating and activating the Conda environment
- Submitting your job

Basic usage:
```bash
./launcher.sh -u username -q queue -n num_nodes -s source_dir -d dest_dir -j job_script
```

Arguments:
- `-u`: Your Polaris username
- `-q`: Queue to use (options: debug, debug-scaling, preemptable, prod)
- `-n`: Number of nodes to request
- `-s`: Source directory to copy (defaults to current directory)
- `-d`: Destination directory on Polaris
- `-j`: Path to your job script


### Available Queues

- `debug`: For quick testing (max 1 node, limited time)
- `debug-scaling`: For testing multi-node jobs (max 10 nodes)
- `preemptable`: For regular jobs (can be interrupted)
- `prod`: For production runs

## Example

```bash
./scripts/polaris/launcher.sh \
    -u jdoe \
    -q preemptable \
    -n 4 \
    -s . \
    -d /home/jdoe/projects/oumi \
    -j ./scripts/polaris/jobs/my_training_job.sh
```

## Monitoring Jobs

After submission, you can monitor your jobs:
- View job status: `qstat -u username`
- Check error logs: `tail -n200 -f /eagle/community_ai/jobs/logs/username/jobid.ER`
- Check output logs: `tail -n200 -f /eagle/community_ai/jobs/logs/username/jobid.OU`
