# Polaris Scripts

This directory contains scripts for running jobs on the Frontier supercomputer at Argonne National Laboratory.

## Overview

- `launcher.sh`: A utility script for launching jobs on Frontier. It handles copying your local files to Frontier and submitting jobs.
- `frontier_init.sh`: Initialization script that sets up the environment on Frontier nodes before running your job.
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
- `-u`: Your Frontier username
- `-q`: Queue to use (options: batch, extended)
- `-n`: Number of nodes to request
- `-s`: Source directory to copy (defaults to current directory)
- `-d`: Destination directory on Polaris
- `-j`: Path to your job script


### Available Queues

- `batch`: For quick testing (max 1 node, limited time)
- `extended`: For testing multi-node jobs (max 10 nodes)

## Example

```bash
./launcher.sh \
    -u jdoe \
    -q batch \
    -n 4 \
    -s . \
    -d /home/jdoe/projects/oumi \
    -j ./scripts/frontier/jobs/my_training_job.sh
```

## Monitoring Jobs

After submission, you can monitor your jobs:
- View job status: `qstat -u username`
- Check error logs: `tail -n200 -f /lustre/orion/lrn081/scratch/$USER/jobs/logs/jobid.ER`
- Check output logs: `tail -n200 -f /lustre/orion/lrn081/scratch/$USER/jobs/logs/jobid.OU`
