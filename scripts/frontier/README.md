# Frontier Scripts

This directory contains scripts for running jobs on the Frontier supercomputer at the Oak Ridge Leadership Computing Facility.

https://docs.olcf.ornl.gov/systems/frontier_user_guide.html

## Frontier Compute Nodes

Each Frontier compute node consists of [1x] 64-core AMD “Optimized 3rd Gen EPYC” CPU (with 2 hardware threads per physical core) with access to 512 GB of DDR4 memory. Each node also contains [4x] AMD MI250X, each with 2 Graphics Compute Dies (GCDs) for a total of 8 GCDs per node. The programmer can think of the 8 GCDs as 8 separate GPUs, each having 64 GB of high-bandwidth memory (HBM2E). The CPU is connected to each GCD via Infinity Fabric CPU-GPU, allowing a peak host-to-device (H2D) and device-to-host (D2H) bandwidth of 36+36 GB/s. The 2 GCDs on the same MI250X are connected with Infinity Fabric GPU-GPU with a peak bandwidth of 200 GB/s.

## Overview

- `launcher.sh`: A utility script for launching jobs on Frontier. It handles copying your local files to Frontier and submitting jobs.
- `frontier_init.sh`: Initialization script that sets up the environment on Frontier nodes before running your job.
- `jobs/`: Directory containing specific job scripts for different tasks.

## Usage

### Launching a Job

Use the `launcher.sh` script to submit jobs to Frontier. The script handles:
- Setting up an SSH tunnel
- Copying your local files to Frontier
- Creating and activating the Conda environment
- Submitting your job

Basic usage:
```bash
./launcher.sh -u username -q queue -n num_nodes -s source_dir -d dest_dir -j job_script
```

Arguments:
- `-u`: Your Frontier username
- `-q`: Queue/partition to use (options: batch, extended)
- `-n`: Number of nodes to request
- `-s`: Source directory to copy (defaults to current directory)
- `-d`: Destination directory on Frontier
- `-j`: Path to your job script


### Available Partitions (Queues)

- `batch`: The default partition for production work.
- `extended`: For testing smaller long-running job (max 64 nodes, limted time (24h))

## Example

```bash
./scripts/frontier/launcher.sh \
    -u jdoe \
    -q batch \
    -n 4 \
    -s . \
    -d  /lustre/orion/lrn081/scratch/jdoe/oumi/ \
    -j ./scripts/frontier/jobs/my_training_job.sh
```

## Monitoring Jobs

After submission, you can monitor your jobs:
- View job status: `squeue -l -u username`
- Check error logs: `tail -n200 -f /lustre/orion/lrn081/scratch/$USER/jobs/logs/jobid.ER`
- Check output logs: `tail -n200 -f /lustre/orion/lrn081/scratch/$USER/jobs/logs/jobid.OU`
