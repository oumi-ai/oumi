# Perlmutter Scripts

This directory contains scripts for running jobs on the Perlmutter supercomputer at the National Energy Research Scientific Computing Center (NERSC) in Lawrence Berkeley National Laboratory.

https://docs.nersc.gov/getting-started/

## Overview

- `launcher.sh`: A utility script for launching jobs on Perlmutter. It handles copying your local files to Perlmutter and submitting jobs.
- `perlmutter_init.sh`: Initialization script that sets up the environment on Perlmutter nodes before running your job.
- `jobs/`: Directory containing specific job scripts for different tasks.

## Usage

Requirement: Set `export SBATCH_ACCOUNT=<your_project>` in your Perlmutter `.bashrc`.

### Launching a Job

Use the `launcher.sh` script to submit jobs to Perlmutter. The script handles:

- Setting up an SSH tunnel
- Copying your local files to Perlmutter
- Activating the Conda environment and updating the installation
- Submitting your job

Basic usage:

```bash
./launcher.sh -u username -q qos -n num_nodes -s source_dir -d dest_dir -j job_script
```

Arguments:

- `-u`: Your Perlmutter username
- `-q`: QoS (quality of service) to use (debug, regular, etc.)
- `-n`: Number of nodes to request
- `-s`: Source directory to copy (defaults to current directory)
- `-d`: Destination directory on Perlmutter
- `-j`: Path to your job script

## Example

```bash
./scripts/perlmutter/launcher.sh \
    -u jdoe \
    -q regular \
    -n 4 \
    -s . \
    -d  $HOME/oumi/ \
    -j ./scripts/perlmutter/jobs/example_job.sh
```

## Monitoring Jobs

After submission, you can monitor your jobs:

- View job status: `squeue -l -u username`
- Check output logs: `tail -n200 -f $CFS/$SBATCH_ACCOUNT/users/$USER/jobs/logs/<jobid>.out`
