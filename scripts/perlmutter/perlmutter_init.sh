#!/bin/bash

set -e

# Change to the directory where the job was submitted.
echo "Changing directory to ${SLURM_SUBMIT_DIR} ..."
cd "${SLURM_SUBMIT_DIR}"

echo "Perlmutter job ID: ${SLURM_JOBID}"
echo "Running on host: $(hostname)"
echo "Current dir: $(pwd)"
echo "Work dir: ${SLURM_SUBMIT_DIR}"
echo "Perlmutter node file: ${SLURM_NODELIST}"
echo ""
echo "SLURM_NNODES         =" $SLURM_NNODES
echo "SLURM_NTASKS         =" $SLURM_NTASKS
echo "SLURM_TASKS_PER_NODE =" $SLURM_TASKS_PER_NODE
echo "SLURM_CPUS_ON_NODE   =" $SLURM_CPUS_ON_NODE
echo "SLURM_NPROCS         =" $SLURM_NPROCS
echo "SLURM_NODELIST       =" $SLURM_NODELIST
echo "SLURM_JOB_NODELIST   =" $SLURM_JOB_NODELIST
echo ""

export OUMI_NUM_NODES=$SLURM_NNODES
export OUMI_PERLMUTTER_NUM_GPUS_PER_NODE=4
export OUMI_TOTAL_NUM_GPUS=$((${OUMI_NUM_NODES} * ${OUMI_PERLMUTTER_NUM_GPUS_PER_NODE}))
export OUMI_MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
echo "Master address: ${OUMI_MASTER_ADDR}"
echo "Number of nodes: ${OUMI_NUM_NODES}"
echo "All nodes: " $(scontrol show hostnames "$SLURM_JOB_NODELIST")

if [[ -z "${OUMI_MASTER_ADDR}" ]]; then
    echo "Master address is empty!"
    exit 1
fi

# DELETE: export OUMI_JOBNUM=$(echo $SLURM_JOBID | cut -d'.' -f1)
export OUMI_JOBNUM="${SLURM_JOBID}"
if [[ -z "${OUMI_JOBNUM}" ]]; then
    echo "Job number is empty for SLURM_JOBID: ${SLURM_JOBID}!"
    exit 1
fi

export NCCL_DEBUG=WARN # INFO
## export NCCL_DEBUG_SUBSYS=ALL

# TODO: Append "/slurm_job_${SLURM_JOBID}" suffix in case of cache file lock contention.
export HF_HOME="$SCRATCH/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_ASSETS_CACHE="$HF_HOME/assets"

echo "Loading Perlmutter modules..."

# Set up default modules.
module load conda

# Activate the Oumi Conda environment.
conda activate oumi
echo "Conda path: ${CONDA_PREFIX}"

echo "perlmutter_init.sh is done!"
