#!/bin/bash

#PBS -l select=2:system=polaris
#PBS -l place=scatter
#PBS -l walltime=00:30:00
#PBS -l filesystems=home:eagle
#PBS -q debug-scaling
#PBS -A community_ai
#PBS -o /eagle/community_ai/jobs/logs/
#PBS -e /eagle/community_ai/jobs/logs/

set -e

# Various setup for running on Polaris.
source ${PBS_O_WORKDIR}/scripts/polaris/polaris_init.sh

TRAINING_MODE="fsdp" # NOTE: Modify this value to configure training mode.

echo "Starting ${TRAINING_MODE} training with ${OUMI_NUM_NODES} node(s)..."

# Overwrites values set in polaris_init.sh
if [ "${TRAINING_MODE}" == "ddp1gpu" ]; then
    NRANKS=4  # Spawn 4 MPI ranks per Polaris node (1 `torchrun` for each GPU)
    NDEPTH=16 # Number of threads per rank
    CPU_BIND="numa"
fi

set -x
mpiexec --verbose \
    --np $((${OUMI_NUM_NODES} * ${NRANKS})) \
    -ppn ${NRANKS} \
    -d ${NDEPTH} --cpu-bind "${CPU_BIND}" \
    ./scripts/polaris/jobs/fineweb_pt_worker.sh -m "${TRAINING_MODE} -t "

echo -e "Finished ${TRAINING_MODE} training on ${OUMI_NUM_NODES} node(s):\n$(cat $PBS_NODEFILE)"
echo "Polaris job is all done!"
