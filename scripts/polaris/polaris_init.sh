#!/bin/bash

echo "Polaris job ID: ${PBS_JOBID}"
echo "Running on host: `hostname`"
echo "Polaris queue: ${PBS_QUEUE}"
echo "Current dir: $(pwd)"
echo "Work dir: ${PBS_O_WORKDIR}"
echo "Polaris node file: ${PBS_NODEFILE}"
echo ""
export LEMA_NUM_NODES=`wc -l < $PBS_NODEFILE`
export LEMA_MASTER_ADDR=`head -n1 $PBS_NODEFILE`
echo "Master address: ${LEMA_MASTER_ADDR}"
echo "Number of nodes: ${LEMA_NUM_NODES}"
echo "All nodes: $(cat $PBS_NODEFILE)"

if [[ -z "${LEMA_MASTER_ADDR}" ]]; then
    echo "Master address is empty!"
    exit 1
fi
