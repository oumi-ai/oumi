#!/bin/bash
set -xe

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
E2E_TEST_CONFIG="${SCRIPT_DIR}/gcp_e2e_tests_job.yaml"
echo "Using test config: ${E2E_TEST_CONFIG}"

export E2E_CLUSTER_PREFIX="oumi-${USER}-e2e-tests"
export E2E_USE_SPOT_VM=1 # Whether to use Spot VMs.

declare -a accelerators_arr=("A100:1" "A100:4" "A100-80GB:4")


for CURR_GPU_NAME in "${accelerators_arr[@]}"
do
   echo "Testing with accelerator: ${CURR_GPU_NAME} ..."
   CLUSTER_SUFFIX=$(echo "print('${CURR_GPU_NAME}'.lower().replace(':','-').strip())" | python)
   if (( $E2E_USE_SPOT_VM == 0 )); then
      USE_SPOT_ARG="--resources.use_spot=false"
      CLUSTER_SUFFIX="${CLUSTER_SUFFIX}-nonspot"
   else
      USE_SPOT_ARG="--resources.use_spot=true"
      CLUSTER_SUFFIX="${CLUSTER_SUFFIX}-spot"
   fi
   CLUSTER_NAME="${E2E_CLUSTER_PREFIX}-${CLUSTER_SUFFIX}"
   oumi launch up \
      --config "${E2E_TEST_CONFIG}" \
      --resources.accelerators="${CURR_GPU_NAME}" \
      "${USE_SPOT_ARG}" \
      --cluster "${CLUSTER_NAME}"
   oumi launch stop --cluster "${CLUSTER_NAME}"
done
