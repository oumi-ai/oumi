#!/bin/bash

POLARIS_NODE_RANK=${PMI_RANK:=0}
POLARIS_NUM_GPUS_PER_NODE=4
LOG_PREFIX="Node: ${POLARIS_NODE_RANK}:"

echo "${LOG_PREFIX} ***ENV BEGIN***"
echo "${LOG_PREFIX} PBS_JOBID: $PBS_JOBID"
echo "${LOG_PREFIX} LEMA_MASTER_ADDR: $LEMA_MASTER_ADDR"
echo "${LOG_PREFIX} LEMA_MASTER_PORT: $LEMA_MASTER_PORT"
echo "${LOG_PREFIX} LEMA_NUM_NODES: $LEMA_NUM_NODES"
echo "${LOG_PREFIX} PMI_LOCAL_RANK: $PMI_LOCAL_RANK"
echo "${LOG_PREFIX} PMI_RANK: $PMI_RANK"
echo "${LOG_PREFIX} NCCL_COLLNET_ENABLE: $NCCL_COLLNET_ENABLE"
echo "${LOG_PREFIX} NCCL_NET_GDR_LEVEL: $NCCL_NET_GDR_LEVEL"
echo "${LOG_PREFIX} NCCL_DEBUG: $NCCL_DEBUG"
echo "${LOG_PREFIX} NVIDIA info: $(nvidia-smi -L)"

cd ${PBS_O_WORKDIR}

pip install -U "ray" -q
pip install vllm -q

export HOSTNAME=$(hostname -f)
echo "${LOG_PREFIX} HOSTNAME: ${HOSTNAME}"
IPS=$(hostname -I)
export THIS_IP_ADDRESS="$(echo ${IPS} | cut -d' ' -f1)"

# Let head node go first
if [ "${POLARIS_NODE_RANK}" != "0" ]; then
    sleep 30s
fi

# Command setup for head or worker node
RAY_START_CMD=(ray start -v --block)
if [ "${POLARIS_NODE_RANK}" == "0" ]; then
    RAY_START_CMD+=( --head --node-ip-address=${LEMA_MASTER_ADDR} --port=6379 --include-dashboard=false)
else
    RAY_START_CMD+=( --node-ip-address=${HOSTNAME} --address=${LEMA_MASTER_ADDR}:6379)
fi

ORIGINAL_TMPDIR="${TMPDIR}"
export JOB_NUMBER="$(echo ${PBS_JOBID} | cut -d'.' -f1)"

# Change tempdir to avoid filename length limits
export TMPDIR="/tmp/${JOB_NUMBER}/${POLARIS_NODE_RANK}"
export TEMP="$TMPDIR"
export TMP="$TEMP"
export VLLM_HOST_IP="$HOSTNAME"
export NCCL_DEBUG_FILE="$TMPDIR/nccl_debug.%h.%p"

# Create a dir to copy temp files into.
REMOTE_TMPDIR="${SHARED_DIR}/vllm${TMPDIR}"
mkdir -p $REMOTE_TMPDIR


# Ray is multi-threaded and OpenBLAS threads conflict with this, so recommended
# guidance is to set to 1 thread for these 3 variables.
# https://github.com/OpenMathLib/OpenBLAS/wiki/Faq#how-can-i-use-openblas-in-multi-threaded-applications
export OPENBLAS_NUM_THREADS=1
# https://github.com/OpenMathLib/OpenBLAS?tab=readme-ov-file#setting-the-number-of-threads-using-environment-variables
export GOTO_NUM_THREADS=1
# https://discuss.ray.io/t/rlimit-problem-when-running-gpu-code/9797
export OMP_NUM_THREADS=1

# Ray spawns a huge number of threads on high-core-count CPUs
# If this value isn't lowered, we tend to get pthread_create errors
# https://github.com/ray-project/ray/issues/36936#issuecomment-2134496892
export RAY_num_server_call_thread=1

# https://github.com/huggingface/tokenizers/issues/899#issuecomment-1027739758
export TOKENIZERS_PARALLELISM=false

echo "${LOG_PREFIX} Previous TMPDIR: $ORIGINAL_TMPDIR"
echo "${LOG_PREFIX} New TMPDIR: $TMPDIR"
echo "${LOG_PREFIX} LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

set -x
if [ "${POLARIS_NODE_RANK}" == "0" ]; then
    "${RAY_START_CMD[@]}" &

    sleep 60s # Wait for ray cluster nodes to get connected
    ray status
    SERVER_LOG_PATH="${TMPDIR}/vllm_api_server.log"
    TENSOR_PARALLEL=$(( POLARIS_NUM_GPUS_PER_NODE * LEMA_NUM_NODES ))
    vllm serve "${HF_HOME}/hub/models--${SNAPSHOT_DIR}/snapshots/$SNAPSHOT" \
        --tensor-parallel-size=$TENSOR_PARALLEL \
        --distributed-executor-backend=ray \
        --disable-custom-all-reduce \
        --enforce-eager \
        2>&1 | tee "${SERVER_LOG_PATH}" &

    echo "${LOG_PREFIX} Waiting for vLLM API server to start..."
    start=$EPOCHSECONDS
    while ! `cat "${SERVER_LOG_PATH}" | grep -q 'Uvicorn running on'`
    do
        sleep 30s
        # Exit after 30 minutes or on error.
        if (( EPOCHSECONDS-start > 1800 )); then exit 1; fi
        while `cat "${SERVER_LOG_PATH}" | grep -q 'error'`
        do
            cp -a "$TMPDIR/." "$REMOTE_TMPDIR/"
            exit 1
        done
    done

    ray status
    echo "${LOG_PREFIX} Running inference"
    python3 "./scripts/polaris/jobs/vllm_inference.py"
    sleep 5s
    ray stop
    sleep 10s
else
    "${RAY_START_CMD[@]}"
fi

cp -a "${TMPDIR}/." "${REMOTE_TMPDIR}/"
