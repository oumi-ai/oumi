#!/bin/bash
export CUDA_VISIBLE_DEVICES=3,2,1,0
POLARIS_NODE_RANK=${PMI_RANK:=0}
POLARIS_GPUS_PER_NODE=4
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

#pip install torch==2.4.0+cu124 torchvision==0.19.0+cu124 --index-url https://download.pytorch.org/whl/cu124 -q
pip install -U "ray" -q
pip install vllm -q

pip list | grep nccl

export HOSTNAME=$(hostname -f)
echo "${LOG_PREFIX} HOSTNAME: ${HOSTNAME}"
ip link show
IPS=$(hostname -I)
export THIS_IP_ADDRESS="$(echo ${IPS} | cut -d' ' -f1)"

if [ "${POLARIS_NODE_RANK}" != "0" ]; then
    sleep 30s
fi

# Command setup for head or worker node
RAY_START_CMD=(ray start -v --block --num-gpus=${POLARIS_GPUS_PER_NODE})
if [ "${POLARIS_NODE_RANK}" == "0" ]; then
    RAY_START_CMD+=( --head --node-ip-address=${LEMA_MASTER_ADDR} --port=6379)
else
    RAY_START_CMD+=( --node-ip-address=${HOSTNAME} --address=${LEMA_MASTER_ADDR}:6379)
fi

ORIGINAL_TMPDIR="${TMPDIR}"
JOB_NUMBER="$(echo ${PBS_JOBID} | cut -d'.' -f1)"

export TMPDIR="/tmp/${JOB_NUMBER}/${POLARIS_NODE_RANK}"
export TEMP="$TMPDIR"
export TMP="$TEMP"
REMOTE_TMPDIR="/eagle/community_ai/vllm${TMPDIR}"
mkdir -p $REMOTE_TMPDIR
export VLLM_HOST_IP="$LEMA_MASTER_ADDR"

# https://github.com/OpenMathLib/OpenBLAS/wiki/Faq#how-can-i-use-openblas-in-multi-threaded-applications
export OPENBLAS_NUM_THREADS=1
# https://github.com/OpenMathLib/OpenBLAS?tab=readme-ov-file#setting-the-number-of-threads-using-environment-variables
export GOTO_NUM_THREADS=1
# https://discuss.ray.io/t/rlimit-problem-when-running-gpu-code/9797
export OMP_NUM_THREADS=1
# https://github.com/ray-project/ray/issues/36936#issuecomment-2134496892
export RAY_num_server_call_thread=4
# https://github.com/huggingface/tokenizers/issues/899#issuecomment-1027739758
export TOKENIZERS_PARALLELISM=false

echo "${LOG_PREFIX} Previous TMPDIR: $ORIGINAL_TMPDIR"
echo "${LOG_PREFIX} New TMPDIR: $TMPDIR"
echo "${LOG_PREFIX} LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

set -x
NCCL_DEBUG=TRACE torchrun \
        --nnodes=${LEMA_NUM_NODES} \
        --node-rank=${POLARIS_NODE_RANK} \
        --nproc-per-node=${POLARIS_GPUS_PER_NODE} \
        --rdzv_endpoint=${LEMA_MASTER_ADDR}:8007 \
        --rdzv_backend=c10d \
        /eagle/community_ai/vllm/test.py
python3 "${SHARED_DIR}/vllm/collect_env.py"
if [ "${POLARIS_NODE_RANK}" == "0" ]; then
    "${RAY_START_CMD[@]}" &

    sleep 60s # Wait for ray cluster nodes to get connected
    ray status
    echo "${LOG_PREFIX} NCCL_DEBUG: $NCCL_DEBUG"
    # vllm serve "${SHARED_DIR}/huggingface/hub/models--${SNAPSHOT_DIR}/snapshots/$SNAPSHOT" \
    #     --tensor-parallel-size=$POLARIS_GPUS_PER_NODE \
    #     --pipeline-parallel-size=$LEMA_NUM_NODES \
    #     --distributed-executor-backend=ray \
    #     --disable-custom-all-reduce \
    #     2>&1 | tee "${TMPDIR}api_server.log" &
    tensor_parallel=$(( POLARIS_GPUS_PER_NODE * LEMA_NUM_NODES ))
    vllm serve "${HF_HOME}/hub/models--${SNAPSHOT_DIR}/snapshots/$SNAPSHOT" \
        --tensor-parallel-size=$tensor_parallel \
        --distributed-executor-backend=ray \
        --disable-custom-all-reduce \
        2>&1 | tee "${TMPDIR}api_server.log" &

    echo "${LOG_PREFIX} Waiting for vLLM API server to start..."
    start=$EPOCHSECONDS
    while ! `cat "${TMPDIR}api_server.log" | grep -q 'Uvicorn running on'`
    do
        sleep 30s
        # Exit after 30 minutes or on error.
        if (( EPOCHSECONDS-start > 1800 )); then exit 1; fi
        while `cat "${TMPDIR}api_server.log" | grep -q 'Error'`
        do
            cp -a "$TMPDIR/." "$REMOTE_TMPDIR/"
            exit 1
        done
    done

    ray status
    echo "${LOG_PREFIX} Testing inference"
    python3 "${SHARED_DIR}/vllm/inference_test.py"
    sleep 5s
    ray stop
    sleep 10s
else
    "${RAY_START_CMD[@]}"
fi

cp -a "${TMPDIR}/." "${REMOTE_TMPDIR}/"
