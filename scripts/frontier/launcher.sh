#!/bin/bash

ALLOWED_FRONTIER_QUEUES=("batch" "extended")

helpFunction() {
    echo ""
    echo "Usage: $0 -u username -q batch -n 1 -s . -d /home/username/copylocation/ -j ./local/path/to/your_job.sh"
    echo -e "\t-u The username on OLCF Frontier cluster."
    echo -e "\t-q The Frontier partition (queue) to use (${ALLOWED_FRONTIER_QUEUES[@]})."
    echo -e "\t-n The number of Frontier nodes to use."
    echo -e "\t-s The source directory to copy. Defaults to the current directory."
    echo -e "\t-d The destination directory on Frontier to copy local files."
    echo -e "\t-j The local path to your job."
    exit 1 # Exit script after printing help
}

# Default values.
SOURCE_DIRECTORY="."

FRONTIER_QUEUE=""
FRONTIER_NODES=1

while getopts "u:q:n:s:d:j:" opt; do
    case "$opt" in
    u) FRONTIER_USER="$OPTARG" ;;
    q) FRONTIER_QUEUE="$OPTARG" ;;
    n) FRONTIER_NODES="$OPTARG" ;;
    s) SOURCE_DIRECTORY="$OPTARG" ;;
    d) COPY_DIRECTORY="$OPTARG" ;;
    j) JOB_PATH="$OPTARG" ;;
    ?) helpFunction ;; # Print a help message for an unknown parameter.
    esac
done

# Print a help message if parameters are empty.
if [ -z "$FRONTIER_USER" ] || [ -z "$FRONTIER_NODES" ]; then
    echo "Some or all required parameters are empty:"
    echo -e "\tFRONTIER_USER: $FRONTIER_USER"
    echo -e "\tFRONTIER_NODES: $FRONTIER_NODES"
    helpFunction
fi

if ! test "$FRONTIER_NODES" -gt 0; then
    echo "The number of Frontier nodes ($FRONTIER_NODES) must be positive"
    helpFunction
fi

# Select default queue if unspecified.
if [ -z "$FRONTIER_QUEUE" ]; then
    FRONTIER_QUEUE="batch"
fi

if ! (echo "${ALLOWED_FRONTIER_QUEUES[@]}" | grep -q -w "${FRONTIER_QUEUE}"); then
    echo "Frontier queue ${FRONTIER_QUEUE} is not allowed. Valid values: ${ALLOWED_FRONTIER_QUEUES[@]}"
    helpFunction
fi

if [ -z "$COPY_DIRECTORY" ] || [ -z "$JOB_PATH" ] || [ -z "$SOURCE_DIRECTORY" ]; then
    echo "Some or all required parameters are empty:"
    echo -e "\tCOPY_DIRECTORY: $COPY_DIRECTORY"
    echo -e "\tJOB_PATH: $JOB_PATH"
    echo -e "\tSOURCE_DIRECTORY: $SOURCE_DIRECTORY"
    helpFunction
fi

# Start an SSH tunnel in the background so we only have to auth once.
# This tunnel will close automatically after 5 minutes of inactivity.
ssh -f -N -M -S ~/.ssh/control-%h-%p-%r -o "ControlPersist 15m" ${FRONTIER_USER}@frontier.olcf.ornl.gov

# Copy files to Frontier over the same SSH tunnel, excluding unnecessary ones.
echo "Copying files to Frontier... -----------------------------------------"
rsync -e "ssh -S ~/.ssh/control-%h-%p-%r" -avz --delete \
    --exclude-from "${SOURCE_DIRECTORY}/.gitignore" \
    --exclude tests \
    "${SOURCE_DIRECTORY}" "${FRONTIER_USER}@frontier.olcf.ornl.gov:${COPY_DIRECTORY}"


# Submit a job on Frontier over the same SSH tunnel.
echo "Setting up environment and submitting job on Frontier..."
# Save the variables to pass to the remote script.
printf -v varsStr '%q ' "$COPY_DIRECTORY" "$JOB_PATH" "$FRONTIER_NODES" "$FRONTIER_QUEUE"
# We need to properly escape the remote script due to the qsub command substitution.
ssh -S ~/.ssh/control-%h-%p-%r "${FRONTIER_USER}@frontier.olcf.ornl.gov" "bash -s $varsStr" <<'EOF'
  COPY_DIRECTORY=$1; JOB_PATH=$2; FRONTIER_NODES=$3; FRONTIER_QUEUE=$4
  cd "${COPY_DIRECTORY}"

  # Set up Conda env if it doesn't exist and activate it.
  module load PrgEnv-gnu/8.6.0
  module load miniforge3/23.11.0-0
  module load rocm/6.2.4
  module load craype-accel-amd-gfx90a

  if [ ! -d /lustre/orion/lrn081/scratch/$USER/miniconda3/envs/oumi ]; then
      echo "Creating Oumi Conda environment... -----------------------------------------"
      conda create -y python=3.10 -c conda-forge --prefix /lustre/orion/lrn081/scratch/$USER/miniconda3/envs/oumi
  fi
  echo "Installing packages... -----------------------------------------"
  if [ ! -z "$CONDA_DEFAULT_ENV" ]; then
    # Deactivate the previous environment (stacked env-s cause `pip install` problems).
    conda deactivate
  fi
  source activate "/lustre/orion/lrn081/scratch/$USER/miniconda3/envs/oumi"

  if ! command -v uv >/dev/null 2>&1; then
      pip install -U uv
  fi

  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
  pip install -e '.[gpu]' 'huggingface_hub[cli]' hf_transfer
  pip uninstall nvidia-smi

  # Optional: print GPU count and GPU name.
  python -c "import torch; print('GPU count: ', torch.cuda.device_count()); print(torch.cuda.get_device_name(0))"

  echo "Submitting job... -----------------------------------------"
  # Create a logs directory for the user if it doesn't exist.
  # This directory must exist for the run to work, as Frontier won't create them.
  mkdir -p /lustre/orion/lrn081/scratch/$USER/jobs/logs/

  set -x

  SBATCH_OUTPUT=$(sbatch --export=NONE -A lrn081 -N ${FRONTIER_NODES} -n ${FRONTIER_NODES} --threads-per-core=1 -m "block:cyclic" -p ${FRONTIER_QUEUE} -o "/lustre/orion/lrn081/scratch/$USER/jobs/logs/%j.OU" -e "/lustre/orion/lrn081/scratch/$USER/jobs/logs/%j.ER" ${JOB_PATH})
  SBATCH_RESULT=$?
  set +x  # Turn-off printing

  if (test "$SBATCH_RESULT" -ne 0) || [ -z "$SBATCH_OUTPUT" ]
  then
      echo "Job submission ('sbatch') failed with error code: $SBATCH_RESULT"
      exit 1
  fi
  JOB_ID=$(echo "${SBATCH_OUTPUT}" | grep -o -E '[0-9]+')
  if [ -z "$JOB_ID" ]
  then
      echo "Failed to extract JOB_ID from: $SBATCH_OUTPUT"
      exit 1
  fi

  echo "Job id: ${JOB_ID}"

  echo
  echo "All jobs:"
  squeue -l -u $USER

  echo
  echo "To view error logs, run (on Frontier):"
  echo "tail -n200 -f /lustre/orion/lrn081/scratch/$USER/jobs/logs/${JOB_ID}.ER"
  echo "To view output logs, run (on Frontier):"
  echo "tail -n200 -f /lustre/orion/lrn081/scratch/$USER/jobs/logs/${JOB_ID}.OU"
EOF
