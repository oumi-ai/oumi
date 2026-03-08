#!/bin/bash

helpFunction() {
    echo ""
    echo "Usage: $0 -u username -q regular -n 1 -s . -d /home/username/copylocation/ -j ./local/path/to/your_job.sh"
    echo -e "\t-u The username on NERSC Perlmutter cluster."
    echo -e "\t-q The Perlmutter QoS to use. https://docs.nersc.gov/jobs/policy/#selecting-a-qos."
    echo -e "\t-n The number of Perlmutter nodes to use. Defaults to 1."
    echo -e "\t-s The source directory to copy. Defaults to the current directory."
    echo -e "\t-d The destination directory on Perlmutter to copy local files."
    echo -e "\t-j The local path to your job."
    exit 1 # Exit script after printing help
}

# Default values.
SOURCE_DIRECTORY="."

PERLMUTTER_QOS=""
PERLMUTTER_NODES=1

while getopts "u:q:n:s:d:j:" opt; do
    case "$opt" in
    u) PERLMUTTER_USER="$OPTARG" ;;
    q) PERLMUTTER_QOS="$OPTARG" ;;
    n) PERLMUTTER_NODES="$OPTARG" ;;
    s) SOURCE_DIRECTORY="$OPTARG" ;;
    d) COPY_DIRECTORY="$OPTARG" ;;
    j) JOB_PATH="$OPTARG" ;;
    ?) helpFunction ;; # Print a help message for an unknown parameter.
    esac
done

# Print a help message if parameters are empty.
if [ -z "$PERLMUTTER_USER" ] || [ -z "$PERLMUTTER_NODES" ]; then
    echo "Some or all required parameters are empty:"
    echo -e "\tPERLMUTTER_USER: $PERLMUTTER_USER"
    echo -e "\tPERLMUTTER_NODES: $PERLMUTTER_NODES"
    helpFunction
fi

if ! test "$PERLMUTTER_NODES" -gt 0; then
    echo "The number of Perlmutter nodes ($PERLMUTTER_NODES) must be positive"
    helpFunction
fi

# Select default queue if unspecified.
if [ -z "$PERLMUTTER_QOS" ]; then
    PERLMUTTER_QOS="debug"
fi

if [ -z "$COPY_DIRECTORY" ] || [ -z "$JOB_PATH" ] || [ -z "$SOURCE_DIRECTORY" ]; then
    echo "Some or all required parameters are empty:"
    echo -e "\tCOPY_DIRECTORY: $COPY_DIRECTORY"
    echo -e "\tJOB_PATH: $JOB_PATH"
    echo -e "\tSOURCE_DIRECTORY: $SOURCE_DIRECTORY"
    helpFunction
fi

SSH_TARGET="${PERLMUTTER_USER}@perlmutter.nersc.gov"

# Start an SSH tunnel in the background so we only have to auth once.
# This tunnel will close automatically after 5 minutes of inactivity.
ssh -f -N -M -S ~/.ssh/control-%h-%p-%r -o "ControlPersist 15m" ${SSH_TARGET}

# Copy files to Perlmutter over the same SSH tunnel, excluding unnecessary ones.
echo "Copying files to Perlmutter... -----------------------------------------"
rsync -e "ssh -S ~/.ssh/control-%h-%p-%r" -avz --delete \
    --exclude-from "${SOURCE_DIRECTORY}/.gitignore" \
    --exclude tests \
    "${SOURCE_DIRECTORY}" "${SSH_TARGET}:${COPY_DIRECTORY}"

# Submit a job on Perlmutter over the same SSH tunnel.
echo "Setting up environment and submitting job on Perlmutter..."
# Save the variables to pass to the remote script.
printf -v varsStr '%q ' "$COPY_DIRECTORY" "$JOB_PATH" "$PERLMUTTER_NODES" "$PERLMUTTER_QOS"
# We need to properly escape the remote script due to the sbatch command substitution.
ssh -S ~/.ssh/control-%h-%p-%r "${SSH_TARGET}" "bash -s $varsStr" <<'EOF'
  COPY_DIRECTORY=$1; JOB_PATH=$2; PERLMUTTER_NODES=$3; PERLMUTTER_QOS=$4
  cd "${COPY_DIRECTORY}"

  # Activate the Conda environment.
  module load conda
  echo "Installing packages... -----------------------------------------"
  if [ ! -z "$CONDA_DEFAULT_ENV" ]; then
    # Deactivate the previous environment (stacked env-s cause `pip install` problems).
    conda deactivate
  fi
  conda activate oumi

  if ! command -v uv >/dev/null 2>&1; then
      pip install -U uv
  fi

  uv pip install -U -e '.[gpu]' 'huggingface_hub[cli]' hf_transfer

  echo "Submitting job... -----------------------------------------"
  # Create a logs directory for the user if it doesn't exist.
  # This directory must exist for the run to work, as Perlmutter won't create them.
  mkdir -p $CFS/$SBATCH_ACCOUNT/users/$USER/jobs/logs/

  set -x

  # Additional options are set by the job script in #SBATCH comments.-
  SBATCH_OUTPUT=$(sbatch \
    -C gpu \
    --gpus-per-node 4 \
    -N ${PERLMUTTER_NODES} \
    -n ${PERLMUTTER_NODES} \
    -q ${PERLMUTTER_QOS} \
    -o "$CFS/$SBATCH_ACCOUNT/users/$USER/jobs/logs/%j.out" \
    ${JOB_PATH}
  )
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
  echo "To view output logs, run (on Perlmutter):"
  echo "tail -n200 -f $CFS/$SBATCH_ACCOUNT/users/$USER/jobs/logs/${JOB_ID}.out"
EOF
