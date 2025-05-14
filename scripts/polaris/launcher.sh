#!/bin/bash

ALLOWED_POLARIS_QUEUES=("debug" "debug-scaling" "preemptable" "prod")

helpFunction() {
    echo ""
    echo "Usage: $0 -u username -q debug -n 1 -s . -d /home/username/copylocation/ -j ./local/path/to/your_job.sh"
    echo -e "\t-u The username on Polaris."
    echo -e "\t-q The Polaris queue to use (${ALLOWED_POLARIS_QUEUES[@]})."
    echo -e "\t-n The number of Polaris nodes to use."
    echo -e "\t-s The source directory to copy. Defaults to the current directory."
    echo -e "\t-d The destination directory on Polaris to copy local files."
    echo -e "\t-j The local path to your job."
    exit 1 # Exit script after printing help
}

# Default values.
SOURCE_DIRECTORY="."

POLARIS_QUEUE=""
POLARIS_NODES=1

while getopts "u:q:n:s:d:j:" opt; do
    case "$opt" in
    u) POLARIS_USER="$OPTARG" ;;
    q) POLARIS_QUEUE="$OPTARG" ;;
    n) POLARIS_NODES="$OPTARG" ;;
    s) SOURCE_DIRECTORY="$OPTARG" ;;
    d) COPY_DIRECTORY="$OPTARG" ;;
    j) JOB_PATH="$OPTARG" ;;
    ?) helpFunction ;; # Print a help message for an unknown parameter.
    esac
done

# Print a help message if parameters are empty.
if [ -z "$POLARIS_USER" ] || [ -z "$POLARIS_NODES" ]; then
    echo "Some or all required parameters are empty:"
    echo -e "\tPOLARIS_USER: $POLARIS_USER"
    echo -e "\tPOLARIS_NODES: $POLARIS_NODES"
    helpFunction
fi

if ! test "$POLARIS_NODES" -gt 0; then
    echo "The number of Polaris nodes ($POLARIS_NODES) must be positive"
    helpFunction
fi

# Select default queue if unspecified (depends on the number of nodes).
if [ -z "$POLARIS_QUEUE" ]; then
    if test "$POLARIS_NODES" -gt 9; then
        POLARIS_QUEUE="prod"
    else
        POLARIS_QUEUE="preemptable"
    fi
fi

if ! (echo "${ALLOWED_POLARIS_QUEUES[@]}" | grep -q -w "${POLARIS_QUEUE}"); then
    echo "Polaris queue ${POLARIS_QUEUE} is not allowed. Valid values: ${ALLOWED_POLARIS_QUEUES[@]}"
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
ssh -f -N -M -S ~/.ssh/control-%h-%p-%r -o "ControlPersist 5m" ${POLARIS_USER}@polaris.alcf.anl.gov

# Copy files to Polaris over the same SSH tunnel, excluding unnecessary ones.
echo "Copying files to Polaris... -----------------------------------------"
rsync -e "ssh -S ~/.ssh/control-%h-%p-%r" -avz --delete \
    --exclude-from "${SOURCE_DIRECTORY}/.gitignore" \
    --exclude tests \
    "${SOURCE_DIRECTORY}" "${POLARIS_USER}@polaris.alcf.anl.gov:${COPY_DIRECTORY}"

# Submit a job on Polaris over the same SSH tunnel.
echo "Setting up environment and submitting job on Polaris..."
# Save the variables to pass to the remote script.
printf -v varsStr '%q ' "$COPY_DIRECTORY" "$JOB_PATH" "$POLARIS_NODES" "$POLARIS_QUEUE"
# We need to properly escape the remote script due to the qsub command substitution.
ssh -S ~/.ssh/control-%h-%p-%r "${POLARIS_USER}@polaris.alcf.anl.gov" "bash -s $varsStr" <<'EOF'
  COPY_DIRECTORY=$1; JOB_PATH=$2; POLARIS_NODES=$3; POLARIS_QUEUE=$4
  cd ${COPY_DIRECTORY}

  # Set up Conda env if it doesn't exist and activate it.
  module use /soft/modulefiles
  module load conda
  if [ ! -d /home/$USER/miniconda3/envs/oumi ]; then
      echo "Creating Oumi Conda environment... -----------------------------------------"
      conda create -y python=3.11 --prefix /home/$USER/miniconda3/envs/oumi
  fi
  echo "Installing packages... -----------------------------------------"
  conda activate /home/$USER/miniconda3/envs/oumi

  if ! command -v uv >/dev/null 2>&1; then
      pip install -U uv
  fi
  pip install -e '.[gpu]'

  echo "Submitting job... -----------------------------------------"
  # Create a logs directory for the user if it doesn't exist.
  # This directory must exist for the run to work, as Polaris won't create them.
  mkdir -p /eagle/community_ai/jobs/logs/$USER/

  set -x
  JOB_ID=$(qsub -l select=${POLARIS_NODES}:system=polaris -q ${POLARIS_QUEUE} -o /eagle/community_ai/jobs/logs/$USER/ -e /eagle/community_ai/jobs/logs/$USER/ ${JOB_PATH})
  QSUB_RESULT=$?
  set +x  # Turn-off printing

  if (test "$QSUB_RESULT" -ne 0) || [ -z "$JOB_ID" ]
  then
      echo "Job submission ('qsub') failed with error code: $QSUB_RESULT"
      exit 1
  fi
  echo "Job id: ${JOB_ID}"

  echo
  echo "All jobs:"
  qstat -s -u $USER
  echo
  echo "To view error logs, run (on Polaris):"
  echo "tail -n200 -f /eagle/community_ai/jobs/logs/$USER/${JOB_ID}.ER"
  echo "To view output logs, run (on Polaris):"
  echo "tail -n200 -f /eagle/community_ai/jobs/logs/$USER/${JOB_ID}.OU"
EOF
