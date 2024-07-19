#!/bin/bash

helpFunction()
{
   echo ""
   echo "Usage: $0 -u username -s . -d /home/username/copylocation/ -j ./local/path/to/your_job.sh"
   echo -e "\t-u The username on Polaris."
   echo -e "\t-s The source directory to copy. Defaults to the current directory."
   echo -e "\t-d The destination directory on Polaris to copy local files."
   echo -e "\t-j The local path to your job."
   exit 1 # Exit script after printing help
}

# Default values.
SOURCE_DIRECTORY="."

while getopts "u:s:d:j:" opt
do
   case "$opt" in
      u ) POLARIS_USER="$OPTARG" ;;
      s ) SOURCE_DIRECTORY="$OPTARG" ;;
      d ) COPY_DIRECTORY="$OPTARG" ;;
      j ) JOB_PATH="$OPTARG" ;;
      ? ) helpFunction ;; # Print a help message for an unknown parameter.
   esac
done

# Print a help message if parameters are empty.
if [ -z "$POLARIS_USER" ] || [ -z "$COPY_DIRECTORY" ] || [ -z "$JOB_PATH" ] || [ -z "$SOURCE_DIRECTORY" ]
then
   echo "Some or all required parameters are empty";
   helpFunction
fi

# Start an SSH tunnel in the background so we only have to auth once.
# This tunnel will close automatically after 5 minutes of inactivity.
ssh -f -N -M -S ~/.ssh/control-%h-%p-%r -o "ControlPersist 5m" ${POLARIS_USER}@polaris.alcf.anl.gov

# Copy files to Polaris over the same SSH tunnel, excluding unnecessary ones.
echo "Copying files to Polaris... -----------------------------------------"
rsync -e "ssh -S ~/.ssh/control-%h-%p-%r" -avz --delete \
--exclude-from ${SOURCE_DIRECTORY}/.gitignore \
--exclude tests \
${SOURCE_DIRECTORY} ${POLARIS_USER}@polaris.alcf.anl.gov:${COPY_DIRECTORY}

# Submit a job on Polaris over the same SSH tunnel.
echo "Setting up environment and submitting job on Polaris..."
# Save the variables to pass to the remote script.
printf -v varsStr '%q ' "$COPY_DIRECTORY" "$JOB_PATH"
# We need to properly escape the remote script due to the qsub command substitution.
ssh -S ~/.ssh/control-%h-%p-%r ${POLARIS_USER}@polaris.alcf.anl.gov "bash -s $varsStr" << 'EOF'
  COPY_DIRECTORY=$1; JOB_PATH=$2
  cd ${COPY_DIRECTORY}

  # Set up Conda env if it doesn't exist and activate it.
  module use /soft/modulefiles
  module load conda
  if [ ! -d /home/$USER/miniconda3/envs/lema ]; then
      echo "Creating LeMa Conda environment... -----------------------------------------"
      conda create -y python=3.11 --prefix /home/$USER/miniconda3/envs/lema
      # Install flash-attn manually since it's not in our pyproject.toml.
      conda activate /home/$USER/miniconda3/envs/lema
      pip install flash-attn --no-build-isolation
  fi
  conda activate /home/$USER/miniconda3/envs/lema

  echo "Installing packages... -----------------------------------------"
  pip install -e '.[train]'
  echo "Submitting job... -----------------------------------------"
  # Create a logs directory for the user if it doesn't exist.
  # This directory must exist for the run to work, as Polaris won't create them.
  mkdir -p /eagle/community_ai/jobs/logs/$USER/
  JOB_ID=$(qsub -o /eagle/community_ai/jobs/logs/$USER/ -e /eagle/community_ai/jobs/logs/$USER/ ${JOB_PATH})
  echo "Job id: ${JOB_ID}"

  echo
  echo "All jobs:"
  qstat -s -u $USER
  echo
  echo "To view error logs, run (on Polaris):"
  echo "cat /eagle/community_ai/jobs/logs/$USER/${JOB_ID}.ER"
  echo "To view output logs, run (on Polaris):"
  echo "cat /eagle/community_ai/jobs/logs/$USER/${JOB_ID}.OU"
EOF
