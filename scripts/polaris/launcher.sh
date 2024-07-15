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

# Copy files to Polaris over the same SSH tunnel.
rsync -e "ssh -S ~/.ssh/control-%h-%p-%r" -avz --delete ${SOURCE_DIRECTORY} ${POLARIS_USER}@polaris.alcf.anl.gov:${COPY_DIRECTORY}

# Submit a job on Polaris over the same SSH tunnel.
ssh -S ~/.ssh/control-%h-%p-%r ${POLARIS_USER}@polaris.alcf.anl.gov << EOF
  cd ${COPY_DIRECTORY}
  module use /soft/modulefiles
  module load conda
  conda activate base
  mkdir -p ./worker_venv/example_environment
  python3 -m venv ./worker_venv/example_environment --system-site-packages
  source ./worker_venv/example_environment/bin/activate
  python3 -m pip install -e '.[train]'
  qsub ${JOB_PATH}
EOF
