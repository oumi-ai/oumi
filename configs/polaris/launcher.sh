#!/bin/bash

# Replace with your username.
POLARIS_USER=matthew
# Replace with your output directory.
OUTPUT_DIR=foo

# Start an SSH tunnel in the background so we only have to auth once.
# This tunnel will close automatically after 10 seconds of inactivity.
ssh -f -N -M -S ~/.ssh/control-%h-%p-%r -o "ControlPersist 10s" ${POLARIS_USER}@polaris.alcf.anl.gov

# Copy files to Polaris over the same SSH tunnel.
rsync -e "ssh -S ~/.ssh/control-%h-%p-%r" -avz --delete . ${POLARIS_USER}@polaris.alcf.anl.gov:/home/${POLARIS_USER}/${OUTPUT_DIR}/

# Submit a job on Polaris over the same SSH tunnel.
ssh -S ~/.ssh/control-%h-%p-%r ${POLARIS_USER}@polaris.alcf.anl.gov << EOF
  cd /home/${POLARIS_USER}/${OUTPUT_DIR}
  qsub ./configs/polaris/jobs/example_job.sh
EOF
