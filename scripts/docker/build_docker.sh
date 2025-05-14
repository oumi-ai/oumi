#!/bin/bash
# Script to build the docker image and push it to the docker hub.

set -xe

OUMI_TAG=oumi:latest

echo "Building image..."
docker build -t $OUMI_TAG .

echo "Running basic tests..."
docker run --rm $OUMI_TAG oumi env

# echo "Pushing docker image"
# docker push $DOCKER_HUB/$OUMI_TAG

echo "Build complete!"
