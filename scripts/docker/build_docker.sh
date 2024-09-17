#!/bin/bash
# Script to build the docker image and push it to the docker hub.

set -xe

DOCKER_HUB="TBD" # OPE-251
VERSION=latest

echo "Building docker image $DOCKER_HUB:$VERSION"
docker build -t $DOCKER_HUB/oumi:$VERSION .

echo "Pushing docker image $DOCKER_HUB:$VERSION"
docker push $DOCKER_HUB/oumi:$VERSION
