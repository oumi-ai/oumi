#!/bin/bash
# Script to build the docker image and push it to the docker hub.

set -xe

DOCKER_HUB="gcr.io/lema/lema"
VERSION=latest

echo "Building docker image $DOCKER_HUB:$VERSION"
docker build -t $DOCKER_HUB/lema:$VERSION .

echo "Pushing docker image $DOCKER_HUB:$VERSION"
docker push $DOCKER_HUB/lema:$VERSION
