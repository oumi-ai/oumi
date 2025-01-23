#!/bin/bash
set -xe

export E2E_CLUSTER_PREFIX="oumi-e2e-tests-cluster"

oumi launch up --config "tests/scripts/gcp_e2e_tests_job.yaml" --resources.accelerators="A100:1" --cluster "${E2E_CLUSTER_PREFIX}-a100-1gpu40gb"
oumi launch up --config "tests/scripts/gcp_e2e_tests_job.yaml" --resources.accelerators="A100:4" --cluster "${E2E_CLUSTER_PREFIX}-a100-1gpu40gb"
oumi launch up --config "tests/scripts/gcp_e2e_tests_job.yaml" --resources.accelerators="A100-80GB:4" --cluster "${E2E_CLUSTER_PREFIX}-a100-1gpu80gb"
