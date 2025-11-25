# Deploying Oumi on Kubernetes

This guide covers deploying Oumi on Kubernetes (k8s) clusters. For automated cluster provisioning and job management using the Oumi launcher, see the {doc}`launch` guide instead.

Please follow this guide to deploy Oumi onto an existing k8s cluster with GPU nodes. For examples per cloud providers on setting up a k8s cluster, you may follow [platform specific examples](#platform-examples).

## Prerequisites

- A running k8s cluster with GPU nodes
- `kubectl` configured to access your cluster
- For GPU workloads: [NVIDIA Device Plugin](https://github.com/NVIDIA/k8s-device-plugin) installed
- Cluster must have internet access to pull Oumi container images from [ghcr.io/oumi-ai/oumi](https://github.com/oumi-ai/oumi/pkgs/container/oumi)

```{note}
Most cloud k8s clusters (EKS, GKE, AKS) use amd64/x86_64 architecture. Verify your node architecture with `kubectl get nodes -o wide` and select the appropriate image from the [container registry](https://github.com/oumi-ai/oumi/pkgs/container/oumi) to use below.
```

## Quick Start

### 1. Create Namespace

```bash
kubectl create namespace oumi
```

### 2. Deploy Oumi

Create `oumi-deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: oumi
  namespace: oumi
spec:
  replicas: 1
  selector:
    matchLabels:
      app: oumi
  template:
    metadata:
      labels:
        app: oumi
    spec:
      containers:
      - name: oumi
        # Use the linux/amd64 image for most cloud providers
        # Get latest images from: https://github.com/oumi-ai/oumi/pkgs/container/oumi
        image: ghcr.io/oumi-ai/oumi:latest
        command: ["sleep", "infinity"]
        # Adjust gpu, memory, and storage based on the model you want to run.
        # Below is configuration for single GPU per pod.
        resources:
          requests:
            nvidia.com/gpu: 1
            memory: "16Gi"
            ephemeral-storage: "100Gi"
          limits:
            nvidia.com/gpu: 1
            memory: "32Gi"
            ephemeral-storage: "1Ti"
      # Configure for GPU nodes
      tolerations:
      - key: nvidia.com/gpu
        operator: Exists
        effect: NoSchedule
      nodeSelector:
        nvidia.com/gpu: "true"  # Adjust based on your node labels
```

Apply the deployment:

```bash
kubectl apply -f oumi-deployment.yaml
```

Deployment can take about 15 minutes for the cluster to pull and load the Oumi container image.

### 3. Access Oumi

```bash
# Get pod name
POD_NAME=$(kubectl get pods -n oumi -l app=oumi -o jsonpath='{.items[0].metadata.name}')

# Execute commands in the pod
kubectl exec -it $POD_NAME -n oumi -- /bin/bash
```

Inside the pod, run Oumi commands:

```bash
oumi train -c /path/to/config.yaml
```

## Platform Examples

::::{tab-set}
:::{tab-item} AWS EKS
````{dropdown} EKS Setup Example

### Prerequisites
- AWS CLI and `eksctl` installed
- AWS account configured
- Sufficient quota for a GPU node

### Create Cluster

Create `gpu-cluster.yaml`:

```yaml
apiVersion: eksctl.io/v1alpha5
kind: ClusterConfig

metadata:
  name: gpu-cluster
  region: us-west-2
  version: "1.31"

vpc:
  cidr: "10.0.0.0/16"
  nat:
    gateway: Single

iam:
  withOIDC: true

nodeGroups:
  - name: cpu-workers
    instanceType: m5.large
    desiredCapacity: 2
    minSize: 1
    maxSize: 4
    volumeSize: 20
    ssh:
      allow: true
      publicKeyName: your-key-name  # Replace with your SSH key
    iam:
      attachPolicyARNs:
        - arn:aws:iam::aws:policy/AmazonEKSWorkerNodePolicy
        - arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy

  - name: gpu-workers
    instancesDistribution:
      instanceTypes: ["g4dn.xlarge", "g4dn.2xlarge"] # Adjust accordingly
      maxPrice: 0.50
      onDemandBaseCapacity: 0
      onDemandPercentageAboveBaseCapacity: 0
      spotInstancePools: 4
    desiredCapacity: 1
    minSize: 0
    maxSize: 3
    volumeSize: 50
    ssh:
      allow: true
      publicKeyName: your-key-name  # Replace with your SSH key
    labels:
      nvidia.com/gpu: "true"
    taints:
      - key: nvidia.com/gpu
        value: "true"
        effect: NoSchedule
    iam:
      attachPolicyARNs:
        - arn:aws:iam::aws:policy/AmazonEKSWorkerNodePolicy
        - arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy

addons:
  - name: vpc-cni
  - name: coredns
  - name: kube-proxy
  - name: aws-ebs-csi-driver
```

Create cluster and deploy:

```bash
# Create cluster (takes 15-20 minutes)
eksctl create cluster -f gpu-cluster.yaml

# Create namespace
kubectl create namespace oumi

# Apply Oumi deployment
kubectl apply -f oumi-deployment.yaml

# Access pod
POD_NAME=$(kubectl get pods -n oumi -l app=oumi -o jsonpath='{.items[0].metadata.name}')
kubectl exec -it $POD_NAME -n oumi -- /bin/bash
```

### Cleanup

```bash
eksctl delete cluster -f gpu-cluster.yaml
```
````
:::

:::{tab-item} GCP GKE
````{dropdown} GKE Setup Example

### Prerequisites
- `gcloud` CLI installed and configured

### Create Cluster

```bash
export PROJECT_ID=your-project-id
export ZONE=us-central1-a
export CLUSTER_NAME=oumi-cluster

gcloud config set project $PROJECT_ID

# Create cluster with GPU nodes
gcloud container clusters create $CLUSTER_NAME \
  --zone=$ZONE \
  --machine-type=n1-standard-4 \
  --num-nodes=2 \
  --accelerator type=nvidia-tesla-t4,count=1 \
  --enable-autoscaling \
  --min-nodes=1 \
  --max-nodes=3 \
  --disk-size=50

# Get credentials
gcloud container clusters get-credentials $CLUSTER_NAME --zone=$ZONE

# Install NVIDIA drivers
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded.yaml
```

### Deploy Oumi

Create `oumi-deployment.yaml` (adjust nodeSelector):

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: oumi
  namespace: oumi
spec:
  replicas: 1
  selector:
    matchLabels:
      app: oumi
  template:
    metadata:
      labels:
        app: oumi
    spec:
      containers:
      - name: oumi
        image: ghcr.io/oumi-ai/oumi:latest
        command: ["sleep", "infinity"]
        resources:
          requests:
            nvidia.com/gpu: 1
            memory: "16Gi"
          limits:
            nvidia.com/gpu: 1
            memory: "32Gi"
      nodeSelector:
        cloud.google.com/gke-accelerator: nvidia-tesla-t4
```

```bash
kubectl create namespace oumi
kubectl apply -f oumi-deployment.yaml
```

### Cleanup

```bash
gcloud container clusters delete $CLUSTER_NAME --zone=$ZONE
```
````
:::
