# Changelog

## v0.1-alpha

This is the initial alpha release of Oumi, an open-source framework for training and evaluating large language models. This release introduces the core functionality and features of the Oumi framework and sets the foundation for many more features to come.

### 🚀 Core Features

- Flexible training loop supporting various model architectures (Llama, Mistral, Phi-3.5, Llava 1.5 among others
- Distributed training capabilities (DDP, FSDP)
- Comprehensive configuration system using OmegaConf
- Efficient data loading and preprocessing pipelines with streaming and packing support
- Integration of Parameter-Efficient Fine-Tuning (PEFT) techniques
- Evaluation framework with LM Evaluation Harness integration
- Multiple inference engines including vLLM and llama.cpp
- Launcher system for deploying jobs across cloud platforms and HPC systems
- Telemetry and profiling tools for performance monitoring
- Mixed-precision training and various optimization techniques

### 📊 Data and Model Management

- Support for various dataset formats and preprocessing techniques
- Integration with Hugging Face's Datasets and Transformers libraries
- Efficient data loading with support for streaming and packing

### 🖥️ Training and Optimization

- Distributed training support (DDP, FSDP)
- Mixed-precision training capabilities
- Integration of Parameter-Efficient Fine-Tuning (PEFT) techniques
- Model FLOPs Utilization (MFU) calculation for performance analysis

### 🔍 Evaluation and Inference

- Integration with LM Evaluation Harness for comprehensive model evaluation
- Support for multiple inference engines (vLLM, llama.cpp)
- Custom evaluation metrics and benchmarks

### 🚀 Deployment and Scaling

- Launcher system for running jobs on various cloud platforms (AWS, GCP, Azure)
- Support for HPC systems like Polaris
- Docker images for reproducible environments

### 📊 Monitoring and Visualization

- Integration with Weights & Biases (WandB) for experiment tracking
- TensorBoard support for metric visualization
- Custom telemetry and profiling tools

### 🛠️ Developer Tools

- API documentation using Sphinx
- Tutorials and notebooks for common use cases
- Pre-commit hooks for code quality and formatting
- GitHub Actions for continuous integration
- Makefile for common development tasks

### 📚 Documentation

- API documentation
- Tutorials and example notebooks
