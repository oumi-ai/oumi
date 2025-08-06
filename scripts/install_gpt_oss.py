#!/usr/bin/env python3
"""Install GPT OSS dependencies for Oumi.

This script installs all the special dependencies needed for OpenAI GPT OSS
model support, including the vLLM GPT OSS build, Flash Attention 3, and MXFP4
quantization.

Usage:
    python scripts/install_gpt_oss.py
    # or
    make install-gpt-oss
"""

import os
import subprocess
import sys
from pathlib import Path


def run_command(cmd: str, description: str = "") -> bool:
    """Run a shell command and return success status."""
    if description:
        print(f"   {description}")

    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=False)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed: {cmd}")
        print(f"   Error: {e}")
        return False


def check_environment():
    """Check if we're in a virtual environment."""
    conda_env = os.environ.get("CONDA_DEFAULT_ENV")
    virtual_env = os.environ.get("VIRTUAL_ENV")

    if not conda_env and not virtual_env:
        print("⚠️  WARNING: No conda or virtual environment detected.")
        print("   It's recommended to install in an isolated environment.")

        response = input("Continue anyway? (y/N): ").strip().lower()
        if response not in ["y", "yes"]:
            print("❌ Installation cancelled.")
            return False

    return True


def verify_installation():
    """Verify that all packages were installed correctly."""
    print("🔍 Step 4: Verifying installation...")

    verification_script = """
import sys
print('Python version:', sys.version)

packages = [
    ('vllm', 'vLLM'),
    ('transformers', 'Transformers'),
    ('mxfp4', 'MXFP4'),
    ('flash_attn_interface', 'Flash Attention 3')
]

failed = False
for module_name, display_name in packages:
    try:
        module = __import__(module_name)
        if module_name == 'flash_attn_interface':
            # Special handling for Flash Attention 3
            print(f'✓ {display_name} interface available')
            # Test that the main function is accessible
            module.flash_attn_func
            print(f'✓ {display_name} function accessible')
        else:
            version = getattr(module, '__version__', 'unknown')
            print(f'✓ {display_name} version: {version}')
    except ImportError as e:
        if module_name == 'flash_attn_interface':
            print(f'❌ {display_name} import failed: {e}')
            print('   Note: Flash Attention 3 requires H100/H800 GPU and CUDA >= 12.3')
        else:
            print(f'❌ {display_name} import failed: {e}')
        failed = True

if not failed:
    print('')
    print('✅ All GPT OSS dependencies are correctly installed!')
else:
    print('')
    print('❌ Some dependencies failed to install correctly!')
    sys.exit(1)
"""

    try:
        result = subprocess.run(
            [sys.executable, "-c", verification_script], capture_output=True, text=True
        )
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False


def main():
    """Main installation function."""
    print("🚀 Installing GPT OSS dependencies for Oumi")
    print("=" * 50)

    # Check environment
    if not check_environment():
        return 1

    # Step 1: Install standard GPT OSS dependencies
    print("\n📦 Step 1: Installing standard GPT OSS dependencies...")
    if not run_command('pip install "oumi[gpt_oss]"'):
        print("❌ Failed to install standard GPT OSS dependencies")
        return 1

    # Step 2: Install vLLM GPT OSS build
    print("\n🔧 Step 2: Installing vLLM GPT OSS build...")
    print("   This may take several minutes...")
    vllm_cmd = (
        "pip install --pre vllm==0.10.1+gptoss "
        "--extra-index-url https://wheels.vllm.ai/gpt-oss/ "
        "--extra-index-url https://download.pytorch.org/whl/nightly/cu128 "
        "--index-strategy unsafe-best-match"
    )
    if not run_command(vllm_cmd):
        print("❌ Failed to install vLLM GPT OSS build")
        return 1

    # Step 3: Install Flash Attention 3 from source
    print("\n⚡ Step 3: Installing Flash Attention 3 from source...")
    print("   This requires H100/H800 GPU and CUDA >= 12.3")

    # Check CUDA version
    try:
        result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            import re

            cuda_match = re.search(r"release (\d+\.\d+)", result.stdout)
            if cuda_match:
                print(f"   Detected CUDA version: {cuda_match.group(1)}")
        else:
            print("⚠️  WARNING: nvcc not found. Flash Attention 3 requires CUDA >= 12.3")
    except FileNotFoundError:
        print("⚠️  WARNING: nvcc not found. Flash Attention 3 requires CUDA >= 12.3")

    # Install compilation dependencies
    print("   Installing compilation dependencies...")
    if not run_command("pip install packaging ninja"):
        print("❌ Failed to install compilation dependencies")
        return 1

    # Check RAM and set MAX_JOBS
    try:
        import psutil

        ram_gb = psutil.virtual_memory().total // (1024**3)
        if ram_gb < 96:
            print(f"   Detected {ram_gb}GB RAM, limiting parallel jobs to 4")
            os.environ["MAX_JOBS"] = "4"
    except ImportError:
        print("   psutil not available, using default compilation settings")

    # Clone and install Flash Attention 3
    import tempfile

    print("   Cloning Flash Attention repository...")
    with tempfile.TemporaryDirectory() as temp_dir:
        repo_path = Path(temp_dir) / "flash-attention"
        if not run_command(
            f'git clone https://github.com/Dao-AILab/flash-attention.git "{repo_path}"'
        ):
            print("❌ Failed to clone Flash Attention repository")
            return 1

        hopper_path = repo_path / "hopper"
        os.chdir(hopper_path)

        print("   Compiling Flash Attention 3 (this may take 10-20 minutes)...")
        if not run_command("python setup.py install"):
            print("❌ Failed to compile Flash Attention 3")
            return 1

        print("   ✓ Flash Attention 3 installed from source")

    # Step 4: Verify installation
    if not verify_installation():
        print("❌ Installation verification failed!")
        print("   Please check the error messages above and retry.")
        return 1

    # Success message
    print("\n🎉 GPT OSS installation completed successfully!")
    print("\n📋 Next steps:")
    print("1. Test with: python scripts/test_gpt_oss.py")
    print(
        "2. Run inference: oumi infer -c "
        "configs/recipes/gpt_oss/inference/20b_vllm_infer.yaml --interactive"
    )
    print(
        "3. Train with LoRA: "
        "oumi train -c configs/recipes/gpt_oss/sft/20b_lora_train.yaml"
    )
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
