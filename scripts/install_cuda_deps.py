#!/usr/bin/env python3
"""Automated installation script for PyTorch, vLLM, and flash-attention.

Detects CUDA driver version and GPU compute capability, then installs
compatible versions of all dependencies.

Usage:
    python install_cuda_deps.py [--dry-run] [--skip-flash-attn] [--skip-vllm]
    python install_cuda_deps.py --mode pinned --preset stable
    python install_cuda_deps.py --mode pinned --preset latest
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any


@dataclass
class CUDAInfo:
    driver_version: str
    max_cuda_version: str
    gpus: list[GPUInfo]

    @property
    def min_compute_capability(self) -> tuple[int, int] | None:
        """Return the minimum compute capability across all GPUs."""
        caps = [g.compute_capability for g in self.gpus if g.compute_capability]
        return min(caps) if caps else None


@dataclass
class GPUInfo:
    index: int
    name: str
    compute_capability: tuple[int, int] | None


@dataclass
class PinnedVersions:
    """A tested combination of package versions."""

    name: str
    description: str
    cuda_version: str
    torch: str
    torchvision: str
    torchaudio: str
    vllm: str | None
    flash_attn: str | None
    python_min: str
    python_max: str


# Tested version combinations (pinned presets)
PINNED_PRESETS: dict[str, PinnedVersions] = {
    "latest": PinnedVersions(
        name="latest",
        description="Latest vLLM (Dec 2025) - requires CUDA 12.9+",
        cuda_version="cu129",
        torch="2.9.1",
        torchvision="0.24.1",
        torchaudio="2.9.1",
        vllm="0.13.0",
        flash_attn="2.7.3",
        python_min="3.10",
        python_max="3.13",
    ),
    "stable": PinnedVersions(
        name="stable",
        description="Stable combination (mid-2024) - CUDA 12.1, widely tested",
        cuda_version="cu121",
        torch="2.4.0",
        torchvision="0.19.0",
        torchaudio="2.4.0",
        vllm="0.5.5",
        flash_attn="2.6.3",
        python_min="3.9",
        python_max="3.12",
    ),
    "cuda118": PinnedVersions(
        name="cuda118",
        description="CUDA 11.8 compatibility - for older drivers",
        cuda_version="cu118",
        torch="2.3.1",
        torchvision="0.18.1",
        torchaudio="2.3.1",
        vllm="0.5.0",
        flash_attn="2.5.9",
        python_min="3.9",
        python_max="3.12",
    ),
    "torch-only": PinnedVersions(
        name="torch-only",
        description="PyTorch only - no vLLM or flash-attn",
        cuda_version="cu121",
        torch="2.5.1",
        torchvision="0.20.1",
        torchaudio="2.5.1",
        vllm=None,
        flash_attn=None,
        python_min="3.9",
        python_max="3.13",
    ),
}

# Mapping of CUDA versions to PyTorch wheel URLs (for auto mode)
CUDA_TO_PYTORCH = {
    "12.9": "cu129",
    "12.8": "cu128",
    "12.6": "cu126",
    "12.4": "cu124",
    "12.1": "cu121",
    "11.8": "cu118",
}

# Minimum compute capability for flash-attention (Ampere = 8.0)
MIN_FLASH_ATTN_COMPUTE = (8, 0)


def run_command(cmd: list[str], capture: bool = True) -> subprocess.CompletedProcess:
    """Run a command and return the result."""
    return subprocess.run(
        cmd,
        capture_output=capture,
        text=True,
        check=False,
    )


def get_cuda_info_nvml() -> CUDAInfo | None:
    """Detect CUDA info using nvidia-ml-py (pynvml). More robust than parsing nvidia-smi."""
    try:
        from pynvml import (
            nvmlDeviceGetCount,
            nvmlDeviceGetCudaComputeCapability,
            nvmlDeviceGetHandleByIndex,
            nvmlDeviceGetName,
            nvmlInit,
            nvmlShutdown,
            nvmlSystemGetCudaDriverVersion_v2,
            nvmlSystemGetDriverVersion,
        )
    except ImportError:
        return None

    try:
        nvmlInit()

        # Get driver version
        driver_version = nvmlSystemGetDriverVersion()

        # Get CUDA version (returns int like 12020 for 12.2)
        cuda_ver_int = nvmlSystemGetCudaDriverVersion_v2()
        cuda_major = cuda_ver_int // 1000
        cuda_minor = (cuda_ver_int % 1000) // 10
        max_cuda = f"{cuda_major}.{cuda_minor}"

        # Get all GPUs
        gpus = []
        device_count = nvmlDeviceGetCount()
        for i in range(device_count):
            handle = nvmlDeviceGetHandleByIndex(i)
            name = nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")
            major, minor = nvmlDeviceGetCudaComputeCapability(handle)
            gpus.append(
                GPUInfo(
                    index=i,
                    name=name,
                    compute_capability=(major, minor),
                )
            )

        nvmlShutdown()

        return CUDAInfo(
            driver_version=driver_version,
            max_cuda_version=max_cuda,
            gpus=gpus,
        )
    except Exception:
        return None


def get_cuda_info_nvidia_smi() -> CUDAInfo | None:
    """Fallback: Detect CUDA info by parsing nvidia-smi output."""
    if not shutil.which("nvidia-smi"):
        return None

    # Get basic info
    result = run_command(["nvidia-smi"])
    if result.returncode != 0:
        return None

    output = result.stdout

    # Parse driver version
    driver_match = re.search(r"Driver Version:\s*(\d+\.\d+(?:\.\d+)?)", output)
    driver_version = driver_match.group(1) if driver_match else "unknown"

    # Parse CUDA version
    cuda_match = re.search(r"CUDA Version:\s*(\d+\.\d+)", output)
    max_cuda = cuda_match.group(1) if cuda_match else "unknown"

    # Get GPU info using structured query
    gpu_result = run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,name,compute_cap",
            "--format=csv,noheader,nounits",
        ]
    )

    gpus = []
    if gpu_result.returncode == 0 and gpu_result.stdout.strip():
        for line in gpu_result.stdout.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                try:
                    idx = int(parts[0])
                    name = parts[1]
                    cap_parts = parts[2].split(".")
                    compute_cap = (
                        (int(cap_parts[0]), int(cap_parts[1]))
                        if len(cap_parts) == 2
                        else None
                    )
                    gpus.append(
                        GPUInfo(index=idx, name=name, compute_capability=compute_cap)
                    )
                except (ValueError, IndexError):
                    continue

    return CUDAInfo(
        driver_version=driver_version,
        max_cuda_version=max_cuda,
        gpus=gpus,
    )


def get_cuda_info() -> CUDAInfo | None:
    """Detect CUDA info using best available method."""
    # Try nvidia-ml-py first (more robust)
    info = get_cuda_info_nvml()
    if info:
        return info

    # Fall back to nvidia-smi parsing
    return get_cuda_info_nvidia_smi()


def select_pytorch_cuda_version(max_cuda: str) -> str | None:
    """Select the best PyTorch CUDA version for the system."""
    try:
        max_major, max_minor = map(int, max_cuda.split("."))
    except ValueError:
        return None

    # Find the highest compatible CUDA version
    for cuda_ver, cuda_tag in sorted(CUDA_TO_PYTORCH.items(), reverse=True):
        ver_major, ver_minor = map(int, cuda_ver.split("."))
        if (ver_major, ver_minor) <= (max_major, max_minor):
            return cuda_tag

    return None


def check_flash_attn_compatible(compute_cap: tuple[int, int] | None) -> bool:
    """Check if GPU supports flash-attention."""
    if compute_cap is None:
        return False
    return compute_cap >= MIN_FLASH_ATTN_COMPUTE


def check_python_version(preset: PinnedVersions) -> bool:
    """Check if current Python version is compatible with preset."""
    current = sys.version_info
    min_parts = [int(x) for x in preset.python_min.split(".")]
    max_parts = [int(x) for x in preset.python_max.split(".")]

    current_tuple = (current.major, current.minor)
    min_tuple = tuple(min_parts[:2])
    max_tuple = tuple(max_parts[:2])

    return min_tuple <= current_tuple <= max_tuple


def install_packages_pinned(preset: PinnedVersions, dry_run: bool = False) -> bool:
    """Install packages using pinned versions."""
    wheel_url = f"https://download.pytorch.org/whl/{preset.cuda_version}"

    # Build torch install command
    torch_pkgs = [
        f"torch=={preset.torch}",
        f"torchvision=={preset.torchvision}",
        f"torchaudio=={preset.torchaudio}",
    ]

    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        *torch_pkgs,
        "--index-url",
        wheel_url,
    ]

    print(f"\n[*] Installing PyTorch stack ({preset.cuda_version})")
    print(f"    Command: {' '.join(cmd)}")

    if not dry_run:
        result = run_command(cmd, capture=False)
        if result.returncode != 0:
            return False

    # Install vLLM if specified
    if preset.vllm:
        cmd = [sys.executable, "-m", "pip", "install", f"vllm=={preset.vllm}"]
        print(f"\n[*] Installing vLLM {preset.vllm}")
        print(f"    Command: {' '.join(cmd)}")

        if not dry_run:
            result = run_command(cmd, capture=False)
            if result.returncode != 0:
                return False

    # Install flash-attn if specified
    if preset.flash_attn:
        cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            f"flash-attn=={preset.flash_attn}",
            "--no-build-isolation",
        ]
        print(f"\n[*] Installing flash-attention {preset.flash_attn}")
        print(f"    Command: {' '.join(cmd)}")

        if not dry_run:
            result = run_command(cmd, capture=False)
            if result.returncode != 0:
                return False

    return True


def install_packages_auto(
    cuda_tag: str,
    skip_vllm: bool,
    skip_flash_attn: bool,
    dry_run: bool = False,
) -> bool:
    """Install packages using auto-detected versions."""
    wheel_url = f"https://download.pytorch.org/whl/{cuda_tag}"

    # Install PyTorch
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "torch",
        "torchvision",
        "torchaudio",
        "--index-url",
        wheel_url,
    ]

    print(f"\n[*] Installing PyTorch with CUDA {cuda_tag}")
    print(f"    Command: {' '.join(cmd)}")

    if not dry_run:
        result = run_command(cmd, capture=False)
        if result.returncode != 0:
            return False

    # Install vLLM
    if not skip_vllm:
        cmd = [sys.executable, "-m", "pip", "install", "vllm"]
        print("\n[*] Installing vLLM (latest)")
        print(f"    Command: {' '.join(cmd)}")

        if not dry_run:
            result = run_command(cmd, capture=False)
            if result.returncode != 0:
                return False

    # Install flash-attn
    if not skip_flash_attn:
        cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "flash-attn",
            "--no-build-isolation",
        ]
        print("\n[*] Installing flash-attention (latest)")
        print(f"    Command: {' '.join(cmd)}")

        if not dry_run:
            result = run_command(cmd, capture=False)
            if result.returncode != 0:
                return False

    return True


def verify_installation(skip_flash_attn: bool, skip_vllm: bool) -> dict[str, Any]:
    """Verify that all packages are installed and working."""
    results: dict[str, Any] = {}

    # Check PyTorch
    try:
        import torch

        results["torch"] = {
            "installed": True,
            "cuda_available": torch.cuda.is_available(),
            "version": torch.__version__,
            "cuda_version": torch.version.cuda,
        }
        print(f"    PyTorch {torch.__version__} (CUDA {torch.version.cuda})")
        print(f"    CUDA available: {torch.cuda.is_available()}")
    except ImportError:
        results["torch"] = {"installed": False}

    # Check vLLM
    if not skip_vllm:
        try:
            import vllm

            results["vllm"] = {"installed": True, "version": vllm.__version__}
            print(f"    vLLM {vllm.__version__}")
        except ImportError:
            results["vllm"] = {"installed": False}

    # Check flash-attn
    if not skip_flash_attn:
        try:
            import flash_attn
            from flash_attn import flash_attn_func  # noqa: F401

            results["flash_attn"] = {
                "installed": True,
                "version": flash_attn.__version__,
            }
            print(f"    flash-attn {flash_attn.__version__}")
        except ImportError:
            results["flash_attn"] = {"installed": False}

    return results


def print_cuda_info(cuda_info: CUDAInfo) -> None:
    """Print detected CUDA information."""
    print(f"    Driver version: {cuda_info.driver_version}")
    print(f"    Max CUDA version: {cuda_info.max_cuda_version}")
    print(f"    GPUs detected: {len(cuda_info.gpus)}")
    for gpu in cuda_info.gpus:
        cap_str = (
            f"{gpu.compute_capability[0]}.{gpu.compute_capability[1]}"
            if gpu.compute_capability
            else "unknown"
        )
        print(f"      [{gpu.index}] {gpu.name} (SM {cap_str})")


def list_presets() -> None:
    """List available pinned presets."""
    print("\nAvailable presets:")
    print("-" * 70)
    for name, preset in PINNED_PRESETS.items():
        print(f"\n  {name}:")
        print(f"    {preset.description}")
        print(f"    PyTorch: {preset.torch} | CUDA: {preset.cuda_version}")
        print(
            f"    vLLM: {preset.vllm or 'N/A'} | flash-attn: {preset.flash_attn or 'N/A'}"
        )
        print(f"    Python: {preset.python_min} - {preset.python_max}")


def main():
    parser = argparse.ArgumentParser(
        description="Install PyTorch, vLLM, and flash-attention with CUDA compatibility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect CUDA and install latest compatible versions
  python install_cuda_deps.py

  # Use a tested/pinned version combination
  python install_cuda_deps.py --mode pinned --preset stable

  # List available presets
  python install_cuda_deps.py --list-presets

  # Dry run to see what would be installed
  python install_cuda_deps.py --dry-run
        """,
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "pinned"],
        default="auto",
        help="Installation mode: 'auto' detects versions, 'pinned' uses tested combinations",
    )
    parser.add_argument(
        "--preset",
        choices=list(PINNED_PRESETS.keys()),
        default="stable",
        help="Preset to use in pinned mode (default: stable)",
    )
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="List available pinned presets and exit",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them",
    )
    parser.add_argument(
        "--skip-flash-attn",
        action="store_true",
        help="Skip flash-attention installation",
    )
    parser.add_argument(
        "--skip-vllm",
        action="store_true",
        help="Skip vLLM installation",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing installation",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("CUDA Dependencies Installer")
    print("=" * 60)

    # List presets mode
    if args.list_presets:
        list_presets()
        sys.exit(0)

    # Detect CUDA info
    print("\n[*] Detecting CUDA environment...")
    cuda_info = get_cuda_info()

    if cuda_info is None:
        print("[!] ERROR: Could not detect NVIDIA GPU or CUDA driver")
        print("    Make sure nvidia-smi is available and working")
        print(
            "    For better detection, install nvidia-ml-py: pip install nvidia-ml-py"
        )
        sys.exit(1)

    print_cuda_info(cuda_info)

    # Verify only mode
    if args.verify_only:
        print("\n[*] Verifying installation...")
        results = verify_installation(args.skip_flash_attn, args.skip_vllm)
        print("\n" + "=" * 60)
        print("Verification Results:")
        all_ok = True
        for pkg, status in results.items():
            if status.get("installed"):
                print(f"    {pkg}: OK")
            else:
                print(f"    {pkg}: NOT INSTALLED")
                all_ok = False
        sys.exit(0 if all_ok else 1)

    # Check flash-attention GPU compatibility
    min_compute = cuda_info.min_compute_capability
    flash_attn_compatible = check_flash_attn_compatible(min_compute)

    if not flash_attn_compatible and not args.skip_flash_attn:
        print(f"\n[!] WARNING: Min GPU compute capability {min_compute} < 8.0")
        print("    flash-attention requires Ampere (SM 80) or newer GPUs")
        args.skip_flash_attn = True

    # Installation
    print("\n" + "=" * 60)
    print("Starting Installation")
    print("=" * 60)

    success = False

    if args.mode == "pinned":
        preset = PINNED_PRESETS[args.preset]
        print(f"\n[*] Using pinned preset: {preset.name}")
        print(f"    {preset.description}")

        # Check Python version
        if not check_python_version(preset):
            print(
                f"\n[!] ERROR: Python {sys.version_info.major}.{sys.version_info.minor} "
                f"not compatible with preset (requires {preset.python_min}-{preset.python_max})"
            )
            sys.exit(1)

        # Check CUDA version compatibility
        try:
            preset_cuda = int(preset.cuda_version[2:])  # e.g., "cu121" -> 121
            preset_major = preset_cuda // 10
            preset_minor = preset_cuda % 10
            system_major, system_minor = map(int, cuda_info.max_cuda_version.split("."))

            if (preset_major, preset_minor) > (system_major, system_minor):
                print(
                    f"\n[!] WARNING: Preset requires CUDA {preset_major}.{preset_minor}, "
                    f"but system supports max {cuda_info.max_cuda_version}"
                )
                print("    Installation may fail or produce incompatible binaries")
        except (ValueError, AttributeError):
            pass

        # Apply skip flags to preset
        if args.skip_vllm:
            preset = PinnedVersions(
                name=preset.name,
                description=preset.description,
                cuda_version=preset.cuda_version,
                torch=preset.torch,
                torchvision=preset.torchvision,
                torchaudio=preset.torchaudio,
                vllm=None,
                flash_attn=preset.flash_attn if not args.skip_flash_attn else None,
                python_min=preset.python_min,
                python_max=preset.python_max,
            )
        elif args.skip_flash_attn:
            preset = PinnedVersions(
                name=preset.name,
                description=preset.description,
                cuda_version=preset.cuda_version,
                torch=preset.torch,
                torchvision=preset.torchvision,
                torchaudio=preset.torchaudio,
                vllm=preset.vllm,
                flash_attn=None,
                python_min=preset.python_min,
                python_max=preset.python_max,
            )

        success = install_packages_pinned(preset, args.dry_run)
    else:
        # Auto mode
        cuda_tag = select_pytorch_cuda_version(cuda_info.max_cuda_version)
        if cuda_tag is None:
            print(
                f"[!] ERROR: No compatible PyTorch version for CUDA {cuda_info.max_cuda_version}"
            )
            print("    Supported CUDA versions: " + ", ".join(CUDA_TO_PYTORCH.keys()))
            sys.exit(1)

        print(f"\n[*] Auto-selected CUDA version: {cuda_tag}")
        success = install_packages_auto(
            cuda_tag,
            args.skip_vllm,
            args.skip_flash_attn,
            args.dry_run,
        )

    if not success:
        print("\n[!] ERROR: Installation failed")
        sys.exit(1)

    # Verify installation
    if not args.dry_run:
        print("\n" + "=" * 60)
        print("Verifying Installation")
        print("=" * 60)
        results = verify_installation(args.skip_flash_attn, args.skip_vllm)

        print("\n" + "=" * 60)
        print("Installation Complete!")
        print("=" * 60)
        all_ok = all(r.get("installed", False) for r in results.values())
        for pkg, status in results.items():
            status_str = "OK" if status.get("installed") else "FAILED"
            print(f"    {pkg}: {status_str}")

        if not all_ok:
            print("\n[!] Some packages failed verification")
            sys.exit(1)
    else:
        print("\n" + "=" * 60)
        print("Dry Run Complete!")
        print("=" * 60)

    print("\nDone!")


if __name__ == "__main__":
    main()
