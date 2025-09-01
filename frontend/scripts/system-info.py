#!/usr/bin/env python3
"""
Lightweight system information detector for Chatterley config selection.
This script provides basic system specs without requiring the full Oumi backend.
"""

import json
import platform
import os
import sys
import subprocess


def get_memory_info():
    """Get system memory information in GB."""
    try:
        if platform.system() == "Darwin":  # macOS
            # Use sysctl to get memory info
            result = subprocess.run(['sysctl', 'hw.memsize'], 
                                  capture_output=True, text=True, check=True)
            bytes_memory = int(result.stdout.split(':')[1].strip())
            return round(bytes_memory / (1024**3))  # Convert to GB
        elif platform.system() == "Linux":
            # Read from /proc/meminfo
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if line.startswith('MemTotal:'):
                        kb = int(line.split()[1])
                        return round(kb / (1024**2))  # Convert to GB
        elif platform.system() == "Windows":
            # Use wmic command
            result = subprocess.run(['wmic', 'computersystem', 'get', 'TotalPhysicalMemory', '/value'], 
                                  capture_output=True, text=True, check=True)
            for line in result.stdout.split('\n'):
                if line.startswith('TotalPhysicalMemory='):
                    bytes_memory = int(line.split('=')[1].strip())
                    return round(bytes_memory / (1024**3))  # Convert to GB
    except Exception:
        pass
    
    # Fallback: return conservative estimate
    return 8


def detect_cuda():
    """Detect CUDA availability and GPU information."""
    try:
        # Try to run nvidia-smi
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, check=True)
        
        # Parse GPU memory sizes
        gpu_memories = []
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                memory_mb = int(line.strip())
                memory_gb = round(memory_mb / 1024, 1)  # Convert MB to GB
                gpu_memories.append({"vram": memory_gb})
        
        return True, gpu_memories
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        return False, []


def get_system_info():
    """Get comprehensive system information."""
    # Basic platform info
    system_info = {
        "platform": platform.system().lower(),
        "architecture": platform.machine().lower(),
        "totalRAM": get_memory_info(),
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    }
    
    # Normalize architecture names
    arch_mapping = {
        "x86_64": "x64",
        "amd64": "x64", 
        "arm64": "arm64",
        "aarch64": "arm64"
    }
    system_info["architecture"] = arch_mapping.get(system_info["architecture"], system_info["architecture"])
    
    # Normalize platform names  
    platform_mapping = {
        "darwin": "darwin",
        "linux": "linux",
        "windows": "win32"
    }
    system_info["platform"] = platform_mapping.get(system_info["platform"], system_info["platform"])
    
    # CUDA detection
    cuda_available, cuda_devices = detect_cuda()
    system_info["cudaAvailable"] = cuda_available
    system_info["cudaDevices"] = cuda_devices
    
    return system_info


def main():
    """Main entry point - output system info as JSON."""
    try:
        info = get_system_info()
        print(json.dumps(info, indent=2))
        return 0
    except Exception as e:
        error_info = {
            "error": str(e),
            "platform": "unknown",
            "architecture": "unknown", 
            "totalRAM": 8,
            "cudaAvailable": False,
            "cudaDevices": []
        }
        print(json.dumps(error_info, indent=2))
        return 1


if __name__ == "__main__":
    sys.exit(main())