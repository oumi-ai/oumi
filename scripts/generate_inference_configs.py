#!/usr/bin/env python3
"""
Script to generate inference config sets for Oumi models.
Follows CLAUDE.md best practices for creating 4-config sets:
1. NATIVE engine (PyTorch/Transformers)
2. vLLM engine 
3. GGUF + vLLM 
4. GGUF + LlamaCPP (macOS optimized)
"""

import os
import argparse
from pathlib import Path
from typing import Dict, Any, List

def create_config_content(
    model_info: Dict[str, Any], 
    engine_type: str, 
    file_suffix: str
) -> str:
    """Generate YAML content for a specific engine configuration."""
    
    # Engine-specific settings
    engine_configs = {
        'native': {
            'engine': 'NATIVE',
            'requirements': 'pip install oumi',
            'torch_dtype': 'bfloat16',
            'use_gguf': False
        },
        'vllm': {
            'engine': 'VLLM', 
            'requirements': 'pip install oumi[gpu]',
            'torch_dtype': 'bfloat16',
            'use_gguf': False
        },
        'gguf_vllm': {
            'engine': 'VLLM',
            'requirements': 'pip install oumi[gpu]', 
            'torch_dtype': 'float16',
            'use_gguf': True
        },
        'gguf_llamacpp': {
            'engine': 'LLAMACPP',
            'requirements': 'pip install oumi[llama_cpp]',
            'torch_dtype': 'float16',
            'use_gguf': True
        }
    }
    
    config = engine_configs[engine_type]
    
    # Generate header comment
    engine_name = {
        'native': 'NATIVE',
        'vllm': 'vLLM',
        'gguf_vllm': 'vLLM',
        'gguf_llamacpp': 'LlamaCPP'
    }[engine_type]
    
    suffix_desc = {
        'native': '',
        'vllm': '',
        'gguf_vllm': ' GGUF',
        'gguf_llamacpp': ' GGUF (macOS optimized)'
    }[engine_type]
    
    header = f"""# {engine_name} Inference config for {model_info['display_name']}{suffix_desc}.
#
# Requirements:
#   - Run `{config['requirements']}`
#
# Usage:
#   oumi infer -i -c configs/recipes/{model_info['family']}/inference/{file_suffix}.yaml
#
# See Also:
#   - Documentation: https://oumi.ai/docs/en/latest/user_guides/infer/infer.html
#   - Config class: oumi.core.configs.InferenceConfig
#   - Config source: https://github.com/oumi-ai/oumi/blob/main/src/oumi/core/configs/inference_config.py
#   - Other inference configs: configs/**/inference/

"""

    # Model configuration
    model_section = f"""model:
  model_name: "{model_info['gguf_repo'] if config['use_gguf'] else model_info['hf_repo']}\""""
    
    # Add tokenizer for GGUF configs
    if config['use_gguf']:
        model_section += f"""
  tokenizer_name: "{model_info['hf_repo']}\""""
    
    model_section += f"""
  model_max_length: {model_info['context_length']}
  torch_dtype_str: "{config['torch_dtype']}"
  trust_remote_code: True"""
    
    # Add attn_implementation for non-GGUF configs
    if not config['use_gguf']:
        model_section += """
  attn_implementation: "sdpa\""""
    else:
        # For GGUF configs, add attn_implementation after trust_remote_code
        model_section += """
  attn_implementation: "sdpa\""""
        
    # Add GGUF filename
    if config['use_gguf']:
        model_section += f"""
  model_kwargs:
    filename: "{model_info['gguf_filename']}\""""
    
    # Generation configuration
    max_tokens = model_info.get('max_new_tokens', 2048)
    if engine_type == 'native':
        max_tokens = min(max_tokens * 2, model_info['context_length'])  # Allow longer for NATIVE
    
    generation_section = f"""
generation:
  max_new_tokens: {max_tokens}
  temperature: {model_info.get('temperature', 0.7)}
  top_p: {model_info.get('top_p', 0.9)}

engine: {config['engine']}"""
    
    return header + model_section + generation_section

def create_model_configs(model_info: Dict[str, Any], base_dir: str) -> None:
    """Create all 4 config files for a model."""
    
    family = model_info['family']
    base_name = model_info['base_name']
    
    # Create directory
    config_dir = Path(base_dir) / "configs" / "recipes" / family / "inference"
    config_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate 4 configs
    configs = [
        ('native', f"{base_name}_infer"),
        ('vllm', f"{base_name}_vllm_infer"), 
        ('gguf_vllm', f"{base_name}_gguf_infer"),
        ('gguf_llamacpp', f"{base_name}_gguf_macos_infer")
    ]
    
    for engine_type, filename in configs:
        content = create_config_content(model_info, engine_type, filename)
        
        file_path = config_dir / f"{filename}.yaml"
        with open(file_path, 'w') as f:
            f.write(content)
        
        print(f"✅ Created: {file_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate Oumi inference config sets")
    parser.add_argument("--base-dir", default=".", help="Base directory for configs")
    args = parser.parse_args()
    
    # Model definitions - add your models here
    models = [
        {
            'family': 'magnum',
            'base_name': 'v4_12b',
            'display_name': 'Magnum V4 12B',
            'hf_repo': 'anthracite-org/magnum-v4-12b',
            'gguf_repo': 'bartowski/magnum-v4-12b-GGUF',
            'gguf_filename': 'magnum-v4-12b-Q5_K_M.gguf',
            'context_length': 32768,
            'max_new_tokens': 4096,
            'temperature': 0.7,
            'top_p': 0.9
        },
        {
            'family': 'mistral_nemo',
            'base_name': '12b_instruct_2407',
            'display_name': 'Mistral Nemo Instruct 2407 12B',
            'hf_repo': 'mistralai/Mistral-Nemo-Instruct-2407',
            'gguf_repo': 'bartowski/Mistral-Nemo-Instruct-2407-GGUF',
            'gguf_filename': 'Mistral-Nemo-Instruct-2407-Q5_K_M.gguf',
            'context_length': 131072,
            'max_new_tokens': 4096,
            'temperature': 0.7,
            'top_p': 0.9
        },
        {
            'family': 'phi3',
            'base_name': '3_5_mini_instruct',
            'display_name': 'Phi-3.5-mini-instruct',
            'hf_repo': 'microsoft/Phi-3.5-mini-instruct',
            'gguf_repo': 'bartowski/Phi-3.5-mini-instruct-GGUF',
            'gguf_filename': 'Phi-3.5-mini-instruct-Q6_K_L.gguf',
            'context_length': 131072,
            'max_new_tokens': 2048,
            'temperature': 0.7,
            'top_p': 0.9
        },
        {
            'family': 'phi4',
            'base_name': 'mini_instruct',
            'display_name': 'Phi-4-mini-instruct',
            'hf_repo': 'microsoft/Phi-4-mini-instruct',
            'gguf_repo': 'bartowski/microsoft_Phi-4-mini-instruct-GGUF',
            'gguf_filename': 'microsoft_Phi-4-mini-instruct-Q5_K_M.gguf',
            'context_length': 4096,
            'max_new_tokens': 2048,
            'temperature': 0.7,
            'top_p': 0.9
        },
        {
            'family': 'phi4',
            'base_name': '14b',
            'display_name': 'Phi-4 14B',
            'hf_repo': 'microsoft/phi-4',
            'gguf_repo': 'bartowski/phi-4-GGUF',
            'gguf_filename': 'phi-4-Q5_K_M.gguf',
            'context_length': 16384,
            'max_new_tokens': 4096,
            'temperature': 0.7,
            'top_p': 0.9
        },
        {
            'family': 'phi3',
            'base_name': 'medium_128k_instruct',
            'display_name': 'Phi-3-medium-128k-instruct',
            'hf_repo': 'microsoft/Phi-3-medium-128k-instruct',
            'gguf_repo': 'bartowski/Phi-3-medium-128k-instruct-GGUF',
            'gguf_filename': 'Phi-3-medium-128k-instruct-Q5_K_M.gguf',
            'context_length': 131072,
            'max_new_tokens': 4096,
            'temperature': 0.7,
            'top_p': 0.9
        },
        {
            'family': 'phi4',
            'base_name': 'mini_reasoning',
            'display_name': 'Phi-4-mini-reasoning',
            'hf_repo': 'microsoft/Phi-4-mini-reasoning',
            'gguf_repo': 'bartowski/microsoft_Phi-4-mini-reasoning-GGUF',
            'gguf_filename': 'microsoft_Phi-4-mini-reasoning-Q6_K_L.gguf',
            'context_length': 4096,
            'max_new_tokens': 2048,
            'temperature': 0.7,
            'top_p': 0.9
        },
        {
            'family': 'ministral',
            'base_name': '8b_instruct_2410',
            'display_name': 'Ministral-8B-Instruct-2410',
            'hf_repo': 'mistralai/Ministral-8B-Instruct-2410',
            'gguf_repo': 'bartowski/Ministral-8B-Instruct-2410-GGUF',
            'gguf_filename': 'Ministral-8B-Instruct-2410-Q5_K_M.gguf',
            'context_length': 131072,
            'max_new_tokens': 4096,
            'temperature': 0.7,
            'top_p': 0.9
        },
        {
            'family': 'qwen2_5',
            'base_name': '7b_instruct',
            'display_name': 'Qwen2.5-7B-Instruct',
            'hf_repo': 'Qwen/Qwen2.5-7B-Instruct',
            'gguf_repo': 'bartowski/Qwen2.5-7B-Instruct-GGUF',
            'gguf_filename': 'Qwen2.5-7B-Instruct-Q5_K_M.gguf',
            'context_length': 131072,
            'max_new_tokens': 4096,
            'temperature': 0.7,
            'top_p': 0.9
        },
        {
            'family': 'deepseek_r1',
            'base_name': 'distill_qwen_7b',
            'display_name': 'DeepSeek-R1-Distill-Qwen-7B',
            'hf_repo': 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B',
            'gguf_repo': 'bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF',
            'gguf_filename': 'DeepSeek-R1-Distill-Qwen-7B-Q5_K_M.gguf',
            'context_length': 131072,
            'max_new_tokens': 4096,
            'temperature': 0.7,
            'top_p': 0.9
        },
        {
            'family': 'llama3',
            'base_name': '8b_instruct',
            'display_name': 'Meta-Llama-3-8B-Instruct',
            'hf_repo': 'meta-llama/Meta-Llama-3-8B-Instruct',
            'gguf_repo': 'bartowski/Meta-Llama-3-8B-Instruct-GGUF',
            'gguf_filename': 'Meta-Llama-3-8B-Instruct-Q5_K_M.gguf',
            'context_length': 8192,
            'max_new_tokens': 4096,
            'temperature': 0.7,
            'top_p': 0.9
        }
    ]
    
    print("🚀 Generating inference config sets...")
    print(f"📁 Base directory: {os.path.abspath(args.base_dir)}")
    print()
    
    for model_info in models:
        print(f"📝 Creating configs for {model_info['display_name']}...")
        create_model_configs(model_info, args.base_dir)
        print()
    
    print("✅ All inference config sets generated successfully!")
    print()
    print("Next steps:")
    print("1. Review the generated configs for accuracy")
    print("2. Test a few configs to ensure they work")
    print("3. Run: npm run generate-configs (in frontend dir) to update static configs")

if __name__ == "__main__":
    main()