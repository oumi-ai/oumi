# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Derived from https://github.com/huggingface/trl/blob/main/examples/scripts/openenv/echo.py.

This file was slightly modified to be an Oumi rollout registry function.
"""

import requests
from envs.echo_env import EchoEnv
from envs.echo_env.models import EchoAction

from oumi.core.registry import RegistryType, register


@register("echo_env_vllm_rollout", RegistryType.ROLLOUT_FUNCTION)
def echo_env_vllm_rollout(
    prompts: list[str], args, processing_class
) -> dict[str, list]:
    """Custom rollout function that generates completions via vLLM server and computes environment rewards.

    Args:
        prompts: List of prompts to generate from
        args: GRPOConfig containing all sampling parameters
        processing_class: Tokenizer/processor for decoding completions

    Returns:
        Dict containing prompt_ids, completion_ids, logprobs, and env_reward
    """  # noqa: E501
    # 1. Generate completions via vLLM inference server (running on port 8000)
    payload = {
        "prompts": prompts,
        "n": args.num_generations,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": -1 if args.top_k is None else args.top_k,
        "min_p": 0.0 if args.min_p is None else args.min_p,
        "max_tokens": args.max_completion_length,
        "repetition_penalty": args.repetition_penalty,
    }
    response = requests.post("http://0.0.0.0:8000/generate/", json=payload)

    if response.status_code != 200:
        print(f"Error response: {response.text}")

    response.raise_for_status()
    result = response.json()

    completions_text = processing_class.batch_decode(
        result["completion_ids"], skip_special_tokens=True
    )

    # 2. Step through the environment to get rewards
    client = EchoEnv(base_url="http://0.0.0.0:8001")
    env_result = client.reset()
    env_rewards = []
    for msg in completions_text:
        env_result = client.step(EchoAction(message=msg))
        env_rewards.append(env_result.reward)

    # 3. Add environment rewards as extra field
    result["env_reward"] = env_rewards

    return result
