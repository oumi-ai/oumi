PROMPT_TEMPLATE = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}{eos_token}"""  # noqa: E501


def alpaca_preprocessing_fn(tokenizer):
    """Build a preprocessing function for the Alpaca dataset.

    Dataset: https://huggingface.co/datasets/tatsu-lab/alpaca
    """

    def prompt_generation_fn(samples):
        instructions = samples["instruction"]
        inputs = samples["input"]
        outputs = samples["output"]

        prompts = []

        for instruction, input, output in zip(instructions, inputs, outputs):
            prompt = PROMPT_TEMPLATE.format(
                instruction=instruction,
                input=input,
                output=output,
                eos_token=tokenizer.eos_token,
            )
            prompts.append(prompt)

        return {
            "prompt": prompts,
        }

    return prompt_generation_fn
