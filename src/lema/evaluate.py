import argparse
import re
import string

from datasets import Dataset, DatasetDict, load_dataset

from lema.core.types import EvaluationConfig
from lema.infer import infer
from lema.logging import logger
from lema.utils.batching import batch, unbatch


def parse_cli():
    """Parse command line arguments and return the configuration filename."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", default=None, help="Path to the configuration file"
    )
    args, arg_list = parser.parse_known_args()
    return args.config, arg_list


def main() -> None:
    """Main entry point for evaluating LeMa.

    Evaluation arguments are fetched from the following sources, ordered by
    decreasing priority:
    1. [Optional] Arguments provided as CLI arguments, in dotfile format
    2. [Optional] Arguments provided in a yaml config file
    3. Default arguments values defined in the data class
    """
    # Load configuration
    config_path, arg_list = parse_cli()

    config: EvaluationConfig = EvaluationConfig.from_yaml_and_arg_list(
        config_path, arg_list, logger=logger
    )

    # Run evaluation
    evaluate(config)


def evaluate(config: EvaluationConfig) -> None:
    """Evaluate a model using the provided configuration.

    Overview:
        This is a hardcoded function, intending to provide a starting point for our
        evaluations. It only works for the MMLU dataset and evaluates a small
        hardcoded portion of its prompts (for testing purposes).
        We need to extend this function, by generalizing the following:
        FIXME (1): Support for multiple datasets.
        Read any dataset, initially downloaded from huggingFace only, later
        also add support for datasets from a local repository. For the earlier, we
        need a `converter registry`, to re-structure known remote datasets into our
        native dataset format (that this function will be compatible with).
        FIXME (2): Support for prompt template library.
        Create a prompt template library and correlate each dataset with a
        relevant prompt template, based on the use case we are targeting.
        There is also a strong correlation between the model and the prompt
        template, so we might need to also do "per model prompt engineering",
        and potentially use different templates for different baseline models.
        FIXME (3): Generalize answer extraction.
        The model's answer is free-form text, but we oftentimes need to identify
        the text that corresponds to the answer within the model's reponse. Note that
        the response may include explanations and reasoning regarding the answer,
        which is not useful for evaluation purposes. For example, to evaluate using
        EM (Exact Match) for multiple-choice Q&A, we need to extract the relevant
        character from the model's response that corresonds to the chosen multiple
        choice option (and compare it vs. the expected answer letter in the evaluation
        data).
        FIXME (4): Metrics
        Create a library of metrics and correlate each dataset with the most
        relevant metric(s). Here, we can reuse HuggingFace or other open-source
        libraries available.

    Args:
        config: The desired configuration for evaluation.

    Returns:
        None for now, we will return a relevant class in the future.
    """
    # Load the dataset from HuggingFace or a local repository.
    if config.data.dataset_name == "cais/mmlu":
        dataset_dict = load_dataset("cais/mmlu", "all")
        assert isinstance(dataset_dict, DatasetDict)
    else:
        # FIXME (1): Support for multiple datasets.
        raise NotImplementedError("Model evaluation only for MMLU for now.")

    # Identify the relevant split within the dataset.
    if config.data.split not in dataset_dict:
        raise ValueError(
            f"Split {config.data.split} not supported for "
            f"dataset {config.data.dataset_name}."
        )
    dataset = dataset_dict[config.data.split]
    assert isinstance(dataset, Dataset)

    # FIXME: Only evaluate a small portion for now for testing purposes.
    # FIXME: It seems that it's simpler to get rid of the complex `Dataset`
    #        class at this point (for simplicity) and cast to a list.
    #        Re-evaluate this choice later.
    dataset = dataset.select(range(8))
    dataset_responses = [entry["answer"] for entry in dataset]  # type: ignore
    dataset = list(dataset)

    # FIXME (2): Support for prompt template library.
    ANSWER_PREFIX = "The correct answer is "
    ANSWER_LETTERS = list(string.ascii_uppercase)

    def dummy_prompt_template(entry):
        def alphabet_letter(index):
            return ANSWER_LETTERS[index]

        def format_multiple_choices(choices_list):
            def choice_fmt(index, choice):
                return f"{alphabet_letter(index)}. {choice}"

            choices = "\n".join([choice_fmt(i, c) for i, c in enumerate(choices_list)])
            return f"Choose the correct answer among the following:\n{choices}"

        def format_question(question):
            return f"Question: {question}"

        formatted_question = format_question(entry["question"])
        formatted_choices = format_multiple_choices(entry["choices"])
        return f"{formatted_question}\n{formatted_choices}\n{ANSWER_PREFIX}"

    dataset = list(map(dummy_prompt_template, dataset))

    # Batch the dataset to items of length `batch_size`.
    batch_size = 4  # FIXME: this should come from the EvaluationConfig.
    dataset_batched = batch(dataset, batch_size)

    # Run inference and then unbatch the model responses.
    responses_batched = infer(
        model_params=config.model,
        generation_config=config.generation,
        input=dataset_batched,
    )
    responses = unbatch(responses_batched)

    # FIXME (3): Generalize answer extraction.
    def dummy_answer_extraction(reponse):
        for index in re.finditer(ANSWER_PREFIX.lower(), reponse.lower()):
            candidate_answer_index = index.start()
            answer = reponse[candidate_answer_index + len(ANSWER_PREFIX)]
            if answer in ANSWER_LETTERS:
                return str(answer)
        return None

    # FIXME (4): Metrics. Using a "custom EM" for now.
    correct_answers_count, incorrect_answers_count, ambigious_answers_count = 0, 0, 0
    for index in range(len(dataset)):
        model_answer = dummy_answer_extraction(responses[index])
        correct_answer_index = dataset_responses[index]
        correct_answer = ANSWER_LETTERS[correct_answer_index]
        if model_answer not in ANSWER_LETTERS:
            ambigious_answers_count += 1
        elif model_answer == correct_answer:
            correct_answers_count += 1
        else:
            incorrect_answers_count += 1

    logger.info(f"Correct answers: {correct_answers_count} of {len(dataset)}")
    logger.info(f"Incorrect answers: {incorrect_answers_count} of {len(dataset)}")
    logger.info(f"Ambigiuous answers: {ambigious_answers_count} of {len(dataset)}")


if __name__ == "__main__":
    main()
