import inspect
from dataclasses import fields, is_dataclass
from typing import List, Type, Union

from oumi.core.configs import (
    BaseConfig,
    EvaluationConfig,
    InferenceConfig,
    JobConfig,
    TrainingConfig,
)


def get_config_summary(config_class: Type[BaseConfig], indent: str = "") -> List[str]:
    """Recursively generate a summary of all attributes for a given configuration class.

    Args:
        config_class (Type[BaseConfig]): The configuration class to summarize.
        indent (str): The current indentation level (used for recursion).

    Returns:
        List[str]: A list of strings, each representing a line in the summary.
    """
    summary = []

    for field in fields(config_class):
        field_type = field.type
        field_name = field.name

        # Handle Optional types
        if hasattr(field_type, "__origin__") and field_type.__origin__ is Union:
            field_type = field_type.__args__[0]

        # If the field is another dataclass, recurse
        if is_dataclass(field_type):
            summary.append(f"{indent}{field_name}:")
            summary.extend(get_config_summary(field_type, indent + "  "))
        else:
            # For simple types, just add the field name and type
            type_name = getattr(field_type, "__name__", str(field_type))
            summary.append(f"{indent}{field_name}: {type_name}")

        # Add the docstring if it exists
        # Check for help metadata (from dataclass field) or docstring
        doc_text = field.metadata.get("help") or inspect.getdoc(
            getattr(config_class, field_name, None)
        )
        if doc_text:
            doc_lines = doc_text.split("\n")
            for line in doc_lines:
                summary.append(f"{indent}  # {line.strip()}")

    return summary


def generate_config_summaries() -> str:
    """Generate summaries for all main configuration classes.

    Returns:
        str: A string containing the summaries of all main configuration classes.
    """
    config_classes = [TrainingConfig, EvaluationConfig, InferenceConfig, JobConfig]
    all_summaries = []

    for config_class in config_classes:
        class_name = config_class.__name__
        all_summaries.append(f"\n## {class_name}\n")
        all_summaries.extend(
            ["```yaml\n"] + get_config_summary(config_class, "") + ["\n```"]
        )

    return "\n".join(all_summaries)


if __name__ == "__main__":
    print(generate_config_summaries())
