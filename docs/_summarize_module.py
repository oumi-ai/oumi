import importlib
import inspect
import os
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer()

GITHUB_BASE_URL = "https://github.com/oumi-ai/oumi/blob/main"


@app.command()
def summarize_module(
    module_name: str = typer.Argument(..., help="The name of the module to inspect"),
    filter_type: Optional[list[str]] = typer.Option(
        None, help="Filter for object types (class, method, attribute, function)"
    ),
    output_file: Optional[str] = typer.Option(
        None, help="File path to save the generated markdown"
    ),
    parent_class: Optional[str] = typer.Option(
        None, help="Only include children of this class (format: module.ClassName)"
    ),
    exclude_imported: bool = typer.Option(
        False, help="Only include objects defined in the module, not imported"
    ),
) -> str:
    """Generate a markdown table of objects defined in a Python module.

    Args:
        module_name: The name of the module to inspect.
        filter_type: Optional filter for object types.
            Can be 'class', 'method', 'attribute', 'function', or a list of these.
        output_file: Optional file path to save the generated markdown.
        parent_class: Optional parent class to filter children.
        exclude_imported: If True, only include objects defined in the module,
            not imported.

    Returns:
        A string containing the generated markdown table.

    Raises:
        ImportError: If the specified module cannot be imported.
        ValueError: If an invalid filter_type is provided.
    """
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        raise ImportError(f"Could not import module '{module_name}'")

    valid_filters = ["class", "method", "attribute", "function"]
    if filter_type:
        if isinstance(filter_type, str):
            filter_type = [filter_type]

        if not all(f in valid_filters for f in filter_type):
            raise ValueError(f"Invalid filter_type. Must be one of {valid_filters}")

    parent = None
    if parent_class:
        try:
            parent_module, parent_class_name = parent_class.rsplit(".", 1)
            parent_module = importlib.import_module(parent_module)
            parent = getattr(parent_module, parent_class_name)
        except (ValueError, ImportError, AttributeError):
            raise ValueError(f"Invalid parent_class: {parent_class}")

    #
    # List of objects to be included in the markdown table
    #
    objects = []

    for name, obj in inspect.getmembers(module):
        if name.startswith("_"):  # skip private objects
            continue

        if exclude_imported and not _is_defined_in_module(obj, module):
            continue

        obj_type = _get_object_type(obj)

        if filter_type and obj_type not in filter_type:
            continue

        if parent and not _is_child_of(obj, parent):
            continue

        docstring = _get_object_docstring(obj, summary=True)
        reference = "{py:obj}" + f"`~{module_name}.{name}`"

        objects.append(
            {
                "name": name,
                "type": obj_type,
                "description": docstring,
                "reference": reference,
            }
        )

    #
    # Generate the markdown table
    #
    markdown = "| Name | Description | Reference |\n"
    markdown += "|------|-------------|-----------|\n"

    for obj in sorted(objects, key=lambda x: x["name"]):
        markdown += f"| {obj['name']} | {obj['description']} | {obj['reference']} |\n"

    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(markdown)
    else:
        print(markdown)

    return markdown


@app.command()
def summarize_configs(
    config_folder: str = typer.Argument(..., help="The folder containing config files"),
    config_class: str = typer.Argument(
        ..., help="The class to instantiate configs (format: module.ClassName)"
    ),
    output_file: Optional[str] = typer.Option(
        None, help="File path to save the generated markdown"
    ),
):
    """Generate a markdown table summarizing config files in a folder.

    Args:
        config_folder: The folder containing config files.
        config_class: The class to instantiate configs (format: module.ClassName).
        output_file: Optional file path to save the generated markdown.

    Returns:
        A string containing the generated markdown table.

    Raises:
        ImportError: If the specified config_class cannot be imported.
        ValueError: If the config_folder does not exist.
    """
    config_path = Path(config_folder)
    if not config_path.is_dir():
        raise ValueError(f"Config folder does not exist: {config_folder}")

    try:
        module_name, class_name = config_class.rsplit(".", 1)
        module = importlib.import_module(module_name)
        config_cls = getattr(module, class_name)
    except (ValueError, ImportError, AttributeError):
        raise ImportError(f"Could not import config class: {config_class}")

    configs = []
    for config_file in config_path.rglob("*.yaml"):
        try:
            config = config_cls.from_yaml(config_file)
            configs.append(
                {
                    "name": Path(config_file).stem,
                    "type": config.__class__.__name__,
                    "path": os.path.relpath(config_file, config_folder),
                    "github_link": f"{GITHUB_BASE_URL}/{config_file}",
                }
            )
        except Exception:
            pass

    markdown = "| Name | Path | GitHub Link |\n"
    markdown += "|------|------|-------------|\n"

    for config in sorted(configs, key=lambda x: x["name"]):
        markdown += (
            f"| {config['name']} | {config['path']} "
            f"| [View on GitHub]({config['github_link']}) |\n"
        )

    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(markdown)
    else:
        print(markdown)

    return markdown


def _get_object_docstring(obj, summary: bool = True) -> str:
    """Get the docstring of an object."""
    docstring = inspect.getdoc(obj) or "No description available"
    if summary:
        return docstring.split("\n")[0]
    return docstring


def _get_object_type(obj) -> str:
    """Get the type of an object."""
    if inspect.isclass(obj):
        return "class"
    elif inspect.isfunction(obj):
        return "function"
    elif inspect.ismodule(obj):
        return "module"
    elif isinstance(obj, property):
        return "property"
    elif inspect.ismethod(obj):
        return "method"
    elif isinstance(obj, (classmethod, staticmethod)):
        return "method"
    else:
        return "other"


def _is_child_of(obj, parent_class) -> bool:
    """Check if an object is a child of a parent class."""
    return (
        inspect.isclass(obj) and issubclass(obj, parent_class) and obj != parent_class
    )


def _is_defined_in_module(obj, module):
    """Check if an object is defined in the given module."""
    try:
        return inspect.getmodule(obj) == module
    except AttributeError:
        # Some objects might not have a module, assume they're not defined in the module
        return False


if __name__ == "__main__":
    app()
