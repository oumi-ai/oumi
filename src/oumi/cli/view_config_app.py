# Copyright 2025 - Oumi
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

"""Textual TUI application for browsing Oumi configuration files."""

import ast
import inspect
from pathlib import Path
from typing import Any, Optional

import yaml
from omegaconf import MISSING, OmegaConf
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal
from textual.widgets import Footer, Header, Input, Static, Tree
from textual.widgets.tree import TreeNode

# Map of config file patterns to their config class paths
CONFIG_TYPE_MAP = {
    "train": "oumi.core.configs.TrainingConfig",
    "sft": "oumi.core.configs.TrainingConfig",
    "dpo": "oumi.core.configs.TrainingConfig",
    "grpo": "oumi.core.configs.TrainingConfig",
    "eval": "oumi.core.configs.EvaluationConfig",
    "evaluation": "oumi.core.configs.EvaluationConfig",
    "infer": "oumi.core.configs.InferenceConfig",
    "inference": "oumi.core.configs.InferenceConfig",
    "synth": "oumi.core.configs.SynthesisConfig",
    "synthesis": "oumi.core.configs.SynthesisConfig",
    "job": "oumi.core.configs.JobConfig",
    "judge": "oumi.core.configs.JudgeConfig",
    "quantiz": "oumi.core.configs.QuantizationConfig",
}

# Map section names to their param class paths
SECTION_CLASS_MAP = {
    "model": "oumi.core.configs.params.model_params.ModelParams",
    "training": "oumi.core.configs.params.training_params.TrainingParams",
    "data": "oumi.core.configs.params.data_params.DataParams",
    "peft": "oumi.core.configs.params.peft_params.PeftParams",
    "fsdp": "oumi.core.configs.params.fsdp_params.FSDPParams",
    "deepspeed": "oumi.core.configs.params.deepspeed_params.DeepSpeedParams",
    "generation": "oumi.core.configs.params.generation_params.GenerationParams",
    # Nested params
    "train": "oumi.core.configs.params.data_params.DatasetSplitParams",
    "validation": "oumi.core.configs.params.data_params.DatasetSplitParams",
    "test": "oumi.core.configs.params.data_params.DatasetSplitParams",
    "grpo": "oumi.core.configs.params.training_params.GRPOParams",
    "dpo": "oumi.core.configs.params.training_params.DPOParams",
    "inference_remote_params": "oumi.core.configs.params.remote_params.RemoteParams",
}


def _extract_field_docstrings(cls: type) -> dict[str, str]:
    """Extract docstrings for dataclass fields from source code.

    Python doesn't natively store attribute docstrings, so we parse the source.

    Args:
        cls: The dataclass class to extract docstrings from.

    Returns:
        Dict mapping field names to their docstrings.
    """
    docstrings = {}

    try:
        source = inspect.getsource(cls)
    except (OSError, TypeError):
        return docstrings

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return docstrings

    # Find the class definition
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            # Iterate through class body looking for field assignments followed by docstrings
            prev_field_name = None
            for item in node.body:
                if isinstance(item, ast.AnnAssign) and isinstance(
                    item.target, ast.Name
                ):
                    prev_field_name = item.target.id
                elif (
                    isinstance(item, ast.Expr)
                    and isinstance(item.value, ast.Constant)
                    and isinstance(item.value.value, str)
                    and prev_field_name
                ):
                    # This is a docstring following a field
                    docstrings[prev_field_name] = item.value.value.strip()
                    prev_field_name = None
                else:
                    prev_field_name = None

    return docstrings


def _extract_field_types(cls: type) -> dict[str, str]:
    """Extract type annotations for dataclass fields.

    Args:
        cls: The dataclass class to extract types from.

    Returns:
        Dict mapping field names to their type strings.
    """
    types = {}
    try:
        hints = getattr(cls, "__annotations__", {})
        for field_name, field_type in hints.items():
            # Convert type to readable string
            if hasattr(field_type, "__origin__"):
                # Generic types like Optional[str], list[int]
                origin = getattr(field_type, "__origin__", None)
                args = getattr(field_type, "__args__", ())
                if origin is not None:
                    origin_name = getattr(origin, "__name__", str(origin))
                    if args:
                        arg_names = ", ".join(
                            getattr(a, "__name__", str(a)) for a in args
                        )
                        types[field_name] = f"{origin_name}[{arg_names}]"
                    else:
                        types[field_name] = origin_name
            elif hasattr(field_type, "__name__"):
                types[field_name] = field_type.__name__
            else:
                types[field_name] = str(field_type)
    except Exception:
        pass
    return types


def _get_class_from_path(class_path: str) -> Optional[type]:
    """Import and return a class from its module path.

    Args:
        class_path: Fully qualified class path like 'oumi.core.configs.TrainingConfig'

    Returns:
        The class object, or None if import fails.
    """
    try:
        parts = class_path.rsplit(".", 1)
        if len(parts) != 2:
            return None
        module_path, class_name = parts
        module = __import__(module_path, fromlist=[class_name])
        return getattr(module, class_name, None)
    except (ImportError, AttributeError, ModuleNotFoundError):
        return None


def _detect_config_class(file_path: Path, config_data: dict) -> Optional[type]:
    """Detect and return the config class for a file.

    Args:
        file_path: Path to the config file.
        config_data: Parsed YAML data.

    Returns:
        The config class, or None if not detected.
    """
    filename = file_path.stem.lower()

    # Check filename patterns
    for pattern, class_path in CONFIG_TYPE_MAP.items():
        if pattern in filename:
            cls = _get_class_from_path(class_path)
            if cls:
                return cls

    # Infer from top-level keys
    top_keys = set(config_data.keys())

    if "training" in top_keys or "peft" in top_keys or "fsdp" in top_keys:
        return _get_class_from_path("oumi.core.configs.TrainingConfig")
    elif "tasks" in top_keys and "generation" in top_keys:
        return _get_class_from_path("oumi.core.configs.EvaluationConfig")
    elif "engine" in top_keys and "generation" in top_keys:
        return _get_class_from_path("oumi.core.configs.InferenceConfig")

    return None


def _get_section_class(section_name: str) -> Optional[type]:
    """Get the param class for a config section.

    Args:
        section_name: Name of the config section (e.g., 'model', 'training').

    Returns:
        The param class, or None if not found.
    """
    class_path = SECTION_CLASS_MAP.get(section_name)
    if class_path:
        return _get_class_from_path(class_path)
    return None


def _get_default_config(cls: type) -> dict:
    """Get the default configuration values from a config class.

    Args:
        cls: The config class to get defaults from.

    Returns:
        Dict with default values, or empty dict if extraction fails.
    """
    try:
        # Create a structured config from the class
        cfg = OmegaConf.structured(cls)
        # Convert to dict, replacing MISSING with a sentinel
        return OmegaConf.to_container(cfg, resolve=False)
    except (ValueError, TypeError, AttributeError) as e:
        # Log specific error types for debugging
        import logging

        logging.getLogger(__name__).debug(f"Failed to get default config: {e}")
        return {}


def _merge_with_defaults(
    yaml_data: dict, defaults: dict, yaml_keys: set
) -> tuple[dict, set]:
    """Merge YAML data with defaults, tracking which keys are from YAML.

    Args:
        yaml_data: The parsed YAML configuration.
        defaults: The default configuration values.
        yaml_keys: Set of dot-paths that were in the original YAML.

    Returns:
        Tuple of (merged dict, set of yaml key paths).
    """
    result = {}
    all_yaml_keys = set(yaml_keys)

    # Start with defaults
    for key, default_value in defaults.items():
        if key in yaml_data:
            yaml_value = yaml_data[key]
            if isinstance(yaml_value, dict) and isinstance(default_value, dict):
                # Recursively merge dicts
                child_yaml_keys = {
                    k.split(".", 1)[1] for k in yaml_keys if k.startswith(f"{key}.")
                }
                merged, child_keys = _merge_with_defaults(
                    yaml_value, default_value, child_yaml_keys
                )
                result[key] = merged
                all_yaml_keys.update(f"{key}.{k}" for k in child_keys)
            else:
                result[key] = yaml_value
                all_yaml_keys.add(key)
        else:
            # Use default value
            result[key] = default_value

    # Add any YAML keys not in defaults
    for key, value in yaml_data.items():
        if key not in result:
            result[key] = value
            all_yaml_keys.add(key)

    return result, all_yaml_keys


def _get_all_yaml_keys(data: dict, prefix: str = "") -> set:
    """Get all dot-paths for keys in the YAML data.

    Args:
        data: The YAML data dict.
        prefix: Current path prefix.

    Returns:
        Set of dot-paths.
    """
    keys = set()
    for key, value in data.items():
        path = f"{prefix}.{key}" if prefix else key
        keys.add(path)
        if isinstance(value, dict):
            keys.update(_get_all_yaml_keys(value, path))
    return keys


def _format_value(value: Any, is_default: bool = False) -> str:
    """Format a config value for display.

    Args:
        value: The value to format.
        is_default: If True, dim the output to indicate it's a default value.
    """
    # Handle MISSING sentinel from OmegaConf
    if value is MISSING or str(value) == "???":
        return "[red italic]REQUIRED[/red italic]"

    dim_start = "[dim]" if is_default else ""
    dim_end = "[/dim]" if is_default else ""

    if value is None:
        return f"{dim_start}[dim]null[/dim]{dim_end}"
    elif isinstance(value, bool):
        color = "green" if value else "red"
        return f"{dim_start}[{color}]{value}[/{color}]{dim_end}"
    elif isinstance(value, str):
        if value.startswith("${"):
            return f"{dim_start}[magenta]{value}[/magenta]{dim_end}"
        return f'{dim_start}[green]"{value}"[/green]{dim_end}'
    elif isinstance(value, (int, float)):
        return f"{dim_start}[yellow]{value}[/yellow]{dim_end}"
    elif isinstance(value, list):
        return f"[dim]list ({len(value)} items)[/dim]"
    elif isinstance(value, dict):
        return f"[dim]dict ({len(value)} keys)[/dim]"
    return f"{dim_start}{value}{dim_end}"


class DocPanel(Static):
    """Panel displaying documentation for the selected field."""

    DEFAULT_CSS = """
    DocPanel {
        width: 100%;
        height: 100%;
        padding: 1 2;
        background: $surface;
        border: solid $primary;
        overflow-y: auto;
    }
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._title = "Documentation"
        self._content = "Select a field to view its documentation."
        self._value = None
        self._is_default = False
        self._field_type = None

    def set_doc(
        self,
        title: str,
        docstring: str,
        value: Any = None,
        is_default: bool = False,
        field_type: Optional[str] = None,
    ) -> None:
        """Set the documentation content."""
        self._title = title
        self._content = docstring
        self._value = value
        self._is_default = is_default
        self._field_type = field_type
        self._refresh_content()

    def set_raw_yaml(self, content: str) -> None:
        """Set raw YAML content with syntax highlighting."""
        # Format YAML with basic highlighting
        lines = []
        for line in content.split("\n"):
            if line.strip().startswith("#"):
                lines.append(f"[dim italic]{line}[/dim italic]")
            elif ":" in line:
                key, _, rest = line.partition(":")
                lines.append(f"[cyan]{key}[/cyan]:{rest}")
            else:
                lines.append(line)

        self._title = "Raw YAML"
        self._content = "\n".join(lines)
        self._value = None
        self._is_default = False
        self._field_type = None
        self.update(
            f"[bold cyan]{self._title}[/bold cyan]\n{'─' * 40}\n\n{self._content}"
        )

    def _refresh_content(self) -> None:
        """Refresh the displayed content."""
        parts = [
            f"[bold cyan]{self._title}[/bold cyan]\n",
            f"{'─' * 40}\n\n",
        ]

        if self._field_type:
            parts.append(
                f"[bold]Type:[/bold] [magenta]{self._field_type}[/magenta]\n\n"
            )

        if self._value is not None:
            value_label = (
                "[bold]Default Value:[/bold]"
                if self._is_default
                else "[bold]Current Value:[/bold]"
            )
            parts.append(f"{value_label} {_format_value(self._value)}\n\n")

        if self._is_default:
            parts.append(
                "[dim italic]This field is using its default value.[/dim italic]\n\n"
            )

        parts.append(self._content or "[dim]No documentation available.[/dim]")

        self.update("".join(parts))


class SearchBar(Static):
    """Search bar for filtering the tree."""

    DEFAULT_CSS = """
    SearchBar {
        dock: top;
        height: 3;
        padding: 0 1;
        background: $surface;
        display: none;
    }

    SearchBar.visible {
        display: block;
    }

    SearchBar Input {
        width: 100%;
    }
    """

    def compose(self) -> ComposeResult:
        yield Input(placeholder="Search fields... (Enter to search, Escape to close)")


class ConfigTree(Tree):
    """Tree widget for displaying config structure."""

    DEFAULT_CSS = """
    ConfigTree {
        width: 100%;
        height: 100%;
        padding: 1;
        background: $surface;
        border: solid $secondary;
    }
    """


class ConfigViewerApp(App):
    """Interactive TUI for browsing Oumi configuration files."""

    TITLE = "Oumi Config Viewer"

    CSS = """
    Screen {
        layout: horizontal;
    }

    #tree-container {
        width: 1fr;
        height: 100%;
        min-width: 30;
        layout: vertical;
    }

    #doc-container {
        width: 1fr;
        height: 100%;
        min-width: 40;
    }

    Header {
        dock: top;
    }

    Footer {
        dock: bottom;
    }

    #loading-indicator {
        dock: bottom;
        height: 1;
        background: $primary-background;
        text-align: center;
    }

    #loading-indicator.hidden {
        display: none;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("d", "toggle_defaults", "Defaults"),
        Binding("r", "toggle_raw", "Raw YAML"),
        Binding("e", "expand_all", "Expand All"),
        Binding("c", "collapse_all", "Collapse All"),
        Binding("/", "search", "Search"),
        Binding("y", "copy_path", "Copy Path"),
        Binding("Y", "copy_value", "Copy Value"),
        Binding("?", "help", "Help"),
        Binding("j", "cursor_down", "Down", show=False),
        Binding("k", "cursor_up", "Up", show=False),
        Binding("h", "collapse", "Collapse", show=False),
        Binding("l", "expand", "Expand", show=False),
        Binding("escape", "close_search", "Close", show=False),
    ]

    def __init__(
        self,
        file_path: Path,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.theme = "flexoki"
        self.file_path = file_path
        self.config_data: dict = {}
        self.default_config: dict = {}
        self.merged_config: dict = {}
        self.yaml_keys: set = set()
        self.config_class: Optional[type] = None
        self.field_docstrings: dict[str, dict[str, str]] = {}
        self.field_types: dict[str, dict[str, str]] = {}
        self.raw_content: str = ""
        self.showing_raw = False
        self.showing_defaults = False
        self.schema_loaded = False
        self.search_term: str = ""

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            with Container(id="tree-container"):
                yield SearchBar(id="search-bar")
                yield ConfigTree("Config", id="config-tree")
                yield Static("Loading schema...", id="loading-indicator")
            with Container(id="doc-container"):
                yield DocPanel(id="doc-panel")
        yield Footer()

    def on_mount(self) -> None:
        """Load the config file when the app mounts."""
        self.sub_title = self.file_path.name
        # Load YAML immediately (fast)
        self._load_yaml()
        self._build_tree()
        # Load schema in background (slow - imports torch, etc.)
        self._load_schema_async()

    def _load_yaml(self) -> None:
        """Load and parse the YAML file (fast operation)."""
        try:
            with open(self.file_path) as f:
                self.raw_content = f.read()
                self.config_data = yaml.safe_load(self.raw_content) or {}
        except yaml.YAMLError as e:
            self.notify(f"YAML parsing error: {e}", severity="error")
            return
        except OSError as e:
            self.notify(f"Error reading file: {e}", severity="error")
            return

        # Track which keys are from the YAML file
        self.yaml_keys = _get_all_yaml_keys(self.config_data)

    @work(thread=True)
    def _load_schema_async(self) -> None:
        """Load config schema in background thread (slow - imports torch, etc.)."""
        # Detect config class and load docstrings/defaults
        config_class = _detect_config_class(self.file_path, self.config_data)
        if config_class:
            docstrings = {"": _extract_field_docstrings(config_class)}
            types = {"": _extract_field_types(config_class)}

            # Load docstrings for each section class (including nested)
            self._load_nested_docstrings(self.config_data, "", docstrings, types)

            # Load default config
            default_config = _get_default_config(config_class)
            # Merge YAML with defaults
            merged_config, _ = _merge_with_defaults(
                self.config_data, default_config, self.yaml_keys
            )

            # Update state on main thread
            self.call_from_thread(
                self._on_schema_loaded,
                config_class,
                docstrings,
                types,
                default_config,
                merged_config,
            )
        else:
            self.call_from_thread(self._on_schema_load_failed)

    def _load_nested_docstrings(
        self, data: dict, path: str, docstrings: dict, types: dict
    ) -> None:
        """Recursively load docstrings for nested param classes."""
        for key, value in data.items():
            section_cls = _get_section_class(key)
            if section_cls:
                full_path = f"{path}.{key}" if path else key
                docstrings[full_path] = _extract_field_docstrings(section_cls)
                types[full_path] = _extract_field_types(section_cls)

            if isinstance(value, dict):
                child_path = f"{path}.{key}" if path else key
                self._load_nested_docstrings(value, child_path, docstrings, types)

    def _on_schema_loaded(
        self,
        config_class: type,
        docstrings: dict,
        types: dict,
        default_config: dict,
        merged_config: dict,
    ) -> None:
        """Called when schema loading completes."""
        self.config_class = config_class
        self.field_docstrings = docstrings
        self.field_types = types
        self.default_config = default_config
        self.merged_config = merged_config
        self.schema_loaded = True

        # Hide loading indicator
        loading = self.query_one("#loading-indicator")
        loading.add_class("hidden")

        # Rebuild tree to show config type
        self._build_tree()
        self.notify(f"Schema loaded: {config_class.__name__}", timeout=3)

    def _on_schema_load_failed(self) -> None:
        """Called when schema loading fails."""
        loading = self.query_one("#loading-indicator")
        loading.update("Schema not detected")
        self.schema_loaded = True  # Mark as done even if failed

    def _build_tree(self) -> None:
        """Build the config tree structure."""
        tree = self.query_one("#config-tree", ConfigTree)
        tree.clear()

        config_type = self.config_class.__name__ if self.config_class else "Config"

        # Choose which data to display
        if self.showing_defaults and self.merged_config:
            display_data = self.merged_config
            label_suffix = " [dim](with defaults)[/dim]"
        else:
            display_data = self.config_data
            label_suffix = ""

        tree.root.set_label(f"[bold]{config_type}[/bold]{label_suffix}")
        tree.root.data = {"path": "", "value": display_data, "key": ""}

        self._add_nodes(tree.root, display_data, "")

        # Expand all nodes by default
        self._expand_all(tree.root)

    def _expand_all(self, node: TreeNode) -> None:
        """Recursively expand all tree nodes."""
        node.expand()
        for child in node.children:
            self._expand_all(child)

    def _collapse_all(self, node: TreeNode) -> None:
        """Recursively collapse all tree nodes."""
        node.collapse()
        for child in node.children:
            self._collapse_all(child)

    def _is_from_yaml(self, path: str) -> bool:
        """Check if a path was in the original YAML file."""
        return path in self.yaml_keys

    def _matches_search(self, key: str, path: str, value: Any) -> bool:
        """Check if a node matches the search term."""
        if not self.search_term:
            return True
        term = self.search_term.lower()
        return term in key.lower() or term in path.lower() or term in str(value).lower()

    def _add_nodes(self, parent: TreeNode, data: Any, path: str, key: str = "") -> None:
        """Recursively add nodes to the tree."""
        if isinstance(data, dict):
            for k, v in data.items():
                child_path = f"{path}.{k}" if path else k
                is_default = self.showing_defaults and not self._is_from_yaml(
                    child_path
                )

                # Skip if doesn't match search
                if self.search_term and not self._node_or_children_match(
                    k, child_path, v
                ):
                    continue

                # Style key differently if it's a default
                key_style = "dim cyan" if is_default else "cyan"

                if isinstance(v, dict):
                    label = f"[{key_style}]{k}[/{key_style}]"
                    if is_default:
                        label += " [dim italic](default)[/dim italic]"
                    node = parent.add(label, expand=False)
                    node.data = {
                        "path": child_path,
                        "value": v,
                        "key": k,
                        "parent": path,
                        "is_default": is_default,
                    }
                    self._add_nodes(node, v, child_path, k)
                elif isinstance(v, list):
                    label = (
                        f"[{key_style}]{k}[/{key_style}] [dim]({len(v)} items)[/dim]"
                    )
                    if is_default:
                        label += " [dim italic](default)[/dim italic]"
                    node = parent.add(label, expand=False)
                    node.data = {
                        "path": child_path,
                        "value": v,
                        "key": k,
                        "parent": path,
                        "is_default": is_default,
                    }
                    self._add_nodes(node, v, child_path, k)
                else:
                    formatted_value = _format_value(v, is_default=is_default)
                    label = f"[{key_style}]{k}[/{key_style}]: {formatted_value}"
                    if is_default:
                        label += " [dim italic](default)[/dim italic]"
                    node = parent.add(label, allow_expand=False)
                    node.data = {
                        "path": child_path,
                        "value": v,
                        "key": k,
                        "parent": path,
                        "is_default": is_default,
                    }
        elif isinstance(data, list):
            for i, item in enumerate(data):
                child_path = f"{path}[{i}]"
                is_default = self.showing_defaults and not self._is_from_yaml(path)

                if isinstance(item, dict):
                    # Try to find a name for the list item
                    item_name = (
                        item.get("task_name")
                        or item.get("name")
                        or item.get("dataset_name")
                        or f"[{i}]"
                    )

                    # Skip if doesn't match search
                    if self.search_term and not self._node_or_children_match(
                        item_name, child_path, item
                    ):
                        continue

                    label = f"[yellow]{item_name}[/yellow]"
                    node = parent.add(label, expand=False)
                    node.data = {
                        "path": child_path,
                        "value": item,
                        "key": str(i),
                        "parent": path,
                        "is_default": is_default,
                    }
                    self._add_nodes(node, item, child_path, str(i))
                else:
                    formatted_value = _format_value(item, is_default=is_default)
                    label = f"[dim][{i}][/dim] {formatted_value}"
                    node = parent.add(label, allow_expand=False)
                    node.data = {
                        "path": child_path,
                        "value": item,
                        "key": str(i),
                        "parent": path,
                        "is_default": is_default,
                    }

    def _node_or_children_match(self, key: str, path: str, value: Any) -> bool:
        """Check if this node or any children match the search."""
        if self._matches_search(key, path, value):
            return True
        if isinstance(value, dict):
            for k, v in value.items():
                if self._node_or_children_match(k, f"{path}.{k}", v):
                    return True
        elif isinstance(value, list):
            for i, item in enumerate(value):
                if self._node_or_children_match(str(i), f"{path}[{i}]", item):
                    return True
        return False

    def _get_docstring_for_path(self, path: str, key: str) -> Optional[str]:
        """Get the docstring for a field at the given path."""
        # Try exact path match first
        if path in self.field_docstrings:
            doc = self.field_docstrings[path].get(key)
            if doc:
                return doc

        # Try parent paths
        parts = path.split(".")
        for i in range(len(parts), 0, -1):
            parent_path = ".".join(parts[:i])
            if parent_path in self.field_docstrings:
                doc = self.field_docstrings[parent_path].get(key)
                if doc:
                    return doc

        # Try section name (first part of path)
        section = parts[0] if parts else ""
        if section in self.field_docstrings:
            doc = self.field_docstrings[section].get(key)
            if doc:
                return doc

        # Fall back to top-level docstrings
        return self.field_docstrings.get("", {}).get(key)

    def _get_type_for_path(self, path: str, key: str) -> Optional[str]:
        """Get the type annotation for a field at the given path."""
        # Similar logic to _get_docstring_for_path
        parts = path.split(".")

        for i in range(len(parts), 0, -1):
            parent_path = ".".join(parts[:i])
            if parent_path in self.field_types:
                field_type = self.field_types[parent_path].get(key)
                if field_type:
                    return field_type

        section = parts[0] if parts else ""
        if section in self.field_types:
            field_type = self.field_types[section].get(key)
            if field_type:
                return field_type

        return self.field_types.get("", {}).get(key)

    def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:
        """Handle tree node selection to show documentation."""
        node = event.node
        if not node.data:
            return

        data = node.data
        key = data.get("key", "")
        value = data.get("value")
        is_default = data.get("is_default", False)
        path = data.get("path", "")

        # Look up docstring and type
        docstring = self._get_docstring_for_path(path, key)
        field_type = self._get_type_for_path(path, key)

        # Try to get docstring from nested class if it's a dict value
        if not docstring and isinstance(value, dict):
            nested_cls = _get_section_class(key)
            if nested_cls:
                docstring = inspect.getdoc(nested_cls)

        doc_panel = self.query_one("#doc-panel", DocPanel)
        title = f"{path}" if path else "Root"
        doc_panel.set_doc(
            title,
            docstring or "No documentation available for this field.",
            value if not isinstance(value, (dict, list)) else None,
            is_default=is_default,
            field_type=field_type,
        )

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle search input submission."""
        self.search_term = event.value
        self._build_tree()

        search_bar = self.query_one("#search-bar")
        search_bar.remove_class("visible")

        if self.search_term:
            self.notify(f"Filtering by: {self.search_term}")
        else:
            self.notify("Filter cleared")

    def action_search(self) -> None:
        """Show search bar."""
        search_bar = self.query_one("#search-bar")
        search_bar.add_class("visible")
        search_input = search_bar.query_one(Input)
        search_input.value = self.search_term
        search_input.focus()

    def action_close_search(self) -> None:
        """Close search bar, or quit if nothing is active."""
        search_bar = self.query_one("#search-bar")
        if search_bar.has_class("visible"):
            search_bar.remove_class("visible")
            self.search_term = ""
            self._build_tree()
        else:
            # Nothing active, quit the app
            self.exit()

    def action_copy_path(self) -> None:
        """Copy the current field path to clipboard."""
        tree = self.query_one("#config-tree", ConfigTree)
        if tree.cursor_node and tree.cursor_node.data:
            path = tree.cursor_node.data.get("path", "")
            if path:
                try:
                    import pyperclip

                    pyperclip.copy(path)
                    self.notify(f"Copied path: {path}")
                except ImportError:
                    self.notify(
                        "Install pyperclip for clipboard support", severity="warning"
                    )
            else:
                self.notify("No path to copy", severity="warning")

    def action_copy_value(self) -> None:
        """Copy the current field value to clipboard."""
        tree = self.query_one("#config-tree", ConfigTree)
        if tree.cursor_node and tree.cursor_node.data:
            value = tree.cursor_node.data.get("value")
            if value is not None:
                try:
                    import pyperclip

                    if isinstance(value, (dict, list)):
                        import json

                        value_str = json.dumps(value, indent=2)
                    else:
                        value_str = str(value)
                    pyperclip.copy(value_str)
                    self.notify("Copied value to clipboard")
                except ImportError:
                    self.notify(
                        "Install pyperclip for clipboard support", severity="warning"
                    )
            else:
                self.notify("No value to copy", severity="warning")

    def action_cursor_down(self) -> None:
        """Move cursor down in tree."""
        tree = self.query_one("#config-tree", ConfigTree)
        tree.action_cursor_down()

    def action_cursor_up(self) -> None:
        """Move cursor up in tree."""
        tree = self.query_one("#config-tree", ConfigTree)
        tree.action_cursor_up()

    def action_expand(self) -> None:
        """Expand current node."""
        tree = self.query_one("#config-tree", ConfigTree)
        if tree.cursor_node:
            tree.cursor_node.expand()

    def action_collapse(self) -> None:
        """Collapse current node."""
        tree = self.query_one("#config-tree", ConfigTree)
        if tree.cursor_node:
            tree.cursor_node.collapse()

    def action_expand_all(self) -> None:
        """Expand all tree nodes."""
        tree = self.query_one("#config-tree", ConfigTree)
        self._expand_all(tree.root)

    def action_collapse_all(self) -> None:
        """Collapse all tree nodes."""
        tree = self.query_one("#config-tree", ConfigTree)
        self._collapse_all(tree.root)
        tree.root.expand()  # Keep root expanded

    def action_toggle_defaults(self) -> None:
        """Toggle showing default values."""
        if not self.schema_loaded:
            self.notify("Schema still loading...", severity="warning")
            return

        if not self.config_class:
            self.notify(
                "No config schema detected - cannot show defaults", severity="warning"
            )
            return

        self.showing_defaults = not self.showing_defaults
        self._build_tree()

        status = "ON" if self.showing_defaults else "OFF"
        self.notify(f"Show defaults: {status}")

    def action_toggle_raw(self) -> None:
        """Toggle raw YAML view."""
        self.showing_raw = not self.showing_raw
        doc_panel = self.query_one("#doc-panel", DocPanel)

        if self.showing_raw:
            doc_panel.set_raw_yaml(self.raw_content)
        else:
            doc_panel.set_doc(
                "Documentation",
                "Select a field to view its documentation.",
            )

    def action_help(self) -> None:
        """Show help."""
        help_text = """[bold]Keyboard Shortcuts[/bold]

[cyan]Navigation:[/cyan]
  j/↓     Move down
  k/↑     Move up
  h/←     Collapse node
  l/→     Expand node
  Enter   Select/toggle node

[cyan]View:[/cyan]
  d       Toggle default values display
  e       Expand all nodes
  c       Collapse all nodes
  r       Toggle raw YAML view

[cyan]Search & Copy:[/cyan]
  /       Search/filter fields
  Escape  Clear search
  y       Copy field path
  Y       Copy field value

[cyan]Other:[/cyan]
  q/Esc   Quit
  ?       Show this help

[cyan]Legend:[/cyan]
  [cyan]field[/cyan]         Value from YAML file
  [dim cyan]field[/dim cyan]         Default value (when 'd' toggled)
  [red italic]REQUIRED[/red italic]   Missing required field
"""
        doc_panel = self.query_one("#doc-panel", DocPanel)
        doc_panel.set_doc("Help", help_text)
