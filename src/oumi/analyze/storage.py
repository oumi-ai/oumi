"""Local storage for analyze eval results.

This module provides persistence for analysis runs, allowing users to:
- Save eval results automatically after each analysis
- List and browse past evals
- Load specific evals for viewing in the web UI
- Delete old evals
"""

import json
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class EvalMetadata(BaseModel):
    """Metadata for a saved eval."""

    id: str = Field(description="Unique identifier for the eval")
    name: str = Field(description="Human-readable name for the eval")
    config_path: str | None = Field(
        default=None, description="Path to the config file used"
    )
    created_at: str = Field(description="ISO timestamp when eval was created")
    dataset_path: str | None = Field(default=None, description="Path to the dataset")
    sample_count: int = Field(default=0, description="Number of samples analyzed")
    pass_rate: float | None = Field(
        default=None, description="Overall pass rate (0.0-1.0)"
    )
    analyzer_count: int = Field(default=0, description="Number of analyzers used")
    test_count: int = Field(default=0, description="Number of tests defined")
    tests_passed: int = Field(default=0, description="Number of tests passed")
    tests_failed: int = Field(default=0, description="Number of tests failed")


class EvalData(BaseModel):
    """Full eval data including results."""

    metadata: EvalMetadata
    config: dict[str, Any] = Field(default_factory=dict, description="The config used")
    analysis_results: dict[str, Any] = Field(
        default_factory=dict, description="Analysis results by analyzer"
    )
    test_results: dict[str, Any] = Field(
        default_factory=dict, description="Test results"
    )
    conversations: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Conversation content for each sample (user/assistant messages)",
    )


class AnalyzeStorage:
    """Local file storage for analyze eval results.

    Storage structure:
        ~/.oumi/analyze/
        ├── index.json          # List of all evals with metadata
        ├── evals/
        │   ├── {uuid}.json     # Full eval results
        │   └── ...
        └── configs/
            └── {name}.yaml     # Saved configs

    Usage:
        storage = AnalyzeStorage()
        storage.save_eval(name="my_eval", results=..., test_results=...)
        evals = storage.list_evals()
        data = storage.load_eval(eval_id)
    """

    DEFAULT_DIR = Path.home() / ".oumi" / "analyze"

    def __init__(self, base_dir: str | Path | None = None):
        """Initialize the storage.

        Args:
            base_dir: Base directory for storage. Defaults to ~/.oumi/analyze/
                      Can also be set via OUMI_ANALYZE_DIR environment variable.
        """
        if base_dir is None:
            base_dir = os.environ.get("OUMI_ANALYZE_DIR", str(self.DEFAULT_DIR))
        self.base_dir = Path(base_dir)
        self.evals_dir = self.base_dir / "evals"
        self.configs_dir = self.base_dir / "configs"
        self.index_path = self.base_dir / "index.json"

        # Ensure directories exist
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        """Create storage directories if they don't exist."""
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.evals_dir.mkdir(exist_ok=True)
        self.configs_dir.mkdir(exist_ok=True)

    def _load_index(self) -> dict[str, Any]:
        """Load the index file."""
        if not self.index_path.exists():
            return {"evals": []}
        try:
            with open(self.index_path) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load index: {e}")
            return {"evals": []}

    def _save_index(self, index: dict[str, Any]) -> None:
        """Save the index file."""
        with open(self.index_path, "w") as f:
            json.dump(index, f, indent=2)

    def save_eval(
        self,
        name: str,
        config: dict[str, Any],
        analysis_results: dict[str, Any],
        test_results: dict[str, Any] | None = None,
        config_path: str | None = None,
        dataset_path: str | None = None,
        conversations: list[dict[str, Any]] | None = None,
    ) -> str:
        """Save an eval to storage.

        Args:
            name: Human-readable name for the eval
            config: The config dict used for the analysis
            analysis_results: Analysis results (analyzer name -> list of results)
            test_results: Test results dict (optional)
            config_path: Path to the config file (optional)
            dataset_path: Path to the dataset (optional)
            conversations: List of conversation dicts with messages (optional)

        Returns:
            The eval ID
        """
        eval_id = str(uuid.uuid4())[:8]
        created_at = datetime.now().isoformat()

        # Calculate statistics from results
        sample_count = 0
        analyzer_count = 0
        if analysis_results:
            analyzer_count = len(analysis_results)
            # Get sample count from first analyzer's results
            for results_list in analysis_results.values():
                if isinstance(results_list, list):
                    sample_count = len(results_list)
                    break

        # Calculate test statistics
        tests_passed = 0
        tests_failed = 0
        test_count = 0
        pass_rate = None
        if test_results:
            # Test results can be under "results" or "tests" key
            tests = test_results.get("results", test_results.get("tests", []))
            test_count = len(tests)
            for test in tests:
                if test.get("passed", False):
                    tests_passed += 1
                else:
                    tests_failed += 1
            if test_count > 0:
                pass_rate = tests_passed / test_count

        # Create metadata
        metadata = EvalMetadata(
            id=eval_id,
            name=name,
            config_path=config_path,
            created_at=created_at,
            dataset_path=dataset_path or config.get("dataset_path"),
            sample_count=sample_count,
            pass_rate=pass_rate,
            analyzer_count=analyzer_count,
            test_count=test_count,
            tests_passed=tests_passed,
            tests_failed=tests_failed,
        )

        # Serialize analysis results (convert Pydantic models to dicts)
        serialized_results = {}
        for analyzer_name, results_list in analysis_results.items():
            if isinstance(results_list, list):
                serialized_results[analyzer_name] = [
                    r.model_dump() if hasattr(r, "model_dump") else r
                    for r in results_list
                ]
            elif hasattr(results_list, "model_dump"):
                serialized_results[analyzer_name] = results_list.model_dump()
            else:
                serialized_results[analyzer_name] = results_list

        # Create full eval data
        eval_data = EvalData(
            metadata=metadata,
            config=config,
            analysis_results=serialized_results,
            test_results=test_results or {},
            conversations=conversations or [],
        )

        # Save eval file
        eval_path = self.evals_dir / f"{eval_id}.json"
        with open(eval_path, "w") as f:
            json.dump(eval_data.model_dump(), f, indent=2, default=str)

        # Update index
        index = self._load_index()
        index["evals"].insert(0, metadata.model_dump())  # Most recent first
        self._save_index(index)

        logger.info(f"Saved eval '{name}' with ID: {eval_id}")
        return eval_id

    def load_eval(self, eval_id: str) -> EvalData | None:
        """Load an eval by ID.

        Args:
            eval_id: The eval ID

        Returns:
            EvalData or None if not found
        """
        eval_path = self.evals_dir / f"{eval_id}.json"
        if not eval_path.exists():
            logger.warning(f"Eval not found: {eval_id}")
            return None

        try:
            with open(eval_path) as f:
                data = json.load(f)
            return EvalData(**data)
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Failed to load eval {eval_id}: {e}")
            return None

    def list_evals(self) -> list[EvalMetadata]:
        """List all saved evals.

        Returns:
            List of EvalMetadata, most recent first
        """
        index = self._load_index()
        return [EvalMetadata(**e) for e in index.get("evals", [])]

    def delete_eval(self, eval_id: str) -> bool:
        """Delete an eval by ID.

        Args:
            eval_id: The eval ID

        Returns:
            True if deleted, False if not found
        """
        eval_path = self.evals_dir / f"{eval_id}.json"
        if eval_path.exists():
            eval_path.unlink()

        # Update index
        index = self._load_index()
        original_count = len(index["evals"])
        index["evals"] = [e for e in index["evals"] if e.get("id") != eval_id]
        self._save_index(index)

        deleted = len(index["evals"]) < original_count
        if deleted:
            logger.info(f"Deleted eval: {eval_id}")
        return deleted

    def rename_eval(self, eval_id: str, new_name: str) -> bool:
        """Rename an eval.

        Args:
            eval_id: The eval ID
            new_name: New name for the eval

        Returns:
            True if renamed, False if not found
        """
        # Update index
        index = self._load_index()
        for e in index["evals"]:
            if e.get("id") == eval_id:
                e["name"] = new_name
                self._save_index(index)

                # Also update the eval file
                eval_data = self.load_eval(eval_id)
                if eval_data:
                    eval_data.metadata.name = new_name
                    eval_path = self.evals_dir / f"{eval_id}.json"
                    with open(eval_path, "w") as f:
                        json.dump(eval_data.model_dump(), f, indent=2, default=str)

                logger.info(f"Renamed eval {eval_id} to '{new_name}'")
                return True
        return False

    def save_config(self, name: str, config: dict[str, Any]) -> Path:
        """Save a config to the configs directory.

        Args:
            name: Name for the config file (without extension)
            config: Config dict to save

        Returns:
            Path to the saved config file
        """
        import yaml

        config_path = self.configs_dir / f"{name}.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        logger.info(f"Saved config: {config_path}")
        return config_path

    def list_configs(self) -> list[Path]:
        """List all saved configs.

        Returns:
            List of config file paths
        """
        return sorted(self.configs_dir.glob("*.yaml"))

    def cleanup_old_evals(self, max_age_days: int = 30, max_count: int = 100) -> int:
        """Remove old evals to save space.

        Args:
            max_age_days: Remove evals older than this many days
            max_count: Keep at most this many evals

        Returns:
            Number of evals deleted
        """
        index = self._load_index()
        evals = index.get("evals", [])
        deleted_count = 0

        # Sort by created_at (most recent first)
        evals.sort(key=lambda e: e.get("created_at", ""), reverse=True)

        # Remove by count
        if len(evals) > max_count:
            to_delete = evals[max_count:]
            for e in to_delete:
                self.delete_eval(e["id"])
                deleted_count += 1
            evals = evals[:max_count]

        # Remove by age
        cutoff = datetime.now()
        from datetime import timedelta

        cutoff = cutoff - timedelta(days=max_age_days)
        cutoff_str = cutoff.isoformat()

        for e in list(evals):
            if e.get("created_at", "") < cutoff_str:
                self.delete_eval(e["id"])
                deleted_count += 1

        if deleted_count > 0:
            logger.info(f"Cleaned up {deleted_count} old evals")
        return deleted_count
