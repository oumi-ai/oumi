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

"""Unit tests for checkpointing functionality."""

import time

import pytest

from oumi.core.prompt_optimization.checkpointing import (
    CheckpointManager,
    OptimizationCheckpoint,
    can_resume_from_checkpoint,
    print_checkpoint_summary,
)


class TestOptimizationCheckpoint:
    """Tests for OptimizationCheckpoint class."""

    def test_checkpoint_creation(self):
        """Test creating a checkpoint."""
        checkpoint = OptimizationCheckpoint(
            optimizer_name="mipro",
            current_trial=5,
            total_trials=10,
            best_score=0.75,
            best_prompt="Test prompt",
            best_demos=[{"input": "test", "output": "result"}],
            training_history=[{"trial": 1, "score": 0.5}],
        )

        assert checkpoint.optimizer_name == "mipro"
        assert checkpoint.current_trial == 5
        assert checkpoint.total_trials == 10
        assert checkpoint.best_score == 0.75
        assert len(checkpoint.best_demos) == 1  # type: ignore[arg-type]
        assert len(checkpoint.training_history) == 1  # type: ignore[arg-type]
        assert not checkpoint.completed

    def test_checkpoint_save_load(self, tmp_path):
        """Test saving and loading a checkpoint."""
        checkpoint_path = tmp_path / "test_checkpoint.json"

        # Create and save
        original = OptimizationCheckpoint(
            optimizer_name="bootstrap",
            current_trial=3,
            total_trials=20,
            best_score=0.85,
            best_prompt="Optimized prompt",
            best_demos=[],
            training_history=[],
            completed=False,
        )

        original.save(checkpoint_path)
        assert checkpoint_path.exists()

        # Load and verify
        loaded = OptimizationCheckpoint.load(checkpoint_path)
        assert loaded is not None
        assert loaded.optimizer_name == original.optimizer_name  # type: ignore[union-attr]
        assert loaded.current_trial == original.current_trial  # type: ignore[union-attr]
        assert loaded.total_trials == original.total_trials  # type: ignore[union-attr]
        assert loaded.best_score == original.best_score  # type: ignore[union-attr]
        assert loaded.best_prompt == original.best_prompt  # type: ignore[union-attr]
        assert loaded.completed == original.completed  # type: ignore[union-attr]

    def test_checkpoint_load_nonexistent(self, tmp_path):
        """Test loading a nonexistent checkpoint."""
        checkpoint_path = tmp_path / "nonexistent.json"
        loaded = OptimizationCheckpoint.load(checkpoint_path)
        assert loaded is None

    def test_checkpoint_to_dict(self):
        """Test converting checkpoint to dictionary."""
        checkpoint = OptimizationCheckpoint(
            optimizer_name="gepa",
            current_trial=7,
            total_trials=15,
            best_score=0.9,
        )

        data = checkpoint.to_dict()
        assert isinstance(data, dict)
        assert data["optimizer_name"] == "gepa"
        assert data["current_trial"] == 7
        assert data["total_trials"] == 15
        assert data["best_score"] == 0.9

    def test_checkpoint_from_dict(self):
        """Test creating checkpoint from dictionary."""
        data = {
            "checkpoint_version": "1.0",
            "timestamp": time.time(),
            "optimizer_name": "mipro",
            "current_trial": 10,
            "total_trials": 50,
            "best_score": 0.88,
            "best_prompt": "Test",
            "best_demos": [],
            "training_history": [],
            "completed": False,
            "metadata": {},
        }

        checkpoint = OptimizationCheckpoint.from_dict(data)
        assert checkpoint.optimizer_name == "mipro"
        assert checkpoint.current_trial == 10
        assert checkpoint.best_score == 0.88


class TestCheckpointManager:
    """Tests for CheckpointManager class."""

    def test_manager_creation(self, tmp_path):
        """Test creating a checkpoint manager."""
        manager = CheckpointManager(
            output_dir=str(tmp_path), optimizer_name="bootstrap"
        )

        assert manager.output_dir == tmp_path
        assert manager.optimizer_name == "bootstrap"
        assert manager.checkpoint_path == tmp_path / "checkpoint.json"
        assert not manager.checkpoint_exists()

    def test_save_and_load_checkpoint(self, tmp_path):
        """Test saving and loading through manager."""
        manager = CheckpointManager(output_dir=str(tmp_path), optimizer_name="mipro")

        # Save checkpoint
        manager.save_checkpoint(
            current_trial=5,
            total_trials=20,
            best_score=0.75,
            best_prompt="Test prompt",
            best_demos=[],
            training_history=[],
            completed=False,
        )

        assert manager.checkpoint_exists()

        # Load checkpoint
        checkpoint = manager.load_checkpoint()
        assert checkpoint is not None
        assert checkpoint.current_trial == 5  # type: ignore[union-attr]
        assert checkpoint.total_trials == 20  # type: ignore[union-attr]
        assert checkpoint.best_score == 0.75  # type: ignore[union-attr]

    def test_clear_checkpoint(self, tmp_path):
        """Test clearing a checkpoint."""
        manager = CheckpointManager(output_dir=str(tmp_path), optimizer_name="gepa")

        # Create checkpoint
        manager.save_checkpoint(
            current_trial=1,
            total_trials=10,
            best_score=0.5,
            best_prompt="",
            best_demos=[],
            training_history=[],
        )

        assert manager.checkpoint_exists()

        # Clear it
        manager.clear_checkpoint()
        assert not manager.checkpoint_exists()

    def test_checkpoint_interval(self, tmp_path):
        """Test checkpoint interval logic."""
        manager = CheckpointManager(
            output_dir=str(tmp_path), optimizer_name="bootstrap"
        )

        # Set short interval for testing
        manager.checkpoint_interval = 1.0  # 1 second

        # Should save immediately (first time)
        assert manager.should_save_checkpoint()

        # Save a checkpoint
        manager.save_checkpoint(
            current_trial=1,
            total_trials=10,
            best_score=0.5,
            best_prompt="",
            best_demos=[],
            training_history=[],
        )

        # Should not save immediately after (within interval)
        assert not manager.should_save_checkpoint()

        # Wait for interval
        time.sleep(1.1)

        # Should save now
        assert manager.should_save_checkpoint()


class TestCheckpointResume:
    """Tests for checkpoint resume functionality."""

    def test_can_resume_valid_checkpoint(self):
        """Test resuming from a valid checkpoint."""
        checkpoint = OptimizationCheckpoint(
            optimizer_name="mipro",
            current_trial=5,
            total_trials=20,
            best_score=0.75,
            completed=False,
        )

        can_resume, reason = can_resume_from_checkpoint(
            checkpoint, optimizer_name="mipro", num_trials=20
        )

        assert can_resume
        assert reason is None

    def test_cannot_resume_completed(self):
        """Test cannot resume from completed optimization."""
        checkpoint = OptimizationCheckpoint(
            optimizer_name="mipro",
            current_trial=20,
            total_trials=20,
            best_score=0.9,
            completed=True,
        )

        can_resume, reason = can_resume_from_checkpoint(
            checkpoint, optimizer_name="mipro", num_trials=20
        )

        assert not can_resume
        assert "completed" in reason.lower()  # type: ignore[union-attr]

    def test_cannot_resume_different_optimizer(self):
        """Test cannot resume with different optimizer."""
        checkpoint = OptimizationCheckpoint(
            optimizer_name="mipro",
            current_trial=5,
            total_trials=20,
            best_score=0.75,
            completed=False,
        )

        can_resume, reason = can_resume_from_checkpoint(
            checkpoint, optimizer_name="gepa", num_trials=20
        )

        assert not can_resume
        assert "optimizer" in reason.lower()  # type: ignore[union-attr]

    def test_cannot_resume_different_num_trials(self):
        """Test cannot resume with different number of trials."""
        checkpoint = OptimizationCheckpoint(
            optimizer_name="mipro",
            current_trial=5,
            total_trials=20,
            best_score=0.75,
            completed=False,
        )

        can_resume, reason = can_resume_from_checkpoint(
            checkpoint, optimizer_name="mipro", num_trials=50
        )

        assert not can_resume
        assert "trials" in reason.lower()  # type: ignore[union-attr]

    def test_cannot_resume_none_checkpoint(self):
        """Test cannot resume from None checkpoint."""
        can_resume, reason = can_resume_from_checkpoint(
            None, optimizer_name="mipro", num_trials=20
        )

        assert not can_resume
        assert "no checkpoint" in reason.lower()  # type: ignore[union-attr]

    def test_print_checkpoint_summary(self, capsys):
        """Test printing checkpoint summary."""
        checkpoint = OptimizationCheckpoint(
            optimizer_name="mipro",
            current_trial=10,
            total_trials=50,
            best_score=0.85,
            best_prompt="This is a test prompt for optimization",
            training_history=[{"trial": i, "score": 0.5 + i * 0.01} for i in range(10)],
        )

        print_checkpoint_summary(checkpoint)

        captured = capsys.readouterr()
        assert "RESUMING FROM CHECKPOINT" in captured.out
        assert "mipro" in captured.out
        assert "10/50" in captured.out
        assert "0.85" in captured.out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
