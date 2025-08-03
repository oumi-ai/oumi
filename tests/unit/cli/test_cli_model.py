from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import typer
from typer.testing import CliRunner

from oumi.cli.model import card, get, ls, rm
from oumi.utils.hf_model_cache_utils import CachedModel

runner = CliRunner()


@pytest.fixture
def app_ls():
    fake_app = typer.Typer()
    fake_app.command()(ls)
    return fake_app


@pytest.fixture
def app_rm():
    fake_app = typer.Typer()
    fake_app.command()(rm)
    return fake_app


@pytest.fixture
def app_get():
    fake_app = typer.Typer()
    fake_app.command()(get)
    return fake_app


@pytest.fixture
def app_card():
    fake_app = typer.Typer()
    fake_app.command()(card)
    return fake_app


@pytest.fixture
def mock_cached_models():
    return [
        CachedModel(
            repo_id="test/model1",
            size_bytes=1000000,
            size="1.0MB",
            repo_path=Path("/cache/model1"),
            last_modified="2024-01-01 10:00:00",
            last_accessed="2024-01-01 11:00:00",
            repo_type="model",
            nb_files=5,
        ),
        CachedModel(
            repo_id="test/model2",
            size_bytes=2000000,
            size="2.0MB",
            repo_path=Path("/cache/model2"),
            last_modified="2024-01-02 10:00:00",
            last_accessed="2024-01-02 11:00:00",
            repo_type="model",
            nb_files=10,
        ),
    ]


class TestLsCommand:
    @patch("oumi.cli.model.list_cached_models")
    def test_ls_with_models(self, mock_list_cached_models, app_ls, mock_cached_models):
        mock_list_cached_models.return_value = mock_cached_models
        result = runner.invoke(app_ls, [])
        assert result.exit_code == 0
        assert "test/model1" in result.stdout
        assert "test/model2" in result.stdout

    @patch("oumi.cli.model.list_cached_models")
    def test_ls_no_models(self, mock_list_cached_models, app_ls):
        mock_list_cached_models.return_value = []
        result = runner.invoke(app_ls, [])
        assert result.exit_code == 0
        assert "No cached models found" in result.stdout

    @patch("oumi.cli.model.list_cached_models")
    def test_ls_with_filter(self, mock_list_cached_models, app_ls, mock_cached_models):
        mock_list_cached_models.return_value = mock_cached_models
        result = runner.invoke(app_ls, ["--filter", "*model1*"])
        assert result.exit_code == 0
        assert "test/model1" in result.stdout
        assert "test/model2" not in result.stdout

    @patch("oumi.cli.model.list_cached_models")
    def test_ls_verbose(self, mock_list_cached_models, app_ls, mock_cached_models):
        mock_list_cached_models.return_value = mock_cached_models

        # Test normal mode
        result_normal = runner.invoke(app_ls, [])
        assert result_normal.exit_code == 0

        # Test verbose mode
        result_verbose = runner.invoke(app_ls, ["--verbose"])
        assert result_verbose.exit_code == 0

        # Verbose mode should have more columns/content
        assert len(result_verbose.stdout) > len(result_normal.stdout)

    @patch("oumi.cli.model.list_cached_models")
    def test_ls_sort_by_name(self, mock_list_cached_models, app_ls, mock_cached_models):
        mock_list_cached_models.return_value = mock_cached_models
        result = runner.invoke(app_ls, ["--sort", "name"])
        assert result.exit_code == 0
        assert "test/model1" in result.stdout
        assert "test/model2" in result.stdout


class TestRmCommand:
    @patch("oumi.cli.model.list_cached_models")
    @patch("oumi.cli.model.shutil.rmtree")
    @patch("typer.confirm")
    def test_rm_with_confirmation(
        self,
        mock_confirm,
        mock_rmtree,
        mock_list_cached_models,
        app_rm,
        mock_cached_models,
    ):
        mock_list_cached_models.return_value = mock_cached_models
        mock_confirm.return_value = True

        result = runner.invoke(app_rm, ["test/model1"])
        assert result.exit_code == 0
        mock_rmtree.assert_called_once_with(Path("/cache/model1"))
        assert "Successfully removed" in result.stdout

    @patch("oumi.cli.model.list_cached_models")
    @patch("oumi.cli.model.shutil.rmtree")
    @patch("typer.confirm")
    def test_rm_cancelled(
        self,
        mock_confirm,
        mock_rmtree,
        mock_list_cached_models,
        app_rm,
        mock_cached_models,
    ):
        mock_list_cached_models.return_value = mock_cached_models
        mock_confirm.return_value = False

        result = runner.invoke(app_rm, ["test/model1"])
        assert result.exit_code == 0
        mock_rmtree.assert_not_called()
        assert "Removal cancelled" in result.stdout

    @patch("oumi.cli.model.list_cached_models")
    @patch("oumi.cli.model.shutil.rmtree")
    def test_rm_force(
        self, mock_rmtree, mock_list_cached_models, app_rm, mock_cached_models
    ):
        mock_list_cached_models.return_value = mock_cached_models

        result = runner.invoke(app_rm, ["test/model1", "--force"])
        assert result.exit_code == 0
        mock_rmtree.assert_called_once_with(Path("/cache/model1"))
        assert "Successfully removed" in result.stdout

    @patch("oumi.cli.model.list_cached_models")
    def test_rm_model_not_found(self, mock_list_cached_models, app_rm):
        mock_list_cached_models.return_value = []

        result = runner.invoke(app_rm, ["nonexistent/model"])
        assert result.exit_code == 1
        assert "not found in cache" in result.stdout


class TestGetCommand:
    @patch("oumi.cli.model.list_cached_models")
    @patch("oumi.cli.model.snapshot_download")
    def test_get_new_model(
        self, mock_snapshot_download, mock_list_cached_models, app_get
    ):
        mock_list_cached_models.return_value = []

        result = runner.invoke(app_get, ["test/new-model"])
        assert result.exit_code == 0
        mock_snapshot_download.assert_called_once_with(
            repo_id="test/new-model", revision=None
        )
        assert "Successfully downloaded" in result.stdout

    @patch("oumi.cli.model.list_cached_models")
    @patch("oumi.cli.model.snapshot_download")
    def test_get_with_revision(
        self, mock_snapshot_download, mock_list_cached_models, app_get
    ):
        mock_list_cached_models.return_value = []

        result = runner.invoke(app_get, ["test/new-model", "--revision", "v1.0"])
        assert result.exit_code == 0
        mock_snapshot_download.assert_called_once_with(
            repo_id="test/new-model", revision="v1.0"
        )

    @patch("oumi.cli.model.list_cached_models")
    @patch("oumi.cli.model.snapshot_download")
    def test_get_already_cached(
        self,
        mock_snapshot_download,
        mock_list_cached_models,
        app_get,
        mock_cached_models,
    ):
        mock_list_cached_models.return_value = mock_cached_models

        result = runner.invoke(app_get, ["test/model1"])
        assert result.exit_code == 0
        mock_snapshot_download.assert_not_called()
        assert "already cached" in result.stdout


class TestCardCommand:
    @patch("oumi.cli.model.list_cached_models")
    @patch("oumi.cli.model.model_info")
    def test_card_cached_model(
        self, mock_model_info, mock_list_cached_models, app_card, mock_cached_models
    ):
        mock_list_cached_models.return_value = mock_cached_models
        mock_info = MagicMock()
        mock_info.pipeline_tag = "text-generation"
        mock_info.downloads = 1000
        mock_info.likes = 50
        mock_info.library_name = "transformers"
        mock_model_info.return_value = mock_info

        result = runner.invoke(app_card, ["test/model1"])
        assert result.exit_code == 0
        assert "Cached locally" in result.stdout
        assert "text-generation" in result.stdout

    @patch("oumi.cli.model.list_cached_models")
    @patch("oumi.cli.model.model_info")
    def test_card_not_cached_model(
        self, mock_model_info, mock_list_cached_models, app_card
    ):
        mock_list_cached_models.return_value = []
        mock_info = MagicMock()
        mock_info.pipeline_tag = "text-generation"
        mock_info.downloads = 1000
        mock_info.likes = 50
        mock_info.library_name = "transformers"
        mock_model_info.return_value = mock_info

        result = runner.invoke(app_card, ["test/new-model"])
        assert result.exit_code == 0
        assert "Not cached locally" in result.stdout
        assert "text-generation" in result.stdout

    @patch("oumi.cli.model.list_cached_models")
    @patch("oumi.cli.model.model_info")
    def test_card_hub_info_error(
        self, mock_model_info, mock_list_cached_models, app_card, mock_cached_models
    ):
        mock_list_cached_models.return_value = mock_cached_models
        mock_model_info.side_effect = Exception("Hub error")

        result = runner.invoke(app_card, ["test/model1"])
        assert result.exit_code == 0
        assert "Unable to fetch from Hugging Face Hub" in result.stdout
