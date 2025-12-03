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

from unittest.mock import patch

import pytest
import typer
from typer.testing import CliRunner

from oumi.cli.bootstrap import (
    ModelInfo,
    bootstrap,
    detect_architecture,
    generate_config_yaml,
    get_lora_target_modules,
    get_min_transformers_version,
    get_transformer_layer_class,
    parse_context_length,
    parse_model_size,
    select_best_variants,
)

runner = CliRunner()


@pytest.fixture
def app():
    fake_app = typer.Typer()
    fake_app.command()(bootstrap)
    return fake_app


@pytest.fixture
def mock_model_info_3b():
    return ModelInfo(
        model_id="mistralai/Ministral-3-3B-Instruct-2512",
        model_name="Ministral-3-3B-Instruct-2512",
        size_billions=3.8,
        context_length=4096,
        architecture="mistral",
        is_instruct=True,
        is_base=False,
        is_reasoning=False,
        license="apache-2.0",
        downloads=1000,
        likes=50,
    )


@pytest.fixture
def mock_model_info_8b():
    return ModelInfo(
        model_id="mistralai/Ministral-3-8B-Instruct-2512",
        model_name="Ministral-3-8B-Instruct-2512",
        size_billions=8.9,
        context_length=4096,
        architecture="mistral",
        is_instruct=True,
        is_base=False,
        is_reasoning=False,
        license="apache-2.0",
        downloads=500,
        likes=25,
    )


@pytest.fixture
def mock_model_info_14b():
    return ModelInfo(
        model_id="mistralai/Ministral-3-14B-Instruct-2512",
        model_name="Ministral-3-14B-Instruct-2512",
        size_billions=13.9,
        context_length=4096,
        architecture="mistral",
        is_instruct=True,
        is_base=False,
        is_reasoning=False,
        license="apache-2.0",
        downloads=200,
        likes=10,
    )


@pytest.fixture
def mock_model_info_70b():
    return ModelInfo(
        model_id="meta-llama/Llama-3.3-70B-Instruct",
        model_name="Llama-3.3-70B-Instruct",
        size_billions=70.0,
        context_length=8192,
        architecture="llama",
        is_instruct=True,
        is_base=False,
        is_reasoning=False,
        license="llama3",
        downloads=10000,
        likes=500,
    )


class TestParseModelSize:
    def test_parse_from_name_7b(self):
        assert parse_model_size("meta-llama/Llama-3-7B") == 7.0

    def test_parse_from_name_70b(self):
        assert parse_model_size("meta-llama/Llama-3-70B") == 70.0

    def test_parse_from_name_0_5b(self):
        assert parse_model_size("Qwen/Qwen3-0.5B") == 0.5

    def test_parse_from_name_1_5b(self):
        assert parse_model_size("Qwen/Qwen3-1.5B") == 1.5

    def test_parse_from_safetensors(self):
        assert parse_model_size("some/model", safetensors_params=7_000_000_000) == 7.0

    def test_parse_moe_model(self):
        # 8x7B = 56B total
        assert parse_model_size("mistralai/Mixtral-8x7B") == 56.0

    def test_parse_default(self):
        # Should return 7.0 as default when can't parse
        assert parse_model_size("unknown/model-name") == 7.0


class TestParseContextLength:
    def test_max_position_embeddings(self):
        assert parse_context_length({"max_position_embeddings": 4096}) == 4096

    def test_n_positions(self):
        assert parse_context_length({"n_positions": 2048}) == 2048

    def test_default(self):
        assert parse_context_length({}) == 4096


class TestDetectArchitecture:
    def test_detect_from_model_id_llama(self):
        assert detect_architecture("meta-llama/Llama-3-8B", {}) == "llama"

    def test_detect_from_model_id_mistral(self):
        assert detect_architecture("mistralai/Mistral-7B", {}) == "mistral"

    def test_detect_from_model_id_qwen(self):
        assert detect_architecture("Qwen/Qwen3-8B", {}) == "qwen3"

    def test_detect_from_config(self):
        config = {"architectures": ["LlamaForCausalLM"]}
        assert detect_architecture("unknown/model", config) == "llama"

    def test_default_to_llama(self):
        assert detect_architecture("unknown/model", {}) == "llama"


class TestGetTransformerLayerClass:
    def test_llama(self):
        assert get_transformer_layer_class("llama") == "LlamaDecoderLayer"

    def test_mistral(self):
        assert get_transformer_layer_class("mistral") == "MistralDecoderLayer"

    def test_qwen(self):
        assert get_transformer_layer_class("qwen") == "Qwen2DecoderLayer"

    def test_unknown(self):
        assert get_transformer_layer_class("unknown") == "LlamaDecoderLayer"


class TestGetLoraTargetModules:
    def test_llama_has_all_modules(self):
        modules = get_lora_target_modules("llama")
        assert "q_proj" in modules
        assert "k_proj" in modules
        assert "v_proj" in modules
        assert "o_proj" in modules
        assert "gate_proj" in modules

    def test_phi3_has_fused_modules(self):
        modules = get_lora_target_modules("phi3")
        assert "qkv_proj" in modules
        assert "gate_up_proj" in modules

    def test_unknown_gets_default(self):
        modules = get_lora_target_modules("unknown_arch")
        assert modules == ["q_proj", "k_proj", "v_proj"]


class TestGetMinTransformersVersion:
    def test_qwen3_requires_recent(self):
        version = get_min_transformers_version("qwen3")
        assert version == "4.51.0"

    def test_llama_no_special_version(self):
        version = get_min_transformers_version("llama")
        assert version is None

    def test_deepseek_requires_version(self):
        version = get_min_transformers_version("deepseek")
        assert version == "4.46.0"


class TestSelectBestVariants:
    def test_small_model_gets_full(self, mock_model_info_3b):
        models = [mock_model_info_3b]
        configs = select_best_variants(models)

        assert len(configs) == 1
        assert configs[0] == (mock_model_info_3b, "full")

    def test_medium_model_gets_full_and_lora(self, mock_model_info_8b):
        models = [mock_model_info_8b]
        configs = select_best_variants(models)

        assert len(configs) == 2
        assert (mock_model_info_8b, "full") in configs
        assert (mock_model_info_8b, "lora") in configs

    def test_large_model_gets_lora(self, mock_model_info_14b):
        models = [mock_model_info_14b]
        configs = select_best_variants(models)

        assert len(configs) == 1
        assert configs[0] == (mock_model_info_14b, "lora")

    def test_very_large_model_gets_qlora(self, mock_model_info_70b):
        models = [mock_model_info_70b]
        configs = select_best_variants(models)

        assert len(configs) == 1
        assert configs[0] == (mock_model_info_70b, "qlora")

    def test_max_configs_limit(
        self, mock_model_info_3b, mock_model_info_8b, mock_model_info_14b
    ):
        models = [mock_model_info_3b, mock_model_info_8b, mock_model_info_14b]
        configs = select_best_variants(models, max_configs=2)

        assert len(configs) == 2


class TestGenerateConfigYaml:
    def test_small_model_full_no_fsdp(self, mock_model_info_3b, tmp_path):
        filename, content = generate_config_yaml(mock_model_info_3b, "full", tmp_path)

        assert filename == "train.yaml"
        assert 'model_name: "mistralai/Ministral-3-3B-Instruct-2512"' in content
        assert "use_peft" not in content
        assert "enable_fsdp" not in content
        assert "per_device_train_batch_size: 4" in content
        assert "learning_rate: 2e-05" in content

    def test_medium_model_full_with_fsdp(self, mock_model_info_8b, tmp_path):
        filename, content = generate_config_yaml(mock_model_info_8b, "full", tmp_path)

        assert filename == "train.yaml"
        assert 'model_name: "mistralai/Ministral-3-8B-Instruct-2512"' in content
        assert "enable_fsdp: True" in content
        assert 'sharding_strategy: "FULL_SHARD"' in content
        assert 'transformer_layer_cls: "MistralDecoderLayer"' in content

    def test_lora_config(self, mock_model_info_14b, tmp_path):
        filename, content = generate_config_yaml(mock_model_info_14b, "lora", tmp_path)

        assert "use_peft: True" in content
        assert "lora_r: 16" in content
        assert "lora_alpha: 32" in content
        assert '"q_proj"' in content
        assert '"k_proj"' in content
        assert '"v_proj"' in content
        assert "q_lora" not in content

    def test_qlora_config(self, mock_model_info_70b, tmp_path):
        filename, content = generate_config_yaml(mock_model_info_70b, "qlora", tmp_path)

        assert "use_peft: True" in content
        assert "q_lora: True" in content
        assert 'bnb_4bit_quant_type: "nf4"' in content
        assert "enable_fsdp: True" in content


class TestBootstrapCommand:
    @patch("oumi.cli.bootstrap.fetch_model_info")
    def test_bootstrap_single_model_dry_run(self, mock_fetch, app, mock_model_info_3b):
        mock_fetch.return_value = mock_model_info_3b

        result = runner.invoke(
            app,
            [
                "https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512",
                "--dry-run",
            ],
        )

        assert result.exit_code == 0
        assert "Found 1 model(s)" in result.stdout
        assert "Dry run complete" in result.stdout
        assert "Would create" in result.stdout

    @patch("oumi.cli.bootstrap.fetch_collection_models")
    @patch("oumi.cli.bootstrap.fetch_model_info")
    def test_bootstrap_collection_dry_run(
        self,
        mock_fetch_info,
        mock_fetch_collection,
        app,
        mock_model_info_3b,
        mock_model_info_8b,
    ):
        mock_fetch_collection.return_value = [
            "mistralai/Ministral-3-3B-Instruct-2512",
            "mistralai/Ministral-3-8B-Instruct-2512",
        ]
        mock_fetch_info.side_effect = [mock_model_info_3b, mock_model_info_8b]

        result = runner.invoke(
            app,
            [
                "https://huggingface.co/collections/mistralai/ministral-3",
                "--dry-run",
            ],
        )

        assert result.exit_code == 0
        assert "Found 2 model(s)" in result.stdout
        assert "Dry run complete" in result.stdout

    @patch("oumi.cli.bootstrap.fetch_model_info")
    def test_bootstrap_creates_files(
        self, mock_fetch, app, mock_model_info_3b, tmp_path
    ):
        mock_fetch.return_value = mock_model_info_3b

        result = runner.invoke(
            app,
            [
                "https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512",
                "--output",
                str(tmp_path / "test_configs"),
            ],
        )

        assert result.exit_code == 0
        assert "Bootstrap complete!" in result.stdout

        # Check that files were created
        config_dir = tmp_path / "test_configs" / "sft" / "4b_full"
        assert config_dir.exists()
        assert (config_dir / "train.yaml").exists()
        assert (tmp_path / "test_configs" / "README.md").exists()

    def test_bootstrap_invalid_url(self, app):
        result = runner.invoke(
            app,
            ["invalid-url", "--dry-run"],
        )

        assert result.exit_code == 1
        assert "Invalid HuggingFace URL" in result.stdout
