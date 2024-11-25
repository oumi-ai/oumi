import os
import sys
from pathlib import Path

from oumi.core.configs import InferenceConfig
from oumi.inference import VLLMInferenceEngine

_INPUT_DIR = "/input_gcs_bucket"
_OUTPUT_DIR = "/output_gcs_bucket"

_INFERENCE_ENV_VAR = "INFERENCE_CONFIG"


def get_new_inputs(source: str, processed: str) -> list[str]:
    """Returns a list of new files in the source directory."""
    source_filenames = {f.name for f in Path(source).rglob("*")}
    processed_filenames = {f.name for f in Path(processed).rglob("*")}
    return list(source_filenames - processed_filenames)


def run_inference(
    file: str,
    input_dir: str,
    output_dir: str,
    config: InferenceConfig,
    engine: VLLMInferenceEngine,
) -> None:
    """Runs inference on a file."""
    config.input_path = str(Path(input_dir) / file)
    config.output_path = str(Path(output_dir) / file)
    print(f"Preparing to read input file: {config.input_path}")
    print(f"Preparing to write output file: {config.output_path}")
    engine.infer_from_file(config.input_path, config)


def main():
    """Runs inference on new files in the input directory."""
    parallelism = 1
    if len(sys.argv) > 1:
        parallelism = int(sys.argv[1])
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    print("Checking for new files...")
    new_files = get_new_inputs(_INPUT_DIR, _OUTPUT_DIR)
    yaml_path = os.getenv(_INFERENCE_ENV_VAR)
    if not yaml_path:
        raise ValueError(f"Environment variable {_INFERENCE_ENV_VAR} not set.")
    config = InferenceConfig.from_yaml(yaml_path)
    engine = VLLMInferenceEngine(
        config.model,
        tensor_parallel_size=parallelism,
    )
    for file in new_files:
        print(f"Running inference on {file}...")
        run_inference(file, _INPUT_DIR, _OUTPUT_DIR, config, engine)
        print(f"Inference complete for {file}.")
    print("All files processed!")


if __name__ == "__main__":
    main()
