#!/usr/bin/env python3
"""Script to run generated image synthesis code artifacts and create a HF dataset.

Generates PIL images from code artifacts and creates a dataset with all existing
columns plus Image column.
"""

import argparse
import gc
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional

from datasets import Dataset, Features, Value
from datasets import Image as HFImage
from huggingface_hub import create_repo
from PIL import Image


def resize_and_convert_image(
    image_path: Path, max_size: int = 1024
) -> Optional[Image.Image]:
    """Resize image to max size on longest side and convert to RGB without alpha."""
    try:
        img = Image.open(image_path)

        # Convert to RGB if needed (remove alpha channel)
        if img.mode in ("RGBA", "LA", "P"):
            # Create a white background
            background = Image.new("RGB", img.size, (255, 255, 255))
            if img.mode == "P":
                img = img.convert("RGBA")
            background.paste(
                img, mask=img.split()[-1] if img.mode in ("RGBA", "LA") else None
            )
            img = background
        elif img.mode != "RGB":
            img = img.convert("RGB")

        # Resize if needed
        width, height = img.size
        if width > max_size or height > max_size:
            if width > height:
                new_width = max_size
                new_height = int(height * (max_size / width))
            else:
                new_height = max_size
                new_width = int(width * (max_size / height))
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        return img
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return None


def preprocess_code(code: str) -> str:
    """Fix common compatibility issues in generated code."""
    if not code:
        return code

    # Fix matplotlib seaborn style deprecation
    code = code.replace("plt.style.use('seaborn')", "plt.style.use('seaborn-v0_8')")
    code = code.replace('plt.style.use("seaborn")', 'plt.style.use("seaborn-v0_8")')

    # Fix other common seaborn style variations
    code = code.replace("style='seaborn'", "style='seaborn-v0_8'")
    code = code.replace('style="seaborn"', 'style="seaborn-v0_8"')

    # Add matplotlib backend setting for headless environments
    if "import matplotlib" in code and "matplotlib.use(" not in code:
        code = 'import matplotlib\nmatplotlib.use("Agg")\n' + code

    # Add common imports if missing
    lines = code.split("\n")
    has_matplotlib = any(
        "import matplotlib" in line or "from matplotlib" in line for line in lines
    )
    has_numpy = any("import numpy" in line or "from numpy" in line for line in lines)
    has_requests = any("import requests" in line for line in lines)
    has_os = any("import os" in line for line in lines)

    imports_to_add = []
    if not has_matplotlib and ("plt." in code or "matplotlib" in code):
        imports_to_add.append(
            'import matplotlib\nmatplotlib.use("Agg")\nimport matplotlib.pyplot as plt'
        )
    if not has_numpy and ("np." in code or "numpy" in code):
        imports_to_add.append("import numpy as np")
    if not has_requests and ("requests." in code):
        imports_to_add.append("import requests")
    if not has_os and ("os." in code):
        imports_to_add.append("import os")

    if imports_to_add:
        code = "\n".join(imports_to_add) + "\n" + code

    # Ensure output image is saved correctly
    if "plt.savefig" in code and "output_image" not in code:
        code = code.replace(
            "plt.savefig(", 'plt.savefig("output_image.png"); plt.savefig('
        )
    elif "plt.show()" in code and "plt.savefig" not in code:
        code = code.replace("plt.show()", 'plt.savefig("output_image.png")\nplt.show()')

    return code


def run_code_artifact(
    code: str, temp_dir: Path, sample_idx: int
) -> Optional[Image.Image]:
    """Run a code artifact and return the generated PIL image."""
    if not code:
        print(f"Sample {sample_idx}: No code artifact")
        return None

    # Preprocess the code to fix common issues
    processed_code = preprocess_code(code)

    # Create a temporary Python file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(processed_code)
        temp_file = f.name

    try:
        # Create a temporary directory for execution
        with tempfile.TemporaryDirectory() as exec_dir:
            # Run the code in the temporary directory
            result = subprocess.run(
                [sys.executable, temp_file],
                cwd=exec_dir,
                capture_output=True,
                text=True,
                timeout=30,  # 30 second timeout
            )

            if result.returncode == 0:
                # Look for the output image
                output_image = Path(exec_dir) / "output_image.png"
                if output_image.exists():
                    # Process and return the image
                    img = resize_and_convert_image(output_image)
                    if img:
                        print(f"Sample {sample_idx}: Image generated successfully")
                        return img
                    else:
                        print(f"Sample {sample_idx}: Failed to process image")
                        return None
                else:
                    print(f"Sample {sample_idx}: Code ran but no image produced")
                    return None
            else:
                print(f"Sample {sample_idx}: Error running code")
                print(f"STDERR: {result.stderr}")
                return None

    except subprocess.TimeoutExpired:
        print(f"Sample {sample_idx}: Code execution timed out")
        return None
    except Exception as e:
        print(f"Sample {sample_idx}: Exception: {str(e)}")
        return None
    finally:
        # Clean up temporary file
        Path(temp_file).unlink()


def create_hf_dataset(
    samples: list[dict[str, Any]], images: list[Optional[Image.Image]], images_dir: Path
) -> Dataset:
    """Create a HuggingFace dataset with all existing columns plus Image column.

    Rows where image generation failed (img is None) are dropped from the dataset.
    """
    # Prepare data dictionary with all columns from the original samples
    data_dict = {}

    # Get all unique keys from samples
    all_keys = set()
    for sample in samples:
        all_keys.update(sample.keys())

    # Initialize columns
    for key in all_keys:
        data_dict[key] = []

    # Add Image column
    data_dict["image"] = []

    # Fill in the data, only for samples where image generation succeeded
    sample_idx = 0
    for sample, img in zip(samples, images):
        # Skip rows where image generation failed
        if img is None:
            sample_idx += 1
            continue

        for key in all_keys:
            data_dict[key].append(sample.get(key, ""))

        # Save image to persistent file and add path to dataset
        image_filename = f"sample_{sample_idx:04d}.png"
        image_path = images_dir / image_filename
        img.save(str(image_path))
        data_dict["image"].append(str(image_path))
        sample_idx += 1

    # Define features explicitly
    features = Features(
        {
            key: Value("string") if key != "image" else HFImage()
            for key in data_dict.keys()
        }
    )

    # Handle non-string values by converting them
    for key in data_dict:
        if key != "image":
            data_dict[key] = [
                str(val) if val is not None else "" for val in data_dict[key]
            ]

    # Create dataset with explicit features
    dataset = Dataset.from_dict(data_dict, features=features)

    return dataset


def main():
    """Run image synthesis and create HuggingFace dataset."""
    parser = argparse.ArgumentParser(
        description="Run image synthesis code and create HuggingFace dataset"
    )
    parser.add_argument("dataset_path", type=str, help="Path to the dataset directory")
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for the HF dataset (default: dataset_path/hf_dataset)",
    )
    parser.add_argument(
        "--push-to-hub", action="store_true", help="Push dataset to HuggingFace Hub"
    )
    parser.add_argument(
        "--repo-id", type=str, help="HuggingFace repo ID (e.g., username/dataset-name)"
    )
    parser.add_argument(
        "--private", action="store_true", help="Make the HuggingFace repo private"
    )
    parser.add_argument(
        "--max-samples", type=int, help="Maximum number of samples to process"
    )

    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    json_path = dataset_path / "dataset.json"

    if not json_path.exists():
        print(f"Error: Dataset JSON not found at {json_path}")
        return 1

    # Load the dataset (handle both JSON and JSONL formats)
    samples = []
    with open(json_path) as f:
        try:
            # Try loading as regular JSON first
            f.seek(0)
            samples = json.load(f)
        except json.JSONDecodeError:
            # If that fails, try loading as JSONL
            f.seek(0)
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))

    # Limit samples if requested
    if args.max_samples:
        samples = samples[: args.max_samples]

    print(f"Processing {len(samples)} samples")

    # Set output directory
    output_dir = (
        Path(args.output_dir) if args.output_dir else dataset_path / "hf_dataset"
    )
    output_dir.mkdir(exist_ok=True, parents=True)

    # Create temporary directory for execution
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create directory for individual images
        images_dir = output_dir / "images"
        images_dir.mkdir(exist_ok=True, parents=True)

        # Generate images for each sample
        images = []
        successful = 0

        for idx, sample in enumerate(samples):
            code = sample.get("image_synthesis_code")
            img = run_code_artifact(code, temp_path, idx)
            images.append(img)
            if img is not None:
                successful += 1
                print(f"Sample {idx}: Image generated successfully")
            else:
                print(f"Sample {idx}: No image to save")

            # Force garbage collection every 10 samples to manage memory
            if (idx + 1) % 10 == 0:
                gc.collect()

        print(f"\nGenerated {successful}/{len(samples)} images successfully")

        # Create HuggingFace dataset
        print("\nCreating HuggingFace dataset...")
        dataset = create_hf_dataset(samples, images, images_dir)

        # Calculate dropped rows
        dropped_rows = len(samples) - len(dataset)
        print(f"Dropped {dropped_rows} rows where image generation failed")
        print(f"Final dataset contains {len(dataset)} rows")

        # Save dataset locally
        dataset_save_path = output_dir / "dataset"
        print(f"Saving dataset to {dataset_save_path}")
        dataset.save_to_disk(str(dataset_save_path))

        # Also save as parquet for easier loading
        parquet_path = output_dir / "dataset.parquet"
        dataset.to_parquet(str(parquet_path))
        print(f"Saved dataset as parquet to {parquet_path}")

        # Clean up memory: clear images list and force garbage collection
        images.clear()
        gc.collect()
        print("Cleared images from memory")

        # Push to hub if requested
        if args.push_to_hub:
            if not args.repo_id:
                print("Error: --repo-id is required when using --push-to-hub")
                return 1

            print(f"\nPushing dataset to HuggingFace Hub: {args.repo_id}")
            try:
                # Create repo if it doesn't exist
                create_repo(
                    repo_id=args.repo_id,
                    repo_type="dataset",
                    private=args.private,
                    exist_ok=True,
                )

                # Push dataset
                dataset.push_to_hub(args.repo_id, private=args.private)
                print(
                    f"Successfully pushed dataset to https://huggingface.co/datasets/{args.repo_id}"
                )
            except Exception as e:
                print(f"Error pushing to hub: {str(e)}")
                return 1

    print("\nDataset creation complete!")
    print(f"Local dataset saved to: {output_dir}")
    print(f"Individual images saved to: {images_dir}")
    print(f"Successfully saved {successful} individual images")
    print(f"Dropped {len(samples) - successful} rows where image generation failed")

    return 0


if __name__ == "__main__":
    sys.exit(main())
