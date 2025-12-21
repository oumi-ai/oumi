#!/usr/bin/env python
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

"""Script to generate the config metadata index.

This script iterates through all aliased configs and extracts metadata
to create a searchable index file.

Usage:
    python scripts/generate_config_index.py
    python scripts/generate_config_index.py --verbose
"""

import argparse
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from oumi.cli.config_index import (
    DEFAULT_INDEX_PATH,
    generate_config_index_from_aliases,
    save_config_index,
)


def main():
    parser = argparse.ArgumentParser(
        description="Generate the config metadata index."
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed progress.",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help=f"Output path for the index file. Default: {DEFAULT_INDEX_PATH}",
    )
    args = parser.parse_args()

    print("Generating config index...")

    # Generate index
    index = generate_config_index_from_aliases(verbose=args.verbose)

    # Save to file
    output_path = args.output or DEFAULT_INDEX_PATH
    save_config_index(index, output_path)

    num_configs = len(index.get("configs", {}))
    print(f"Generated index with {num_configs} configs.")
    print(f"Saved to: {output_path}")

    # Print summary by config type
    configs = index.get("configs", {})
    by_type = {}
    for meta in configs.values():
        ct = meta.get("config_type", "unknown")
        by_type[ct] = by_type.get(ct, 0) + 1

    print("\nBy config type:")
    for ct, count in sorted(by_type.items()):
        print(f"  {ct}: {count}")


if __name__ == "__main__":
    main()
