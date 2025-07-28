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

"""Test to verify DSParams instantiation works without AttributeError."""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from oumi.core.configs.params.ds_params import DSParams, OffloadConfig


def test_dsparams_instantiation():
    """Test that DSParams can be instantiated without errors."""
    print("Testing DSParams instantiation...")

    try:
        # Test basic instantiation
        params1 = DSParams()
        print("✓ Basic DSParams instantiation successful")

        # Test instantiation with parameters
        params2 = DSParams(enable_deepspeed=True, zero_stage="3")
        print("✓ DSParams with parameters instantiation successful")

        # Test instantiation with nested dataclass
        params3 = DSParams(enable_deepspeed=True, offload_optimizer=OffloadConfig())
        print("✓ DSParams with nested OffloadConfig instantiation successful")

        # Test that validation works
        try:
            # This should raise ValueError
            params4 = DSParams(zero_stage="2", offload_param=OffloadConfig())
        except ValueError as e:
            print(f"✓ Validation error caught as expected: {e}")

        print("\nAll tests passed! No AttributeError with super().__post_init__()")
        return True

    except AttributeError as e:
        if "'super' object has no attribute '__post_init__'" in str(e):
            print(f"✗ FAILED: {e}")
            print("The AttributeError is still present!")
            return False
        else:
            raise


if __name__ == "__main__":
    success = test_dsparams_instantiation()
    sys.exit(0 if success else 1)
