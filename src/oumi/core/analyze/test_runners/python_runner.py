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

"""Python test runner - executes custom Python functions."""

from typing import Any

import pandas as pd

from oumi.core.analyze.test_result import TestResult
from oumi.core.analyze.test_runners.base import BaseTestRunner
from oumi.core.configs.params.test_params import TestParams
from oumi.utils.logging import logger


class PythonTestRunner(BaseTestRunner):
    """Runner for custom Python function tests.

    Executes a user-provided Python function to perform custom validation.
    The function receives the DataFrames and summary, and must return a dict
    with specific keys.

    Function signature:
        def check(message_df, conversation_df, summary) -> dict

    Required return keys:
        - passed: bool - Whether the test passed
        - affected_samples: int - Number of affected samples

    Optional return keys:
        - sample_indices: list[int] - Indices of affected samples
        - details: dict - Additional details
        - actual_value: float - Computed value for display
        - threshold: float - Threshold used

    Example config:
        - id: custom_diversity_check
          type: python
          function: |
            def check(message_df, conversation_df, summary):
                if 'diversity__unique_ratio' in message_df.columns:
                    low_div = message_df['diversity__unique_ratio'] < 0.3
                    return {
                        'passed': low_div.mean() < 0.1,
                        'affected_samples': int(low_div.sum()),
                        'sample_indices': message_df[low_div].index.tolist()[:20],
                    }
                return {'passed': True, 'affected_samples': 0}
          severity: medium
    """

    def run(
        self,
        test: TestParams,
        message_df: pd.DataFrame,
        conversation_df: pd.DataFrame,
        summary: dict[str, Any],
    ) -> TestResult:
        """Execute the Python function test.

        Args:
            test: Test configuration with function code.
            message_df: Message-level DataFrame.
            conversation_df: Conversation-level DataFrame.
            summary: Analysis summary.

        Returns:
            TestResult based on function output.
        """
        if test.function is None:
            return self.create_error_result(
                test, "No function specified for python test"
            )

        # Execute the Python function
        try:
            result_dict = self._execute_function(
                test.function, message_df, conversation_df, summary
            )
        except Exception as e:
            logger.warning(f"Python test '{test.id}' failed with error: {e}")
            return self.create_error_result(test, f"Function execution failed: {e}")

        # Validate result structure
        if not isinstance(result_dict, dict):
            return self.create_error_result(
                test,
                f"Function must return a dict, got {type(result_dict).__name__}",
            )

        if "passed" not in result_dict:
            return self.create_error_result(
                test, "Function result must include 'passed' key"
            )

        if "affected_samples" not in result_dict:
            return self.create_error_result(
                test, "Function result must include 'affected_samples' key"
            )

        # Extract values from result
        passed = bool(result_dict["passed"])
        affected_samples = int(result_dict["affected_samples"])
        sample_indices = result_dict.get("sample_indices", [])
        details = result_dict.get("details", {})
        actual_value = result_dict.get("actual_value")
        threshold = result_dict.get("threshold")

        # Get total samples based on scope
        df = self.get_dataframe(test, message_df, conversation_df)
        total_samples = len(df)

        return self.create_result(
            test=test,
            passed=passed,
            affected_samples=affected_samples,
            total_samples=total_samples,
            threshold=threshold,
            actual_value=actual_value,
            details=details,
            sample_indices=sample_indices,
        )

    def _execute_function(
        self,
        function_code: str,
        message_df: pd.DataFrame,
        conversation_df: pd.DataFrame,
        summary: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute the Python function code.

        Args:
            function_code: Python code containing a 'check' function.
            message_df: Message-level DataFrame.
            conversation_df: Conversation-level DataFrame.
            summary: Analysis summary.

        Returns:
            Result dictionary from the function.

        Raises:
            Exception: If function execution fails.
        """
        # Create a restricted namespace for execution
        namespace: dict[str, Any] = {
            "pd": pd,
            "__builtins__": {
                # Allow safe builtins
                "len": len,
                "sum": sum,
                "min": min,
                "max": max,
                "abs": abs,
                "round": round,
                "int": int,
                "float": float,
                "str": str,
                "bool": bool,
                "list": list,
                "dict": dict,
                "set": set,
                "tuple": tuple,
                "range": range,
                "enumerate": enumerate,
                "zip": zip,
                "map": map,
                "filter": filter,
                "sorted": sorted,
                "any": any,
                "all": all,
                "isinstance": isinstance,
                "hasattr": hasattr,
                "getattr": getattr,
                "True": True,
                "False": False,
                "None": None,
            },
        }

        # Try to import numpy if available
        try:
            import numpy as np

            namespace["np"] = np
        except ImportError:
            pass

        # Execute the function definition
        exec(function_code, namespace)

        # Check that 'check' function was defined
        if "check" not in namespace:
            raise ValueError(
                "Function code must define a 'check' function. "
                "Example: def check(message_df, conversation_df, summary): ..."
            )

        check_func = namespace["check"]

        # Call the function with the DataFrames and summary
        return check_func(message_df, conversation_df, summary)
