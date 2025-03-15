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

import os
from typing import Any
from oumi.utils.logging import logger

def log_tokenized_example(raw_example: Any, formatted_example: str, tokenized_example: list[tuple[int, str]], model_input: dict[str, Any]):
    """Logs raw, formatted, tokenized examples, and model input for debugging."""
    logger.debug("Raw Example: %s", raw_example)
    logger.debug("Formatted Example: %s", formatted_example)
    logger.debug("Tokenized Example: %s", tokenized_example)
    logger.debug("Model Input: %s", model_input)

    # Write the debug information into a nicely-formatted HTML file
    html_content = f"""
    <html>
    <head>
        <title>Debug Information</title>
        <style>
            body {{ font-family: Arial, sans-serif; }}
            pre {{ background-color: #f4f4f4; padding: 10px; border: 1px solid #ddd; }}
        </style>
    </head>
    <body>
        <h1>Debug Information</h1>
        <h2>Raw Example</h2>
        <pre>{raw_example}</pre>
        <h2>Formatted Example</h2>
        <pre>{formatted_example}</pre>
        <h2>Tokenized Example</h2>
        <pre>{tokenized_example}</pre>
        <h2>Model Input</h2>
        <pre>{model_input}</pre>
    </body>
    </html>
    """

    output_dir = "debug_logs"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "debug_info.html")
    with open(output_file, "w") as f:
        f.write(html_content)
    logger.info("Debug information written to %s", output_file)
