import datetime
import json
import uuid
from pathlib import Path
from typing import Any

from oumi.utils.logging import logger


def log_example_for_debugging(
    raw_example: Any,
    formatted_example: str,
    tokenized_example: list[tuple[int, str]],
    model_input: dict[str, Any],
) -> None:
    """Logs an example of the data in each step for debugging purposes.

    Args:
        raw_example: The raw example from the dataset.
        formatted_example: The formatted example after processing.
        tokenized_example: The tokenized example after tokenization.
        model_input: The final model input after collating.
    """
    # Log to debug file
    logger.debug("Raw example: %s", raw_example)
    logger.debug("Formatted example: %s", formatted_example)
    logger.debug("Tokenized example: %s", tokenized_example)
    logger.debug("Model input: %s", model_input)

    # Generate timestamp for the debug file
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    session_id = str(uuid.uuid4())[:6]

    # Format data for HTML display
    def format_for_html(obj):
        try:
            return json.dumps(obj, indent=2, default=str)
        except Exception:
            return str(obj)

    # Create simplified HTML content
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Debug Log {timestamp}</title>
    <style>
        body {{ font-family: sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        .info {{ color: #666; margin-bottom: 20px; }}
        .section {{ margin-bottom: 30px; }}
        h2 {{ background: #eee; padding: 8px; }}
        pre {{ background: #f8f8f8; padding: 10px; overflow: auto;
             border: 1px solid #ddd; }}
        .copy-btn {{ float: right; padding: 3px 8px; background: #eee;
                  border: 1px solid #ccc; cursor: pointer; }}
    </style>
</head>
<body>
    <h1>Oumi Debug Information</h1>
    <div class="section">
        <h2>Raw Example</h2>
        <pre id="raw">{format_for_html(raw_example)}</pre>
    </div>

    <div class="section">
        <h2>Formatted Example</h2>
        <pre id="formatted">{formatted_example}</pre>
    </div>

    <div class="section">
        <h2>Tokenized Example</h2>
        <pre id="tokenized">{format_for_html(tokenized_example)}</pre>
    </div>

    <div class="section">
        <h2>Model Input</h2>
        <pre id="model">{format_for_html(model_input)}</pre>
    </div>
</body>
</html>"""

    # Create output directory and write HTML files
    output_dir = "debug_logs"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Write to timestamped file
    output_file = Path(output_dir) / f"debug_logs_{timestamp}_{session_id}.html"
    with open(output_file, "w") as f:
        f.write(html_content)

    # Also update the latest.html file
    latest_file = Path(output_dir) / "latest.html"
    with open(latest_file, "w") as f:
        f.write(html_content)
