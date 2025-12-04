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

"""Utilities for exporting conversations to various formats."""

import csv
import html
import json
from datetime import datetime

from oumi_chat.commands.command_context import CommandContext


class ExportUtilities:
    """Utility class for exporting conversations to different formats."""

    def __init__(self, context: CommandContext):
        """Initialize export utilities.

        Args:
            context: Shared command context.
        """
        self.context = context
        self.console = context.console
        self._style = context._style

    def export_conversation(
        self, file_path: str, format_type: str, conversation_history: list
    ) -> tuple[bool, str]:
        """Export conversation to the specified format.

        Args:
            file_path: Path where to save the export.
            format_type: Format to export to (pdf, text, markdown, json, csv, html).
            conversation_history: List of conversation messages.

        Returns:
            Tuple of (success, message).
        """
        try:
            if format_type == "pdf":
                return self._export_to_pdf(file_path, conversation_history)
            elif format_type == "text":
                return self._export_to_text(file_path, conversation_history)
            elif format_type == "markdown":
                return self._export_to_markdown(file_path, conversation_history)
            elif format_type == "json":
                return self._export_to_json(file_path, conversation_history)
            elif format_type == "csv":
                return self._export_to_csv(file_path, conversation_history)
            elif format_type == "html":
                return self._export_to_html(file_path, conversation_history)
            else:
                return False, f"Unsupported export format: {format_type}"
        except Exception as e:
            return False, f"Export failed: {str(e)}"

    def _export_to_pdf(
        self, file_path: str, conversation_history: list
    ) -> tuple[bool, str]:
        """Export conversation to PDF format."""
        try:
            from reportlab.lib.colors import HexColor
            from reportlab.lib.enums import TA_LEFT
            from reportlab.lib.pagesizes import letter
            from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
            from reportlab.lib.units import inch
            from reportlab.platypus import (
                Paragraph,
                SimpleDocTemplate,
                Spacer,
            )
        except ImportError:
            return (
                False,
                "PDF export requires reportlab. Install with: pip install reportlab",
            )

        try:
            # Create PDF document
            doc = SimpleDocTemplate(
                file_path,
                pagesize=letter,
                rightMargin=0.75 * inch,
                leftMargin=0.75 * inch,
                topMargin=1 * inch,
                bottomMargin=1 * inch,
            )

            # Get styles
            styles = getSampleStyleSheet()

            # Custom styles for different roles
            user_style = ParagraphStyle(
                "UserMessage",
                parent=styles["Normal"],
                fontSize=11,
                leftIndent=0,
                rightIndent=20,
                textColor=HexColor("#2E86AB"),
                spaceBefore=12,
                spaceAfter=6,
                alignment=TA_LEFT,
            )

            assistant_style = ParagraphStyle(
                "AssistantMessage",
                parent=styles["Normal"],
                fontSize=11,
                leftIndent=20,
                rightIndent=0,
                textColor=HexColor("#A23B72"),
                spaceBefore=12,
                spaceAfter=6,
                alignment=TA_LEFT,
            )

            system_style = ParagraphStyle(
                "SystemMessage",
                parent=styles["Normal"],
                fontSize=10,
                leftIndent=10,
                rightIndent=10,
                textColor=HexColor("#666666"),
                spaceBefore=6,
                spaceAfter=6,
                alignment=TA_LEFT,
            )

            # Build document content
            story = []

            # Add title
            title = Paragraph("Oumi Conversation Export", styles["Title"])
            story.append(title)
            story.append(Spacer(1, 0.2 * inch))

            # Add export info
            export_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            info = Paragraph(f"Exported on: {export_time}", styles["Normal"])
            story.append(info)
            story.append(Spacer(1, 0.3 * inch))

            # Process conversation messages
            for i, msg in enumerate(conversation_history):
                role = msg.get("role", "unknown")
                content = msg.get("content", "")

                # Skip empty messages
                if not content.strip():
                    continue

                # Escape HTML and handle special characters
                safe_content = html.escape(str(content))

                # Choose style based on role
                if role.lower() == "user":
                    style = user_style
                    role_prefix = "👤 User:"
                elif role.lower() == "assistant":
                    style = assistant_style
                    role_prefix = "🤖 Assistant:"
                else:
                    style = system_style
                    role_prefix = f"⚙️ {role.title()}:"

                # Add role header
                role_para = Paragraph(f"<b>{role_prefix}</b>", style)
                story.append(role_para)

                # Add message content
                content_para = Paragraph(safe_content, style)
                story.append(content_para)

                # Add spacing between messages
                if i < len(conversation_history) - 1:
                    story.append(Spacer(1, 0.1 * inch))

            # Build PDF
            doc.build(story)

            return True, f"Conversation exported to PDF: {file_path}"

        except Exception as e:
            return False, f"Error creating PDF: {str(e)}"

    def _export_to_text(
        self, file_path: str, conversation_history: list
    ) -> tuple[bool, str]:
        """Export conversation to plain text format."""
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write("# Oumi Conversation Export\\n")
                f.write(
                    f"# Exported on: "
                    f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n\\n"
                )

                for i, msg in enumerate(conversation_history):
                    role = msg.get("role", "unknown")
                    content = msg.get("content", "")

                    # Skip empty messages
                    if not content.strip():
                        continue

                    # Add role indicator
                    role_emoji = {"user": "👤", "assistant": "🤖", "system": "⚙️"}.get(
                        role.lower(), "❓"
                    )

                    f.write(f"{role_emoji} {role.upper()}:\\n")
                    f.write(f"{content}\\n")

                    # Add separator between messages
                    if i < len(conversation_history) - 1:
                        f.write("\\n" + "=" * 50 + "\\n\\n")

            return True, f"Conversation exported to text: {file_path}"

        except Exception as e:
            return False, f"Error writing text file: {str(e)}"

    def _export_to_markdown(
        self, file_path: str, conversation_history: list
    ) -> tuple[bool, str]:
        """Export conversation to Markdown format."""
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write("# Oumi Conversation Export\\n\\n")
                f.write(
                    f"**Exported on:** "
                    f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n\\n"
                )
                f.write("---\\n\\n")

                for i, msg in enumerate(conversation_history):
                    role = msg.get("role", "unknown")
                    content = msg.get("content", "")

                    # Skip empty messages
                    if not content.strip():
                        continue

                    # Add role header
                    role_emoji = {"user": "👤", "assistant": "🤖", "system": "⚙️"}.get(
                        role.lower(), "❓"
                    )

                    f.write(f"## {role_emoji} {role.title()}\\n\\n")
                    f.write(f"{content}\\n\\n")

                    # Add horizontal rule between messages (except last)
                    if i < len(conversation_history) - 1:
                        f.write("---\\n\\n")

            return True, f"Conversation exported to Markdown: {file_path}"

        except Exception as e:
            return False, f"Error writing Markdown file: {str(e)}"

    def _export_to_json(
        self, file_path: str, conversation_history: list
    ) -> tuple[bool, str]:
        """Export conversation to JSON format."""
        try:
            export_data = {
                "export_info": {
                    "timestamp": datetime.now().isoformat(),
                    "format": "json",
                    "version": "1.0",
                },
                "conversation": conversation_history,
            }

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)

            return True, f"Conversation exported to JSON: {file_path}"

        except Exception as e:
            return False, f"Error writing JSON file: {str(e)}"

    def _export_to_csv(
        self, file_path: str, conversation_history: list
    ) -> tuple[bool, str]:
        """Export conversation to CSV format."""
        try:
            with open(file_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)

                # Write header
                writer.writerow(["timestamp", "role", "content", "message_index"])

                # Write conversation data
                export_time = datetime.now().isoformat()
                for i, msg in enumerate(conversation_history):
                    role = msg.get("role", "unknown")
                    content = msg.get("content", "")

                    # Clean content for CSV
                    clean_content = str(content).replace("\\n", " ").replace("\\r", " ")

                    writer.writerow([export_time, role, clean_content, i])

            return True, f"Conversation exported to CSV: {file_path}"

        except Exception as e:
            return False, f"Error writing CSV file: {str(e)}"

    def _export_to_html(
        self, file_path: str, conversation_history: list
    ) -> tuple[bool, str]:
        """Export conversation to HTML format."""
        try:
            # HTML template with embedded CSS
            html_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Oumi Conversation Export</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto,
                sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f8f9fa;
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            background: white;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .message {{
            margin: 20px 0;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .message.user {{
            background: #e3f2fd;
            margin-left: 40px;
            border-left: 4px solid #2196f3;
        }}
        .message.assistant {{
            background: #fce4ec;
            margin-right: 40px;
            border-left: 4px solid #e91e63;
        }}
        .message.system {{
            background: #f5f5f5;
            margin: 20px 40px;
            border-left: 4px solid #9e9e9e;
        }}
        .role {{
            font-weight: bold;
            margin-bottom: 8px;
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .content {{
            white-space: pre-wrap;
            word-wrap: break-word;
        }}
        .timestamp {{
            font-size: 12px;
            color: #666;
            text-align: center;
            margin: 30px 0;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🤖 Oumi Conversation Export</h1>
        <p class="timestamp">Exported on {timestamp}</p>
    </div>

    {messages}

</body>
</html>"""

            # Generate messages HTML
            messages_html = []
            for msg in conversation_history:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")

                if not content.strip():
                    continue

                # Role emoji mapping
                role_emoji = {"user": "👤", "assistant": "🤖", "system": "⚙️"}.get(
                    role.lower(), "❓"
                )

                # Determine CSS class
                role_class = role if role in ["user", "assistant"] else "system"

                message_html = f"""
    <div class="message {role_class}">
        <div class="role">{role_emoji} {html.escape(role.title())}</div>
        <div class="content">{html.escape(content)}</div>
    </div>"""
                messages_html.append(message_html)

            # Generate final HTML
            final_html = html_template.format(
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                messages="\\n".join(messages_html),
            )

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(final_html)

            return True, f"Conversation exported to HTML: {file_path}"

        except Exception as e:
            return False, f"Error writing HTML file: {str(e)}"
