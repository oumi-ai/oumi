"""Web and shell command handler."""

import subprocess
from typing import Optional

from rich.panel import Panel
from rich.text import Text

from oumi_chat.commands.base_handler import BaseCommandHandler, CommandResult
from oumi_chat.commands.command_parser import ParsedCommand


class WebShellHandler(BaseCommandHandler):
    """Handles web and shell commands: fetch, shell."""

    def get_supported_commands(self) -> list[str]:
        """Get list of commands this handler supports."""
        return ["fetch", "shell"]

    def handle_command(self, command: ParsedCommand) -> CommandResult:
        """Handle web/shell commands."""
        if command.command == "fetch":
            return self._handle_fetch(command)
        elif command.command == "shell":
            return self._handle_shell(command)
        else:
            return CommandResult(
                success=False,
                message=f"Unsupported command: {command.command}",
                should_continue=False,
            )

    def _handle_fetch(self, command: ParsedCommand) -> CommandResult:
        """Handle the /fetch(url) command to fetch web content."""
        if not command.args:
            return CommandResult(
                success=False,
                message="fetch command requires a URL argument",
                should_continue=False,
            )

        url = command.args[0].strip()

        try:
            # Fetch web content
            content = self._fetch_web_content(url)

            if content:
                # Add to conversation
                fetch_msg = {
                    "role": "attachment",
                    "file_name": url,
                    "file_type": "web",
                    "content": content,
                    "text_content": content,
                }
                self.conversation_history.append(fetch_msg)

                # Display success
                self.console.print(
                    Panel(
                        Text(f"Fetched content from {url}\\n\\nLength: {len(content)} chars"),
                        title="Web Fetch",
                        border_style="cyan",
                    )
                )

                # Update context monitor
                self._update_context_in_monitor()

                return CommandResult(
                    success=True,
                    message=f"Fetched content from {url}",
                    should_continue=False,
                )
            else:
                return CommandResult(
                    success=False,
                    message=f"Failed to fetch content from {url}",
                    should_continue=False,
                )

        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Error fetching URL: {str(e)}",
                should_continue=False,
            )

    def _handle_shell(self, command: ParsedCommand) -> CommandResult:
        """Handle the /shell(command) command to execute shell commands."""
        if not command.args:
            return CommandResult(
                success=False,
                message="shell command requires a command argument",
                should_continue=False,
            )

        shell_command = " ".join(command.args).strip()

        # Security check - block dangerous commands
        dangerous_patterns = [
            "rm -rf",
            "dd if=",
            "mkfs",
            "> /dev",
            ":(){ :|:& };:",  # Fork bomb
            "sudo",
            "chmod 777",
        ]

        for pattern in dangerous_patterns:
            if pattern in shell_command.lower():
                return CommandResult(
                    success=False,
                    message=f"Blocked potentially dangerous command: {shell_command}",
                    should_continue=False,
                )

        try:
            # Execute command
            result = subprocess.run(
                shell_command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30,  # 30 second timeout
            )

            output = result.stdout if result.returncode == 0 else result.stderr

            # Add to conversation
            shell_msg = {
                "role": "attachment",
                "file_name": f"shell: {shell_command}",
                "file_type": "shell",
                "content": output,
                "text_content": output,
            }
            self.conversation_history.append(shell_msg)

            # Display output
            panel_style = "green" if result.returncode == 0 else "red"
            self.console.print(
                Panel(
                    Text(output if output else "(no output)"),
                    title=f"Shell Output (exit code: {result.returncode})",
                    border_style=panel_style,
                )
            )

            # Update context monitor
            self._update_context_in_monitor()

            return CommandResult(
                success=result.returncode == 0,
                message=f"Executed: {shell_command}",
                should_continue=False,
            )

        except subprocess.TimeoutExpired:
            return CommandResult(
                success=False,
                message=f"Command timed out after 30 seconds: {shell_command}",
                should_continue=False,
            )
        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Error executing command: {str(e)}",
                should_continue=False,
            )

    def _fetch_web_content(self, url: str) -> Optional[str]:
        """Fetch content from a URL.

        Args:
            url: URL to fetch.

        Returns:
            Fetched content as string, or None if failed.
        """
        try:
            import requests
            from bs4 import BeautifulSoup

            # Add http:// if no scheme
            if not url.startswith(("http://", "https://")):
                url = "https://" + url

            # Fetch with timeout
            response = requests.get(url, timeout=10)
            response.raise_for_status()

            # Parse HTML and extract text
            soup = BeautifulSoup(response.content, "html.parser")

            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()

            # Get text
            text = soup.get_text()

            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = "\\n".join(chunk for chunk in chunks if chunk)

            return text

        except ImportError:
            # Fallback if requests/beautifulsoup not available
            try:
                import urllib.request

                with urllib.request.urlopen(url, timeout=10) as response:
                    content = response.read().decode("utf-8")
                    return content
            except Exception:
                return None
        except Exception:
            return None
