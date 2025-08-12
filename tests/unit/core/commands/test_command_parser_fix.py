"""Unit tests for CommandParser input handling fix."""

import pytest
from oumi.core.commands.command_parser import CommandParser


class TestCommandParserFix:
    """Test suite for the CommandParser fix for handling paths and special content."""

    def setup_method(self):
        """Set up test fixtures."""
        self.parser = CommandParser()

    def test_file_paths_not_detected_as_commands(self):
        """Test that file paths are not incorrectly detected as commands."""
        file_paths = [
            "/Users/user/Documents/file.txt",
            "/var/log/system.log",
            "/opt/myapp/config.ini",
            "/home/user/Desktop/project/notes/file",
            "/absolute/path/without/parentheses",
            "//double/slash/path",
            "/path/with spaces/file.txt",
            "/path-with-hyphens/file",
            "/path_with_underscores/file",
        ]
        
        for path in file_paths:
            assert not self.parser.is_command(path), f"File path incorrectly detected as command: {path}"

    def test_valid_commands_still_detected(self):
        """Test that valid commands are still correctly detected."""
        valid_commands = [
            "/help()",
            "/exit()",
            "/attach(file.pdf)",
            "/save(output.txt)",
            "/set(temperature=0.8)",
            "/import(data.csv)",
            "/swap(model_name)",
            "/branch(experimental)",
            "/ml",
            "/sl",
        ]
        
        for cmd in valid_commands:
            assert self.parser.is_command(cmd), f"Valid command not detected: {cmd}"

    def test_complex_content_not_detected_as_commands(self):
        """Test that complex content with special characters is not detected as commands."""
        complex_content = [
            "$\\boxed{\\text{Yes, you can DIY this project.}}$",
            "if (condition) { return /path/to/file; }",
            "const path = '/usr/local/bin'",
            '{"path": "/home/user", "config": true}',
            "<!-- Comment about /paths -->",
            "ls /home/user",
            "cd /var/www/html", 
            "Please check /var/log/system.log for errors",
            "Der Pfad ist /home/müller/dokumente",
            "路径在 /home/用户/文档",
        ]
        
        for content in complex_content:
            assert not self.parser.is_command(content), f"Complex content incorrectly detected as command: {content}"

    def test_multiline_content_not_detected_as_commands(self):
        """Test that multiline content is not detected as commands."""
        multiline_content = [
            """CONFIG_PATH="/opt/myapp/config"
LOG_PATH="/var/log/myapp"
DATA_DIR="/data/processing\"""",
            """Here's how to fix it:
1. Edit /etc/nginx/sites-available/default
2. Restart nginx: sudo systemctl restart nginx
3. Check logs in /var/log/nginx/""",
        ]
        
        for content in multiline_content:
            assert not self.parser.is_command(content), f"Multiline content incorrectly detected as command: {content}"

    def test_url_schemes_not_detected_as_commands(self):
        """Test that URLs with various schemes are not detected as commands."""
        urls = [
            "file:///Users/test/file.txt",
            "ftp://server.com/path/file", 
            "ssh://user@server/path",
            "https://example.com/path/to/page",
            "http://localhost:8080/api/endpoint",
        ]
        
        for url in urls:
            assert not self.parser.is_command(url), f"URL incorrectly detected as command: {url}"

    def test_invalid_command_syntax_not_detected(self):
        """Test that text starting with / but not matching command syntax is not detected."""
        invalid_syntax = [
            "/command-without-parentheses",
            "/command with spaces",
            "/123numbers",
            "/ single slash with space",
            "/",  # Just a slash
            "//",  # Double slash
        ]
        
        for invalid in invalid_syntax:
            assert not self.parser.is_command(invalid), f"Invalid command syntax incorrectly detected: {invalid}"

    def test_empty_and_none_inputs(self):
        """Test edge cases with empty or None inputs."""
        edge_cases = [
            None,
            "",
            "   ",  # Just whitespace
        ]
        
        for case in edge_cases:
            assert not self.parser.is_command(case), f"Edge case incorrectly detected as command: {case}"

    def test_command_parsing_still_works(self):
        """Test that command parsing functionality is not broken."""
        test_command = "/set(temperature=0.8, top_p=0.9)"
        
        assert self.parser.is_command(test_command)
        
        parsed = self.parser.parse_command(test_command)
        assert parsed is not None
        assert parsed.command == "set"
        assert parsed.kwargs["temperature"] == "0.8"
        assert parsed.kwargs["top_p"] == "0.9"

    def test_command_validation_still_works(self):
        """Test that command validation functionality is not broken."""
        valid_cmd = "/help()"
        invalid_cmd = "/nonexistent()"
        
        # Valid command should parse and validate
        assert self.parser.is_command(valid_cmd)
        parsed_valid = self.parser.parse_command(valid_cmd)
        assert parsed_valid is not None
        is_valid, _ = self.parser.validate_command(parsed_valid)
        assert is_valid
        
        # Invalid command should not be detected as a command at all now
        assert not self.parser.is_command(invalid_cmd)

    def test_special_mode_commands(self):
        """Test that special mode switching commands still work."""
        mode_commands = ["/ml", "/sl"]
        
        for cmd in mode_commands:
            assert self.parser.is_command(cmd), f"Mode command not detected: {cmd}"

    def test_real_world_problematic_content(self):
        """Test with the actual problematic content that caused the issue."""
        problematic_content = '''PROMPT:

"We need to re-direct a dryer vent through the concrete foundation in the basement. Is putting a 4` hole through the concrete wall something I can DIY ?"

RESPONSE:

"1. **Determine the location and size of the hole:**\\n   - Identify the location where the dryer vent needs to be redirected through the concrete foundation.\\n   - Measure the size of the vent pipe to determine the diameter of the hole needed. For a standard dryer vent, the diameter is typically 4 inches.\\n\\n2. **Prepare the area around the hole:**\\n   - Use a hammer and chisel to chip away at the concrete around the hole to create a rough opening. This will help in creating a cleaner and more precise cut with the saw.\\n\\n3. **Use a masonry saw:**\\n   - A masonry saw is designed to cut through concrete, brick, and other masonry materials. It works by using a diamond-encrusted blade that rotates at high speeds to cut through the material.\\n   - Position the masonry saw at the starting point of the hole and guide it along the marked line. Apply steady pressure and let the saw do the work.\\n\\n4. **Clean up the edges:**\\n   - Once the hole is cut, use a hammer and chisel to remove any remaining concrete around the edges of the hole. This will ensure that the hole is smooth and even.\\n\\n5. **Install the vent pipe:**\\n   - Insert the dryer vent pipe into the newly created hole. Ensure that the pipe is properly sealed and secured in place.\\n\\nBy following these steps, you can safely and effectively redirect the dryer vent through the concrete foundation in your basement.\\n\\n$\\\\boxed{\\\\text{Yes, you can DIY this project.}}$<|eot_id|>"

JUDGMENT:'''
        
        # The full content should not be detected as a command
        assert not self.parser.is_command(problematic_content)
        
        # Individual lines should also not be detected as commands
        lines = problematic_content.split('\n')
        for line in lines:
            if line.strip():
                assert not self.parser.is_command(line), f"Line incorrectly detected as command: {line[:50]}..."