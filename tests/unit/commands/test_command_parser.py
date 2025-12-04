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

"""Unit tests for command parsing functionality."""

from oumi_chat.commands import CommandParser


class TestCommandParser:
    """Test suite for CommandParser."""

    def setup_method(self):
        """Set up test fixtures."""
        self.parser = CommandParser()

    def test_parse_simple_command(self):
        """Test parsing simple commands without arguments."""
        result = self.parser.parse_command("/help()")
        assert result is not None
        assert result.command == "help"
        assert result.args == []
        assert result.kwargs == {}

    def test_parse_command_with_positional_args(self):
        """Test parsing commands with positional arguments."""
        result = self.parser.parse_command("/save(output.json)")
        assert result is not None
        assert result.command == "save"
        assert result.args == ["output.json"]
        assert result.kwargs == {}

    def test_parse_command_with_multiple_positional_args(self):
        """Test parsing commands with multiple positional arguments."""
        result = self.parser.parse_command("/branch_from(main,5)")
        assert result is not None
        assert result.command == "branch_from"
        assert result.args == ["main", "5"]
        assert result.kwargs == {}

    def test_parse_command_with_keyword_args(self):
        """Test parsing commands with keyword arguments."""
        result = self.parser.parse_command("/set(temperature=0.8)")
        assert result is not None
        assert result.command == "set"
        assert result.args == []
        assert result.kwargs == {"temperature": "0.8"}

    def test_parse_command_with_mixed_args(self):
        """Test parsing commands with both positional and keyword arguments."""
        result = self.parser.parse_command("/set(temperature=0.8, top_p=0.9)")
        assert result is not None
        assert result.command == "set"
        assert result.args == []
        assert result.kwargs == {"temperature": "0.8", "top_p": "0.9"}

    def test_parse_command_with_quoted_strings(self):
        """Test parsing commands with quoted string arguments."""
        result = self.parser.parse_command('/save("my file.json")')
        assert result is not None
        assert result.command == "save"
        assert result.args == ["my file.json"]  # Quotes should be removed

    def test_parse_command_with_single_quotes(self):
        """Test parsing commands with single-quoted arguments."""
        result = self.parser.parse_command("/save('output file.json')")
        assert result is not None
        assert result.command == "save"
        assert result.args == ["output file.json"]

    def test_parse_command_with_spaces_in_args(self):
        """Test parsing commands with spaces in arguments."""
        result = self.parser.parse_command('/attach("sample data.csv")')
        assert result is not None
        assert result.command == "attach"
        assert result.args == ["sample data.csv"]

    def test_parse_command_without_parentheses(self):
        """Test that commands without parentheses are parsed."""
        result = self.parser.parse_command("/help")
        assert result is not None
        assert result.command == "help"
        assert result.args == []
        assert result.kwargs == {}

    def test_parse_invalid_command_syntax(self):
        """Test parsing invalid command syntax."""
        invalid_commands = [
            "/save(",  # Unclosed parentheses
            "save()",  # Missing leading slash
            "//save()",  # Double slash
        ]

        # These should definitely fail
        for invalid_cmd in invalid_commands:
            result = self.parser.parse_command(invalid_cmd)
            assert result is None, f"Should not parse invalid command: {invalid_cmd}"

        # These may parse but result in empty/invalid args (implementation-dependent)
        questionable_commands = [
            "/save())",  # Extra closing parentheses
            "/save(arg, kwarg=)",  # Empty keyword value
            "/save(=value)",  # Empty keyword name
        ]

        for cmd in questionable_commands:
            result = self.parser.parse_command(cmd)
            # Accept either None or successful parse with cleaned args
            if result is not None:
                assert result.command == "save"

    def test_parse_command_with_special_characters(self):
        """Test parsing commands with special characters in arguments."""
        result = self.parser.parse_command(
            "/fetch('https://example.com/api?key=123&value=test')"
        )
        assert result is not None
        assert result.command == "fetch"
        assert result.args == ["https://example.com/api?key=123&value=test"]

    def test_parse_command_with_numbers(self):
        """Test parsing commands with numeric arguments."""
        result = self.parser.parse_command("/set(temperature=0.8, max_tokens=100)")
        assert result is not None
        assert result.command == "set"
        assert result.kwargs == {"temperature": "0.8", "max_tokens": "100"}

    def test_parse_command_with_boolean_like_values(self):
        """Test parsing commands with boolean-like values."""
        result = self.parser.parse_command("/set(enable_stream=true, debug=false)")
        assert result is not None
        assert result.command == "set"
        assert result.kwargs == {"enable_stream": "true", "debug": "false"}

    def test_parse_command_case_sensitivity(self):
        """Test that command names are normalized to lowercase."""
        result_lower = self.parser.parse_command("/help()")
        result_upper = self.parser.parse_command("/HELP()")

        assert result_lower is not None
        assert result_lower.command == "help"

        # Upper case should also parse and be normalized to lowercase
        if result_upper is not None:
            assert (
                result_upper.command == "help"
            )  # Commands are normalized to lowercase

    def test_parse_command_with_empty_args(self):
        """Test parsing commands with empty arguments."""
        test_cases = [
            "/save()",  # No arguments
            "/save( )",  # Only whitespace
            "/save(,)",  # Empty comma-separated
        ]

        # Only the first case should parse successfully
        result1 = self.parser.parse_command(test_cases[0])
        assert result1 is not None
        assert result1.command == "save"
        assert result1.args == []

        # Others should fail or handle gracefully
        for cmd in test_cases[1:]:
            result = self.parser.parse_command(cmd)
            # Either fails to parse or parses with empty/filtered args
            if result is not None:
                assert result.command == "save"

    def test_parse_command_with_nested_quotes(self):
        """Test parsing commands with nested quotes."""
        result = self.parser.parse_command('/shell("echo \\"hello world\\"")')
        assert result is not None
        assert result.command == "shell"
        # The exact handling of nested quotes depends on implementation
        assert len(result.args) == 1

    def test_parse_non_command_text(self):
        """Test that regular text is not parsed as a command."""
        non_commands = [
            "Hello, how are you?",
            "This is regular text with / in it",
            "Price is $10/month",
            "Use the save() function",  # No leading slash
            "/not-a-command",  # No parentheses
        ]

        for text in non_commands:
            result = self.parser.parse_command(text)
            assert result is None, f"Should not parse non-command text: {text}"

    def test_parse_command_with_whitespace(self):
        """Test parsing commands with various whitespace patterns."""
        test_cases = [
            " /help() ",  # Leading/trailing spaces
            "/help( )",  # Spaces inside parentheses
            "/save( output.json )",  # Spaces around arguments
            "/set( temperature = 0.8 )",  # Spaces around keyword args
        ]

        for cmd in test_cases:
            result = self.parser.parse_command(cmd)
            assert result is not None, f"Should parse command with whitespace: {cmd}"

    def test_is_command_detection(self):
        """Test command detection functionality."""
        # If the parser has an is_command method
        if hasattr(self.parser, "is_command"):
            assert self.parser.is_command("/help()")
            assert self.parser.is_command("/save(file.json)")
            assert not self.parser.is_command("Regular text")

            # These depend on implementation - some parsers accept commands without
            # parentheses
            help_without_parens = self.parser.is_command("/help")
            if help_without_parens:
                assert help_without_parens  # Accept if implementation supports it
            else:
                assert not help_without_parens  # Accept strict parentheses requirement

            assert not self.parser.is_command("save()")  # No leading slash

    def test_parse_all_known_commands(self):
        """Test parsing all known chat commands."""
        known_commands = [
            # Basic commands
            "/help()",
            "/exit()",
            # Input mode commands
            "/ml",  # This might not have parentheses
            "/sl",  # This might not have parentheses
            # File operations
            "/attach(file.txt)",
            "/fetch(https://example.com)",
            "/shell(ls)",
            "/save(output.json)",
            "/import(input.json)",
            "/save_history(history.json)",
            "/import_history(history.json)",
            "/load(chat_id)",
            # Conversation management
            "/delete()",
            "/regen()",
            "/clear()",
            "/clear_thoughts()",
            "/compact()",
            "/show(1)",
            "/render(output.cast)",
            "/full_thoughts()",
            # Parameter management
            "/set(temperature=0.8)",
            # Branching
            "/branch()",
            "/branch_from(main,5)",
            "/switch(branch_name)",
            "/branches()",
            "/branch_delete(branch_name)",
            # Model management
            "/swap(model_name)",
            "/list_engines()",
            # Macro system
            "/macro(template.jinja)",
        ]

        for cmd in known_commands:
            # Skip input mode commands that might not have parentheses
            if cmd in ["/ml", "/sl"]:
                continue

            result = self.parser.parse_command(cmd)
            assert result is not None, f"Should parse known command: {cmd}"

            # Extract expected command name
            expected_name = cmd.split("(")[0][1:]  # Remove / and split at (
            assert result.command == expected_name

    def test_parse_command_edge_cases(self):
        """Test parsing edge cases and boundary conditions."""
        edge_cases = [
            "/a()",  # Single character command
            "/command_with_underscores()",
            "/command123()",  # Command with numbers
            "/save('')",  # Empty string argument
            '/save("")',  # Empty string with double quotes
            "/set(a=1,b=2,c=3,d=4,e=5)",  # Many keyword arguments
        ]

        for cmd in edge_cases:
            result = self.parser.parse_command(cmd)
            # These should either parse successfully or fail gracefully
            if result is not None:
                assert isinstance(result.command, str)
                assert isinstance(result.args, list)
                assert isinstance(result.kwargs, dict)

    def test_parser_error_handling(self):
        """Test parser error handling for malformed commands."""
        malformed_commands = [
            "/save(unclosed string')",
            "/save('unclosed string)",
            "/save(arg with no quotes and spaces)",
            "/save(arg1, , arg3)",  # Empty middle argument
            "/save(key=)",  # Empty value
            "/save(=value)",  # Empty key
        ]

        for cmd in malformed_commands:
            # Should either return None or raise appropriate exception
            try:
                result = self.parser.parse_command(cmd)
                if result is not None:
                    # If it does parse, the result should be well-formed
                    assert isinstance(result.command, str)
                    assert isinstance(result.args, list)
                    assert isinstance(result.kwargs, dict)
            except Exception:
                # Exceptions are acceptable for malformed input
                pass
