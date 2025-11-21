"""Pluggable progress parsers for different oumi verbs."""

import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from oumi.workflow.job import JobMetrics


class ProgressParser(ABC):
    """Base class for progress parsers."""

    @abstractmethod
    def parse_line(self, line: str, metrics: "JobMetrics") -> bool:
        """Parse a line of output and update metrics.

        Args:
            line: Line of output to parse
            metrics: Metrics object to update

        Returns:
            True if line was parsed successfully, False otherwise
        """
        pass

    @abstractmethod
    def supports_verb(self, verb: str) -> bool:
        """Check if this parser supports the given verb.

        Args:
            verb: Oumi verb name (train, eval, etc.)

        Returns:
            True if parser supports this verb
        """
        pass


class TrainingProgressParser(ProgressParser):
    """Parser for training job progress."""

    def supports_verb(self, verb: str) -> bool:
        """Support train and tune verbs."""
        return verb in ("train", "tune")

    def parse_line(self, line: str, metrics: "JobMetrics") -> bool:
        """Parse training progress from line.

        Looks for patterns like:
        - Step X/Y
        - Epoch X
        - Loss: X.XX
        - Learning rate: X.XXe-X
        """
        parsed = False

        # Step progress (e.g., "Step 100/1000" or "step: 100")
        step_match = re.search(
            r"(?:Step|step|STEP)[:\s]+(\d+)(?:[/\s]+(?:of\s+)?(\d+))?", line
        )
        if step_match:
            metrics.current_step = int(step_match.group(1))
            if step_match.group(2):
                metrics.total_steps = int(step_match.group(2))
                metrics.progress_percent = (
                    metrics.current_step / metrics.total_steps * 100
                )
            parsed = True

        # Epoch progress
        epoch_match = re.search(r"(?:Epoch|epoch|EPOCH)[:\s]+(\d+)", line)
        if epoch_match:
            metrics.epoch = int(epoch_match.group(1))
            parsed = True

        # Loss (various formats)
        loss_match = re.search(
            r"(?:loss|Loss|LOSS|train_loss|training_loss)[:\s=]+([0-9.]+)", line
        )
        if loss_match:
            metrics.loss = float(loss_match.group(1))
            parsed = True

        # Learning rate
        lr_match = re.search(
            r"(?:lr|LR|learning_rate|learning-rate)[:\s=]+([0-9.e\-+]+)", line
        )
        if lr_match:
            metrics.learning_rate = float(lr_match.group(1))
            parsed = True

        return parsed


class EvaluationProgressParser(ProgressParser):
    """Parser for evaluation job progress."""

    def supports_verb(self, verb: str) -> bool:
        """Support evaluate and eval verbs."""
        return verb in ("evaluate", "eval")

    def parse_line(self, line: str, metrics: "JobMetrics") -> bool:
        """Parse evaluation progress from line.

        Looks for patterns like:
        - Evaluating X/Y
        - Accuracy: X.XX
        - Score: X.XX
        """
        parsed = False

        # Evaluation progress
        eval_match = re.search(
            r"(?:Evaluating|Processing|Sample)[:\s]+(\d+)(?:/(\d+))?", line
        )
        if eval_match:
            metrics.current_step = int(eval_match.group(1))
            if eval_match.group(2):
                metrics.total_steps = int(eval_match.group(2))
                metrics.progress_percent = (
                    metrics.current_step / metrics.total_steps * 100
                )
            parsed = True

        # Accuracy/Score metrics
        metric_match = re.search(
            r"(?:accuracy|Accuracy|score|Score|metric)[:\s=]+([0-9.]+)", line
        )
        if metric_match:
            # Store in custom metrics
            metrics.custom["score"] = float(metric_match.group(1))
            parsed = True

        return parsed


class InferenceProgressParser(ProgressParser):
    """Parser for inference job progress."""

    def supports_verb(self, verb: str) -> bool:
        """Support infer verb."""
        return verb == "infer"

    def parse_line(self, line: str, metrics: "JobMetrics") -> bool:
        """Parse inference progress from line.

        Looks for patterns like:
        - Generated X/Y
        - Processing sample X
        """
        parsed = False

        # Generation progress
        gen_match = re.search(
            r"(?:Generated|Processing|Sample)[:\s]+(\d+)(?:/(\d+))?", line
        )
        if gen_match:
            metrics.current_step = int(gen_match.group(1))
            if gen_match.group(2):
                metrics.total_steps = int(gen_match.group(2))
                metrics.progress_percent = (
                    metrics.current_step / metrics.total_steps * 100
                )
            parsed = True

        return parsed


class GenericProgressParser(ProgressParser):
    """Generic fallback parser for any verb."""

    def supports_verb(self, verb: str) -> bool:
        """Support all verbs."""
        return True

    def parse_line(self, line: str, metrics: "JobMetrics") -> bool:
        """Parse generic progress patterns."""
        parsed = False

        # Generic progress indicators
        patterns = [
            r"(\d+)/(\d+)",  # X/Y format
            r"(\d+)%",  # X% format
            r"Progress[:\s]+([0-9.]+)",  # Progress: X
        ]

        for pattern in patterns:
            match = re.search(pattern, line)
            if match:
                if len(match.groups()) == 2:
                    # X/Y format
                    current = int(match.group(1))
                    total = int(match.group(2))
                    metrics.current_step = current
                    metrics.total_steps = total
                    metrics.progress_percent = (
                        (current / total * 100) if total > 0 else 0
                    )
                    parsed = True
                    break
                elif "%" in pattern:
                    # X% format
                    metrics.progress_percent = float(match.group(1))
                    parsed = True
                    break

        return parsed


class ProgressParserRegistry:
    """Registry for progress parsers."""

    def __init__(self):
        """Initialize registry with default parsers."""
        self._parsers: list[ProgressParser] = [
            TrainingProgressParser(),
            EvaluationProgressParser(),
            InferenceProgressParser(),
            GenericProgressParser(),  # Fallback
        ]

    def register(self, parser: ProgressParser) -> None:
        """Register a new parser.

        Args:
            parser: Parser to register
        """
        # Insert before generic parser
        self._parsers.insert(-1, parser)

    def get_parser(self, verb: str) -> ProgressParser:
        """Get appropriate parser for verb.

        Args:
            verb: Oumi verb name

        Returns:
            Parser that supports this verb (fallback to generic)
        """
        for parser in self._parsers:
            if parser.supports_verb(verb):
                return parser
        # Should never reach here due to GenericProgressParser
        return self._parsers[-1]

    def parse_line(self, verb: str, line: str, metrics: "JobMetrics") -> bool:
        """Parse a line using appropriate parser.

        Args:
            verb: Oumi verb name
            line: Line to parse
            metrics: Metrics to update

        Returns:
            True if line was parsed successfully
        """
        parser = self.get_parser(verb)
        return parser.parse_line(line, metrics)


# Global registry instance
_registry = ProgressParserRegistry()


def get_parser_registry() -> ProgressParserRegistry:
    """Get the global parser registry.

    Returns:
        Global ProgressParserRegistry instance
    """
    return _registry


def register_parser(parser: ProgressParser) -> None:
    """Register a custom progress parser.

    Args:
        parser: Parser to register
    """
    _registry.register(parser)
