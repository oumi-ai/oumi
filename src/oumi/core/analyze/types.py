"""Data types for analysis results."""

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class MessageAnalysisResult:
    """Result of analyzing a single message.

    Attributes:
        message_index: Index of the message within the conversation
        role: Role of the message (user, assistant, system, etc.)
        message_id: Unique identifier for the message
        text_content: The text content of the message
        analyzer_metrics: Dictionary containing analyzer metrics for this message
    """

    message_index: int
    role: str
    message_id: str
    text_content: str
    analyzer_metrics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert the analysis result to a dictionary.

        Returns:
            Dictionary representation of the analysis result
        """
        return asdict(self)


@dataclass
class ConversationAnalysisResult:
    """Result of analyzing a conversation as a whole.

    Attributes:
        analyzer_metrics: Dictionary containing analyzer metrics for the conversation
    """

    analyzer_metrics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert the analysis result to a dictionary.

        Returns:
            Dictionary representation of the analysis result
        """
        return asdict(self)


@dataclass
class SampleAnalysisResult:
    """Result of analyzing a single conversation sample.

    This class combines both message-level and conversation-level analysis results
    for a single conversation, making it easier to work with analyzer results.

    Attributes:
        conversation_id: Unique identifier for the conversation
        conversation_index: Index of the conversation in the dataset
        messages: List of analysis results for each individual message
        conversation: Analysis result for the conversation as a whole
    """

    conversation_id: str
    conversation_index: int
    messages: list[MessageAnalysisResult]
    conversation: ConversationAnalysisResult

    def to_dict(self) -> dict[str, Any]:
        """Convert the analysis result to a dictionary.

        Returns:
            Dictionary representation of the analysis result
        """
        return asdict(self)


@dataclass
class DatasetAnalysisResult:
    """Complete result of dataset analysis.

    Attributes:
        dataset_name: Name of the analyzed dataset
        total_conversations: Total number of conversations in the dataset
        conversations_analyzed: Number of conversations actually analyzed
        samples: List of analysis results for each conversation sample
    """

    dataset_name: str
    total_conversations: int
    conversations_analyzed: int
    samples: list[SampleAnalysisResult]

    def to_dict(self) -> dict[str, Any]:
        """Convert the analysis result to a dictionary.

        Returns:
            Dictionary representation of the analysis result
        """
        return asdict(self)
