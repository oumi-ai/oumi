from oumi.core.datasets.base_dataset import BaseMapDataset

_PROMPT_KEY = "prompt"
_CHOSEN_KEY = "chosen"
_REJECTED_KEY = "rejected"

_ROLE = "role"
_CONTENT = "content"
_ASSISTANT = "assistant"


class BaseExperimentalDpoPreprocessor(BaseMapDataset):
    """Preprocess the samples to the Oumi format."""

    def __getitem__(self, index: int) -> dict:
        """Transform the samples to the Oumi format."""
        return self.transform_preference(self.dataset[index])

    def transform_preference(self, samples: dict) -> dict:
        """Transform the samples to the Oumi format."""
        prompt = samples[_PROMPT_KEY]
        chosen_chat = samples[_CHOSEN_KEY]
        rejected_chat = samples[_REJECTED_KEY]

        chosen_chat_response = self._extract_from_chat_format(chosen_chat)
        rejected_chat_response = self._extract_from_chat_format(rejected_chat)

        return {
            _PROMPT_KEY: prompt,
            _CHOSEN_KEY: chosen_chat_response,
            _REJECTED_KEY: rejected_chat_response,
        }

    def _extract_from_chat_format(self, sample: dict) -> str:
        """Extract the last 'assistant' turn in the chat."""
        for turn in sample[::-1]:
            if turn[_ROLE] == _ASSISTANT:
                return turn[_CONTENT]

        raise ValueError("No chat turn was found with an 'assistant' role.")
