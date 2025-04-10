from dataclasses import dataclass, field

try:
    from nltk.tokenize import sent_tokenize  # pyright: ignore[reportMissingImports]
except ImportError:
    sent_tokenize = None


@dataclass
class Claim:
    """A claim extracted by HallOumi from the model's response with metadata."""
    claim_id: int = -1
    claim_string: str = ""
    subclaims: list[str] = field(default_factory=list)
    citations: list[int] = field(default_factory=list)
    rationale: str = ""
    supported: bool = True


class HallOumi:
    """HallOumi model for hallucination detection."""

    def __init__(self) -> None:
        """Initialize the HallOumi model."""
        if not sent_tokenize:
            raise RuntimeError(
                "The `nltk` package is NOT installed. Please either install `nltk` "
                "with `pip install nltk`. You also need to install the `punkt` "
                "sentence tokenizer with `python -m nltk.downloader punkt`."
            )

    def get_claims(self, context: str, request: str, response: str) -> list[Claim]:
        """Extracts claims from a context, model request, and model response."""
        raise NotImplementedError("HallOumi model is not implemented yet.")
