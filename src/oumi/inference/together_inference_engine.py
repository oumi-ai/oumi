from typing import Optional

from typing_extensions import override

from oumi.inference.remote_inference_engine import RemoteInferenceEngine


class TogetherInferenceEngine(RemoteInferenceEngine):
    """Engine for running inference against the Together AI API."""

    @property
    @override
    def base_url(self) -> Optional[str]:
        """Return the default base URL for the Together API."""
        return "https://api.together.xyz/v1/chat/completions"

    @property
    @override
    def api_key_env_varname(self) -> Optional[str]:
        """Return the default environment variable name for the Together API key."""
        return "TOGETHER_API_KEY"
