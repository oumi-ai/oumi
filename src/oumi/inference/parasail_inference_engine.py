from typing import Optional

from typing_extensions import override

from oumi.inference.remote_inference_engine import RemoteInferenceEngine


class ParasailInferenceEngine(RemoteInferenceEngine):
    """Engine for running inference against the Parasail API."""

    @property
    @override
    def base_url(self) -> Optional[str]:
        """Return the default base URL for the Parasail API."""
        return "https://api.parasail.com/v1/chat/completions"

    @property
    @override
    def api_key_env_varname(self) -> Optional[str]:
        """Return the default environment variable name for the Parasail API key."""
        return "PARASAIL_API_KEY"
