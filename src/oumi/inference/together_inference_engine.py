from oumi.inference.remote_inference_engine import RemoteInferenceEngine


class TogetherInferenceEngine(RemoteInferenceEngine):
    """Engine for running inference against the Together AI API."""

    base_url = "https://api.together.xyz/v1/chat/completions"
    """The base URL for the Together API."""

    api_key_env_varname = "TOGETHER_API_KEY"
    """The environment variable name for the Together API key."""
