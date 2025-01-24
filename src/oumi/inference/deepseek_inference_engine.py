from oumi.inference.remote_inference_engine import RemoteInferenceEngine


class DeepSeekInferenceEngine(RemoteInferenceEngine):
    """Engine for running inference against the DeepSeek API.

    Documentation: https://api-docs.deepseek.com
    """

    base_url = "https://api.deepseek.com/v1/chat/completions"
    """The base URL for the DeepSeek API."""

    api_key_env_varname = "DEEPSEEK_API_KEY"
    """The environment variable name for the DeepSeek API key."""
