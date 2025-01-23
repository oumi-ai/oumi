from oumi.inference.remote_inference_engine import RemoteInferenceEngine


class OpenAIInferenceEngine(RemoteInferenceEngine):
    """Engine for running inference against the OpenAI API."""

    base_url = "https://api.openai.com/v1/chat/completions"
    """The base URL for the OpenAI API."""

    api_key_env_varname = "OPENAI_API_KEY"
    """The environment variable name for the OpenAI API key."""
