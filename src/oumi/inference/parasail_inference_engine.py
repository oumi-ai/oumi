from oumi.inference.remote_inference_engine import RemoteInferenceEngine


class ParasailInferenceEngine(RemoteInferenceEngine):
    """Engine for running inference against the Parasail API."""

    base_url = "https://api.parasail.com/v1/chat/completions"
    """The base URL for the Parasail API."""

    api_key_env_varname = "PARASAIL_API_KEY"
    """The environment variable name for the Parasail API key."""
