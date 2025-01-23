import copy
from typing import Optional

from oumi.core.configs import ModelParams, RemoteParams
from oumi.inference.remote_inference_engine import RemoteInferenceEngine


class DeepSeekInferenceEngine(RemoteInferenceEngine):
    """Engine for running inference against the DeepSeek API.

    Documentation: https://api-docs.deepseek.com
    """

    def __init__(
        self, model_params: ModelParams, remote_params: Optional[RemoteParams] = None
    ):
        """Initializes the inference Engine.

        Args:
            model_params: The model parameters to use for inference.
            remote_params: Remote server params.
        """
        self._model = model_params.model_name

        if remote_params is None:
            self._remote_params = RemoteParams(
                api_url="https://api.deepseek.com/v1/chat/completions",
                api_key_env_varname="DEEPSEEK_API_KEY",
            )
        else:
            self._remote_params = copy.deepcopy(remote_params)
