from oumi.core.configs import TunerType, TuningParams
from oumi.core.tuners import BaseTuner, OptunaTuner


def build_tuner(tuning_params: TuningParams) -> BaseTuner:
    """Build a tuner based on the configuration.

    Args:
        tuning_params: Tuning configuration parameters.

    Returns:
        An instance of the appropriate tuner implementation.

    Raises:
        NotImplementedError: If the tuner type is not supported.
    """
    if tuning_params.tuner_type == TunerType.OPTUNA:
        return OptunaTuner(tuning_params)

    raise NotImplementedError(f"Tuner type {tuning_params.tuner_type} not supported.")
