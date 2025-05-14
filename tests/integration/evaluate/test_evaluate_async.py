import datetime
import tempfile

from oumi import evaluate_async
from oumi.core.configs import (
    AsyncEvaluationConfig,
    EvaluationConfig,
    EvaluationTaskParams,
    ModelParams,
)


def test_evaluate_async_polling_interval():
    with tempfile.TemporaryDirectory() as output_temp_dir:
        start_time = datetime.datetime.now()
        config: AsyncEvaluationConfig = AsyncEvaluationConfig(
            evaluation=EvaluationConfig(
                output_dir=output_temp_dir,
                tasks=[
                    EvaluationTaskParams(
                        evaluation_backend="lm_harness",
                        task_name="mmlu",
                        num_samples=4,
                    )
                ],
                model=ModelParams(
                    model_name="openai-community/gpt2",
                    trust_remote_code=True,
                ),
            ),
            polling_interval=0.5,
            num_retries=3,
            checkpoints_dir=output_temp_dir,
        )

        evaluate_async(config)
        end_time = datetime.datetime.now()
        # We should take more than 2 seconds (3 retries = 4 tries total)
        assert (end_time - start_time).seconds >= 2
        assert (end_time - start_time).seconds < 3
