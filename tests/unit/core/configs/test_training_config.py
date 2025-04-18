from oumi.core.configs import (
    DataParams,
    DatasetParams,
    DatasetSplitParams,
    ModelParams,
    TrainingConfig,
)


def test_training_config_processor_kwargs():
    """Test that json, regex, and choice parameters are mutually exclusive."""
    config = TrainingConfig(
        model=ModelParams(
            model_name="llava-hf/llava-1.5-7b-hf",
            processor_kwargs={"num_patches": 16},
        ),
        data=DataParams(
            train=DatasetSplitParams(
                collator_name="llava_collator",
                datasets=[
                    DatasetParams(
                        dataset_name="merve/vqav2-small",
                        split="train",
                        dataset_kwargs={"processor_name": "llava-hf/llava-1.5-7b-hf"},
                    ),
                    DatasetParams(
                        dataset_name="HuggingFaceH4/llava-instruct-mix-vsft",
                        split="train",
                        dataset_kwargs={
                            "processor_name": "llava-hf/llava-1.5-7b-hf",
                            "processor_kwargs": {"num_patches": 32, "foo": "bar"},
                        },
                    ),
                    DatasetParams(
                        dataset_name="coco_captions",
                        split="train",
                        dataset_kwargs={
                            "processor_name": "microsoft/Phi-3-vision-128k-instruct"
                        },
                    ),
                    DatasetParams(
                        dataset_name="coco_captions",
                        split="test",
                        dataset_kwargs={
                            "processor_name": "llava-hf/llava-1.5-7b-hf",
                            "processor_kwargs": {},
                        },
                    ),
                ],
            ),
            validation=DatasetSplitParams(
                collator_name="llava_collator",
                datasets=[
                    DatasetParams(
                        dataset_name="merve/vqav2-small",
                        split="validation",
                        dataset_kwargs={"processor_name": "llava-hf/llava-1.5-7b-hf"},
                    )
                ],
            ),
            test=DatasetSplitParams(
                collator_name="llava_collator",
                datasets=[
                    DatasetParams(
                        dataset_name="HuggingFaceH4/llava-instruct-mix-vsft",
                        split="test",
                        dataset_kwargs={"processor_name": "llava-hf/llava-1.5-7b-hf"},
                    )
                ],
            ),
        ),
    )
    assert len(config.data.train.datasets) == 4
    assert config.data.train.datasets[0] == DatasetParams(
        dataset_name="merve/vqav2-small",
        split="train",
        dataset_kwargs={
            "processor_name": "llava-hf/llava-1.5-7b-hf",
            "processor_kwargs": {
                "num_patches": 16,
            },
        },
    )
    assert config.data.train.datasets[1] == DatasetParams(
        dataset_name="HuggingFaceH4/llava-instruct-mix-vsft",
        split="train",
        dataset_kwargs={
            "processor_name": "llava-hf/llava-1.5-7b-hf",
            "processor_kwargs": {"num_patches": 32, "foo": "bar"},
        },
    )
    assert config.data.train.datasets[2] == DatasetParams(
        dataset_name="coco_captions",
        split="train",
        dataset_kwargs={"processor_name": "microsoft/Phi-3-vision-128k-instruct"},
    )
    assert config.data.train.datasets[3] == DatasetParams(
        dataset_name="coco_captions",
        split="test",
        dataset_kwargs={
            "processor_name": "llava-hf/llava-1.5-7b-hf",
            "processor_kwargs": {},
        },
    )

    assert len(config.data.validation.datasets) == 1
    assert config.data.validation.datasets[0] == DatasetParams(
        dataset_name="merve/vqav2-small",
        split="validation",
        dataset_kwargs={
            "processor_name": "llava-hf/llava-1.5-7b-hf",
            "processor_kwargs": {
                "num_patches": 16,
            },
        },
    )

    assert len(config.data.test.datasets) == 1
    assert config.data.test.datasets[0] == DatasetParams(
        dataset_name="HuggingFaceH4/llava-instruct-mix-vsft",
        split="test",
        dataset_kwargs={
            "processor_name": "llava-hf/llava-1.5-7b-hf",
            "processor_kwargs": {
                "num_patches": 16,
            },
        },
    )
