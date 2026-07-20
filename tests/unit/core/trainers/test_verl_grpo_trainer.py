from unittest.mock import MagicMock, patch

import pytest

from oumi.core.trainers.verl_grpo_trainer import VerlGrpoTrainer
from oumi.core.types.conversation import (
    ContentItem,
    Conversation,
    Message,
    Role,
    Type,
)

try:
    verl_import_failed = False
    import verl  # pyright: ignore[reportMissingImports]  # noqa: F401
except ModuleNotFoundError:
    verl_import_failed = True


def _example(conversation: Conversation) -> dict:
    return {"conversation_json": conversation.to_json()}


@pytest.mark.skipif(verl_import_failed, reason="verl not available")
def test_init_without_verl():
    with patch("oumi.core.trainers.verl_grpo_trainer.verl", None):
        with pytest.raises(RuntimeError, match="verl is not installed"):
            VerlGrpoTrainer(
                processing_class=MagicMock(),
                config=MagicMock(),
                reward_funcs=[MagicMock()],
                train_dataset=MagicMock(),
                eval_dataset=MagicMock(),
            )


def test_create_verl_data_entry_single_turn():
    convo = Conversation(
        messages=[
            Message(role=Role.USER, content="What is 2+2?"),
            Message(role=Role.ASSISTANT, content="4"),
        ]
    )
    entry = VerlGrpoTrainer._create_verl_data_entry_from_conversation(
        _example(convo), 7, "my_dataset", "train"
    )
    assert entry["data_source"] == "my_dataset"
    assert entry["prompt"] == [{"role": "user", "content": "What is 2+2?"}]
    assert entry["images"] == []
    assert entry["reward_model"] == {"style": "rule", "ground_truth": "4"}
    assert entry["extra_info"]["split"] == "train"
    assert entry["extra_info"]["index"] == 7
    assert entry["extra_info"]["answer"] == "4"


def test_create_verl_data_entry_multi_turn():
    convo = Conversation(
        messages=[
            Message(role=Role.SYSTEM, content="You are helpful."),
            Message(role=Role.USER, content="Hi"),
            Message(role=Role.ASSISTANT, content="Hello!"),
            Message(role=Role.USER, content="What is 2+2?"),
            Message(role=Role.ASSISTANT, content="4"),
        ]
    )
    entry = VerlGrpoTrainer._create_verl_data_entry_from_conversation(
        _example(convo), 0, "my_dataset", "validation"
    )
    assert entry["prompt"] == [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
        {"role": "user", "content": "What is 2+2?"},
    ]
    assert entry["images"] == []
    assert entry["reward_model"]["ground_truth"] == "4"


def test_create_verl_data_entry_single_turn_image_prepends_marker():
    convo = Conversation(
        messages=[
            Message(
                role=Role.USER,
                content=[
                    ContentItem(type=Type.IMAGE_BINARY, binary=b"imgbytes"),
                    ContentItem(type=Type.TEXT, content="Describe this."),
                ],
            ),
            Message(role=Role.ASSISTANT, content="A cat."),
        ]
    )
    entry = VerlGrpoTrainer._create_verl_data_entry_from_conversation(
        _example(convo), 0, "my_dataset", "train"
    )
    assert entry["prompt"] == [{"role": "user", "content": "<image>Describe this."}]
    assert entry["images"] == [{"bytes": b"imgbytes"}]


def test_create_verl_data_entry_single_turn_image_with_system():
    convo = Conversation(
        messages=[
            Message(role=Role.SYSTEM, content="You are helpful."),
            Message(
                role=Role.USER,
                content=[
                    ContentItem(type=Type.IMAGE_BINARY, binary=b"imgbytes"),
                    ContentItem(type=Type.TEXT, content="Describe this."),
                ],
            ),
            Message(role=Role.ASSISTANT, content="A cat."),
        ]
    )
    entry = VerlGrpoTrainer._create_verl_data_entry_from_conversation(
        _example(convo), 0, "my_dataset", "train"
    )
    assert entry["prompt"] == [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "<image>Describe this."},
    ]
    assert entry["images"] == [{"bytes": b"imgbytes"}]


def test_create_verl_data_entry_multi_turn_with_images_raises():
    convo = Conversation(
        messages=[
            Message(
                role=Role.USER,
                content=[
                    ContentItem(type=Type.IMAGE_BINARY, binary=b"imgbytes"),
                    ContentItem(type=Type.TEXT, content="Describe this."),
                ],
            ),
            Message(role=Role.ASSISTANT, content="A cat."),
            Message(role=Role.USER, content="And this one?"),
            Message(role=Role.ASSISTANT, content="A dog."),
        ]
    )
    with pytest.raises(ValueError, match="multi-turn"):
        VerlGrpoTrainer._create_verl_data_entry_from_conversation(
            _example(convo), 0, "my_dataset", "train"
        )


@pytest.mark.skipif(verl_import_failed, reason="verl not available")
def test_init_with_multiple_reward_funcs():
    with pytest.raises(ValueError, match="We only support up to one reward function"):
        VerlGrpoTrainer(
            processing_class=MagicMock(),
            config=MagicMock(),
            reward_funcs=[MagicMock(), MagicMock()],
            train_dataset=MagicMock(),
            eval_dataset=MagicMock(),
        )
