import json

import tqdm
from datasets import load_dataset

from oumi.core.types.conversation import Conversation, Message, Role

ds = load_dataset("open-thoughts/OpenThoughts3-1.2M", split="train")
conversation_dataset = []

for sample in tqdm.tqdm(ds):
    sample = sample["conversations"]
    conversation = Conversation(
        messages=[
            Message(role=Role.USER, content=sample[0]["value"]),
            Message(
                role=Role.ASSISTANT,
                content=sample[1]["value"],
            ),
        ]
    ).to_dict()
    conversation_dataset.append(conversation)

# save to jsonl
json.dump(conversation_dataset, open("open_thoughts_train.json", "w"), indent=4)
