import json

from oumi.core.configs import InferenceConfig
from oumi.core.types.conversation import Conversation, Message, Role
from oumi.inference import VLLMInferenceEngine


def main():
    config = InferenceConfig.from_yaml("infer_qwen3_32b.yaml")
    engine = VLLMInferenceEngine(
        model_params=config.model,
        generation_params=config.generation,
    )

    data = json.load(open("phishing_final.json"))

    for sample in data[:10]:
        sample = sample["messages"]
        conversation = Conversation(
            messages=[
                Message(
                    role=Role.SYSTEM,
                    content="Given a piece of text and its final classification (spam or not spam), explain the reasoning behind the classification. Your explanation should highlight specific features, words, or patterns in the text that contributed to the classification decision. Consider both linguistic cues (e.g., tone, grammar, urgency) and content features (e.g., presence of links, offers, suspicious phrases).",
                ),
                Message(
                    role=Role.USER,
                    content=f"Text: {sample[1]['content']}",
                ),
                Message(
                    role=Role.USER,
                    content=f"Final classification: {sample[2]['content']}",
                ),
            ]
        )
        output_conversations = engine.infer(
            input=[conversation], inference_config=config
        )
        print(f"Model output: {output_conversations[0].messages[-1].content}")
        print("--------------------------------")


if __name__ == "__main__":
    main()
