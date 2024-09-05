import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    BlipForConditionalGeneration,
    CLIPModel,
    ViltModel,
)

from lema.datasets.vision_language import (
    COCOCaptionsDataset,
    Flickr30kDataset,
    VQAv2Dataset,
)


def run_model(model, dataloader, device, num_samples=5):  # noqa: D103
    for batch in tqdm(dataloader, total=num_samples):
        images = batch["image"].to(device)
        text = batch["text"]

        input_ids = text["input_ids"].to(device)
        attention_mask = text["attention_mask"].to(device)

        with torch.no_grad():
            output_ids = model.generate(
                images,
                input_ids,
                attention_mask,
                max_length=50,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
            )

        for i, ids in enumerate(output_ids):
            generated_text = model.tokenizer.decode(ids, skip_special_tokens=True)
            print(f"Sample {i + 1}:")
            print(f"Generated text: {generated_text}")
            print()

        if num_samples <= 0:
            break
        num_samples -= 1


def main():  # noqa: D103
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = [
        ("CLIP", "openai/clip-vit-base-patch32", CLIPModel),
        ("ViLT", "dandelin/vilt-b32-mlm", ViltModel),
        (
            "BLIP",
            "Salesforce/blip-image-captioning-base",
            BlipForConditionalGeneration,
        ),
    ]

    datasets = [
        ("COCO Captions", COCOCaptionsDataset, "train"),
        ("Flickr30k", Flickr30kDataset, "test"),
        ("VQA v2", VQAv2Dataset, "train"),
    ]

    for model_name, model_cls in models:
        model = model_cls()
        model.eval()
        model.to(device)
        print(f"\nTesting {model_name} model:")

        for dataset_name, dataset_class, split in datasets:
            print(f"\nTesting {dataset_name} dataset:")
            dataset = dataset_class(split=split, processor=model.processor)

            dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
            run_model(model, dataloader, device)


if __name__ == "__main__":
    main()
