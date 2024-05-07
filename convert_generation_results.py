import json
import os
from typing import List, TypedDict

import tqdm

from custom_dataset import CustomDataset

GENERATION_RESULTS_FILE = "data/dataset/generation_results.json"
VAL_METADATA_FILE = "./data/dataset/val.json"
OUTPUT_SFT_FILE = "data/dataset/sft_verifier.json"


class GeneratedResultEntry(TypedDict):
    id: int
    qa_id: int
    predicted: str
    correctness: bool


class SftEntry(TypedDict):
    id: str
    image: str
    conversations: List["ConversationDescriptor"]


class ConversationDescriptor(
    TypedDict(
        "ConversationDescriptor",
        {
            "from": str,
            "value": str,
        },
    )
):
    pass


def main():
    os.makedirs(os.path.dirname(OUTPUT_SFT_FILE), exist_ok=True)

    with open(GENERATION_RESULTS_FILE, "r") as f:
        generated_results: List[GeneratedResultEntry] = json.load(f)

    dataset = CustomDataset(metadata_file_path=VAL_METADATA_FILE)

    sft_entries: List[SftEntry] = []
    for generated_result in tqdm.tqdm(generated_results):
        qa = dataset[generated_result["qa_id"]]

        id = str(generated_result["id"])
        image = f"{qa['image_id']}.png"

        prediction = generated_result["predicted"]
        prediction = prediction.removeprefix("<s>")
        prediction = prediction.removesuffix("</s>")

        conversations: List[ConversationDescriptor] = [
            {
                "from": "human",
                "value": f"<image>\n{qa['question']}\n{prediction}",
            },
            {
                "from": "gpt",
                "value": "yes" if generated_result["correctness"] else "no",
            },
        ]

        sft_entries.append(
            {
                "id": id,
                "image": image,
                "conversations": conversations,
            }
        )

    with open(OUTPUT_SFT_FILE, "w") as f:
        json.dump(sft_entries, f, indent=4)


if __name__ == "__main__":
    main()
