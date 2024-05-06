import json
from typing import List, TypedDict

import tqdm

TRAIN_METADATA_FILE = "data/dataset/train.json"
OUTPUT_SFT_FILE = "data/dataset/sft.json"


class QaEntry(TypedDict):
    id: int
    image_id: int
    question: str
    steps: List[str]
    answer: str


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
    with open(TRAIN_METADATA_FILE, "r") as f:
        train_qa_entries: List[QaEntry] = json.load(f)

    sft_entries: List[SftEntry] = []
    for qa_entry in tqdm.tqdm(train_qa_entries):
        id = str(qa_entry["id"])
        image = f"{qa_entry['image_id']}.png"

        gpt_output = ""
        for step_num, step in enumerate(qa_entry["steps"]):
            gpt_output += f"{step_num + 1}. {step}\n"

        gpt_output += f"<answer>{qa_entry['answer']}</answer>"

        conversations: List[ConversationDescriptor] = [
            {
                "from": "human",
                "value": "<image>\n" + qa_entry["question"],
            },
            {
                "from": "gpt",
                "value": gpt_output,
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
