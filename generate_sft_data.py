import json
import os
from typing import List, TypedDict

IMAGE_DIR = "data/custom/images/"
QA_FILE = "data/custom/qa.json"
OUTPUT_SFT_DATA_FILE = "data/custom/sft_data.json"


class Entry(TypedDict):
    instruction: str
    output: str
    images: List[str]


class QaEntry(TypedDict):
    image_id: int
    question: str
    steps: List[str]
    answer: str


def main():
    with open(QA_FILE, "r") as f:
        qa_entries: List[QaEntry] = json.load(f)

    sft_data: List[Entry] = []
    for qa_entry in qa_entries:
        image_id = qa_entry["image_id"]
        question = qa_entry["question"]
        steps = qa_entry["steps"]
        answer = qa_entry["answer"]

        instruction = f"{question}"
        output = ""
        for step_num, step in enumerate(steps):
            output += f"{step_num + 1}. {step}\n"
        output += f"<answer>{answer}</answer>"
        images = [os.path.join(IMAGE_DIR, f"{image_id}.png")]

        sft_data.append(
            {
                "instruction": instruction,
                "output": output,
                "images": images,
            }
        )

    with open(OUTPUT_SFT_DATA_FILE, "w") as f:
        json.dump(sft_data, f, indent=4)


if __name__ == "__main__":
    main()
