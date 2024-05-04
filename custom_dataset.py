import json
import os
from typing import List, TypedDict

import PIL.Image
import torch.utils.data


class CustomDataset(torch.utils.data.Dataset):
    class Entry(TypedDict):
        image: PIL.Image.Image
        question: str
        steps: List[str]
        answer: str

    class _QaEntry(TypedDict):
        id: int
        image_id: int
        question: str
        steps: List[str]
        answer: str

    def __init__(self, qa_file_path: str, image_dir: str):
        super().__init__()

        with open(qa_file_path, "r") as f:
            self._qa_entries: List[CustomDataset._QaEntry] = json.load(f)

        self._image_dir = image_dir

    def __len__(self):
        return len(self._qa_entries)

    def __getitem__(self, index: int) -> Entry:
        qa_entry = self._qa_entries[index]

        image_path = os.path.join(self._image_dir, f"{qa_entry['image_id']}.png")
        image = PIL.Image.open(image_path)

        return {
            "image": image,
            "question": qa_entry["question"],
            "steps": qa_entry["steps"],
            "answer": qa_entry["answer"],
        }
