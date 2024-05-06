import json
from typing import List, TypedDict

import torch.utils.data


class CustomDataset(torch.utils.data.Dataset):
    class Entry(TypedDict):
        id: int
        image_id: int
        question: str
        steps: List[str]
        answer: str

    def __init__(self, metadata_file_path: str):
        super().__init__()

        with open(metadata_file_path, "r") as f:
            self._qa_entries: List[CustomDataset.Entry] = json.load(f)

    def __iter__(self):
        for qa_entry in self._qa_entries:
            result: CustomDataset.Entry = {
                "id": qa_entry["id"],
                "image_id": qa_entry["image_id"],
                "question": qa_entry["question"],
                "steps": qa_entry["steps"],
                "answer": qa_entry["answer"],
            }

            yield result

    def __len__(self):
        return len(self._qa_entries)

    def __getitem__(self, qa_id: int) -> Entry:
        # Find the QA entry with the given ID
        qa_entry = next(
            (entry for entry in self._qa_entries if entry["id"] == qa_id), None
        )

        if qa_entry is None:
            raise KeyError(f"QA entry with ID {qa_id} not found")

        return {
            "id": qa_entry["id"],
            "image_id": qa_entry["image_id"],
            "question": qa_entry["question"],
            "steps": qa_entry["steps"],
            "answer": qa_entry["answer"],
        }
