import json
import os
from typing import List, Tuple, TypedDict

import PIL.Image
import torch.utils.data


class LoRADataset(torch.utils.data.Dataset):
    def __init__(self, question_file_path: str, image_dir: str):
        super().__init__()

        with open(question_file_path, "r") as f:
            self._metadata_list: List[Metadata] = json.load(f)

        self._image_dir = image_dir

    def __len__(self):
        return len(self._metadata_list)

    def __getitem__(self, index: int) -> Tuple[PIL.Image.Image, str, str]:
        metadata = self._metadata_list[index]

        image_path = os.path.join(self._image_dir, metadata["image_id"])
        image = PIL.Image.open(image_path)

        return image, metadata["question"], metadata["answer"]


class Metadata(TypedDict):
    question_id: int
    question: str
    answer: str
    logical_operator: str
    logical_types: str
    logical_level: int
    lora_group: int
    image_id: str
    split: str
