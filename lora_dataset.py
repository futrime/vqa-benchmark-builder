import json
import os
from typing import List, Tuple, TypedDict

import PIL.Image
import torch.utils.data


class LoRADataset(torch.utils.data.Dataset):
    def __init__(self, question_file_path: str, image_dir: str):
        super().__init__()

        with open(question_file_path, "r") as f:
            self._questions: List[Question] = json.load(f)

        self._image_dir = image_dir

    def __len__(self):
        return len(self._questions)

    def __getitem__(self, index: int) -> Tuple[PIL.Image.Image, "Question"]:
        question = self._questions[index]
        image_path = os.path.join(self._image_dir, question["image_id"])
        image = PIL.Image.open(image_path)

        return image, question


class Question(TypedDict):
    question_id: int
    question: str
    answer: str
    logical_operator: str
    logical_types: str
    logical_level: int
    lora_group: int
    image_id: str
    split: str
