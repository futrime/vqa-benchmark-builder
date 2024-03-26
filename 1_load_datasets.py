import os
from typing import List

import datasets
import pandas as pd
import tqdm

from dataset_converters import *
from dataset_metadata import DatasetMetadata

DATASET_METADATA_LIST: List[DatasetMetadata] = [
    {
        "path": "lmms-lab/ChartQA",
        "name": None,
        "split": "test",
        "converter": make_convert_simple_dataset(),
    },
    {
        "path": "lmms-lab/DocVQA",
        "name": "DocVQA",
        "split": "validation",
        "converter": make_convert_multiple_answers_dataset(),
    },
    {
        "path": "lmms-lab/DocVQA",
        "name": "InfographicVQA",
        "split": "validation",
        "converter": make_convert_multiple_answers_dataset(),
    },
    {
        "path": "lmms-lab/GQA",
        "name": "train_all_instructions",
        "split": "train",
        "converter": make_convert_simple_dataset(answer_column="fullAnswer"),
    },
    {
        "path": "lmms-lab/llava-bench-coco",
        "name": None,
        "split": "train",
        "converter": make_convert_simple_dataset(),
    },
    {
        "path": "lmms-lab/llava-bench-in-the-wild",
        "name": None,
        "split": "train",
        "converter": make_convert_simple_dataset(answer_column="gpt_answer"),
    },
    {
        "path": "lmms-lab/MP-DocVQA",
        "name": None,
        "split": "val",
        "converter": make_convert_multiple_answers_dataset(),
    },
    {
        "path": "lmms-lab/OK-VQA",
        "name": None,
        "split": "val2014",
        "converter": make_convert_multiple_answers_dataset(),
    },
    {
        "path": "flaviagiammarino/path-vqa",
        "name": None,
        "split": "train",
        "converter": make_convert_simple_dataset(),
    },
    {
        "path": "lmms-lab/textvqa",
        "name": None,
        "split": "validation",
        "converter": make_convert_simple_dataset(),
    },
    {
        "path": "ruanchaves/visual7w-gpt",
        "name": None,
        "split": "train",
        "converter": make_convert_simple_dataset(),
    },
    {
        "path": "lmms-lab/VizWiz-VQA",
        "name": None,
        "split": "val",
        "converter": make_convert_simple_dataset(),
    },
    {
        "path": "flaviagiammarino/vqa-rad",
        "name": None,
        "split": "train",
        "converter": make_convert_simple_dataset(),
    },
    {
        "path": "lmms-lab/VQAv2",
        "name": None,
        "split": "validation",
        "converter": make_convert_simple_dataset(
            answer_column="multiple_choice_answer"
        ),
    },
]


def main() -> None:
    os.makedirs("data/", exist_ok=True)

    dataframe_list: List[pd.DataFrame] = []

    for metadata in tqdm.tqdm(DATASET_METADATA_LIST, desc="Loading datasets"):
        dataset = datasets.load_dataset(
            path=metadata["path"],
            name=metadata["name"],
            split=metadata["split"],
        )
        assert isinstance(dataset, datasets.Dataset)

        dataframe = metadata["converter"](dataset)

        dataframe_list.append(dataframe)

    concatenated_dataframe = pd.concat(dataframe_list, ignore_index=True)

    concatenated_dataframe.to_csv("data/qa.csv")


if __name__ == "__main__":
    main()
