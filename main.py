import os
from typing import List

import datasets
import pandas as pd
import tqdm

from config import DATASET_LIST, PATTERN_LIST
from dataset_metadata import DatasetMetadata
from pattern import Pattern


def load_datasets(metadata_list: List[DatasetMetadata]) -> pd.DataFrame:
    dataset_list: List[datasets.Dataset] = []

    for metadata in tqdm.tqdm(metadata_list, desc="Loading datasets"):
        dataset = datasets.load_dataset(
            path=metadata["path"],
            name=metadata["name"],
            split=metadata["split"],
        )

        assert isinstance(dataset, datasets.Dataset)

        for key, value in metadata["column_casting"].items():
            dataset = dataset.cast_column(key, value)

        dataset = dataset.select_columns(
            list(metadata["column_renaming"].keys()),
        ).rename_columns(
            metadata["column_renaming"],
        )

        dataset_list.append(dataset)

    dataset = datasets.concatenate_datasets(dataset_list)
    assert isinstance(dataset, datasets.Dataset)

    dataframe = dataset.to_pandas()
    assert isinstance(dataframe, pd.DataFrame)

    return dataframe


def analyze_tags(dataframe: pd.DataFrame, pattern_list: List[Pattern]) -> pd.DataFrame:
    dataframe["tags"] = [[] for _ in range(len(dataframe))]

    for _, row in tqdm.tqdm(
        dataframe.iterrows(), desc="Analyzing tags", total=len(dataframe)
    ):
        question = row["question"]
        answer = row["answer"]
        assert isinstance(question, str)
        assert isinstance(answer, str)

        for pattern in pattern_list:
            if pattern["matcher"](question, answer):
                tags = row["tags"]
                assert isinstance(tags, list)
                tags.append(pattern["name"])

    return dataframe


def main() -> None:
    dataset = load_datasets(DATASET_LIST)
    dataset = analyze_tags(dataset, PATTERN_LIST)

    os.makedirs("data/", exist_ok=True)
    dataset.to_csv("data/qa.csv")


if __name__ == "__main__":
    main()
