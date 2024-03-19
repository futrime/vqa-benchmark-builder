import os
import re
from typing import Dict

import datasets
import pandas as pd

from all_datasets import DATASET_LIST
from all_patterns import PATTERN_LIST
from dataset_metadata import DatasetMetadata
from pattern import Pattern


def load_dataset(metadata: DatasetMetadata) -> datasets.Dataset:
    dataset = datasets.load_dataset(
        path=metadata["path"],
        name=metadata["name"],
        split=metadata["split"],
    )

    assert isinstance(dataset, datasets.Dataset)

    return dataset.select_columns(
        list(metadata["column_mapping"].keys()),
    ).rename_columns(
        metadata["column_mapping"],
    )


def match_pattern(dataframe: pd.DataFrame, pattern: Pattern) -> pd.DataFrame:
    regex_common = re.compile("|".join(pattern["common"]), re.IGNORECASE)
    regex_question = re.compile("|".join(pattern["question"]), re.IGNORECASE)
    regex_answer = re.compile("|".join(pattern["answer"]), re.IGNORECASE)

    for index, row in dataframe.iterrows():
        question = row["question"]
        answer = row["answer"]
        assert isinstance(question, str)
        assert isinstance(answer, str)

        combined = f"{question} {answer}"

        if (
            regex_question.search(question)
            or regex_answer.search(answer)
            or regex_common.search(combined)
        ):
            patterns = row["patterns"]
            assert isinstance(patterns, list)
            patterns.append(pattern["name"])

    return dataframe


def main() -> None:
    raw_dataset = datasets.concatenate_datasets(
        [load_dataset(metadata) for metadata in DATASET_LIST],
    )

    dataset = raw_dataset.to_pandas()
    assert isinstance(dataset, pd.DataFrame)

    # Add patterns column
    dataset["patterns"] = [[] for _ in range(len(dataset))]

    for pattern in PATTERN_LIST:
        dataset = match_pattern(dataset, pattern)

    os.makedirs("data/", exist_ok=True)
    dataset.to_csv("data/qa.csv")


if __name__ == "__main__":
    main()
