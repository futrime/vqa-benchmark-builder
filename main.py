import os

import datasets

from dataset_list import DATASET_METADATA_LIST
from dataset_metadata import DatasetMetadata


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


def main() -> None:
    dataset = datasets.concatenate_datasets(
        [load_dataset(metadata) for metadata in DATASET_METADATA_LIST],
    )

    assert isinstance(dataset, datasets.Dataset)

    os.makedirs("data/", exist_ok=True)

    dataset.to_csv("data/raw_qa.csv")


if __name__ == "__main__":
    main()
