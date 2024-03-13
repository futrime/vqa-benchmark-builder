import os

import datasets


def load(output_file: str) -> None:
    docvqa_dataset = load_docvqa(output_file)
    infographicvqa_dataset = load_infographicvqa(output_file)

    dataset = datasets.concatenate_datasets([docvqa_dataset, infographicvqa_dataset])


def load_docvqa(output_file: str) -> datasets.Dataset:
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    dataset = datasets.load_dataset(
        path="lmms-lab/DocVQA",
        name="DocVQA",
        split="validation",
    )

    assert isinstance(dataset, datasets.Dataset)

    dataset = dataset.select_columns(["question", "answers"])
    dataset = dataset.rename_column("answers", "answer")

    return dataset


def load_infographicvqa(output_file: str) -> datasets.Dataset:
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    dataset = datasets.load_dataset(
        path="lmms-lab/DocVQA",
        name="InfographicVQA",
        split="validation",
    )

    assert isinstance(dataset, datasets.Dataset)

    dataset = dataset.select_columns(["question", "answers"])
    dataset = dataset.rename_column("answers", "answer")

    return dataset


if __name__ == "__main__":
    load("data/raw_qa/docvqa.csv")
