import os

import datasets


def load(output_file: str) -> None:
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    dataset = datasets.load_dataset(
        path="lmms-lab/textvqa",
        split="validation",
    )

    assert isinstance(dataset, datasets.Dataset)

    dataset = dataset.select_columns(["question", "answers"])
    dataset = dataset.rename_column("answers", "answer")

    dataset.to_csv(output_file)


if __name__ == "__main__":
    load("data/raw_qa/textvqa.csv")
