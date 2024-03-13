import os

import datasets


def load(output_file: str) -> None:
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    dataset = datasets.load_dataset(
        path="lmms-lab/MP-DocVQA",
        split="val",
    )

    assert isinstance(dataset, datasets.Dataset)

    dataset = dataset.select_columns(["question", "answers"])
    dataset = dataset.rename_column("answers", "answer")

    dataset.to_csv(output_file)


if __name__ == "__main__":
    load("data/raw_qa/mp_docvqa.csv")
