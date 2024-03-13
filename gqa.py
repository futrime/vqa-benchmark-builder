import os

import datasets


def load(output_file: str) -> None:
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    dataset = datasets.load_dataset(
        path="lmms-lab/GQA",
        name="train_all_instructions",
        split="train",
    )

    assert isinstance(dataset, datasets.Dataset)

    dataset = dataset.select_columns(["question", "fullAnswer"])
    dataset = dataset.rename_column("fullAnswer", "answer")

    dataset.to_csv(output_file)


if __name__ == "__main__":
    load("data/raw_qa/gqa.csv")
