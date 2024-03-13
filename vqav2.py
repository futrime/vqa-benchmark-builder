import os

import datasets


def load(output_file: str) -> None:
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    dataset = datasets.load_dataset(
        path="lmms-lab/VQAv2",
        split="validation",
    )

    assert isinstance(dataset, datasets.Dataset)

    dataset = dataset.select_columns(["question", "multiple_choice_answer"])
    dataset = dataset.rename_column("multiple_choice_answer", "answer")

    dataset.to_csv(output_file)


if __name__ == "__main__":
    load("data/raw_qa/vqav2.csv")
