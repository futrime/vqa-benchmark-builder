import os

import datasets


def load(output_file: str) -> None:
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    dataset = datasets.load_dataset(
        path="lmms-lab/llava-bench-in-the-wild",
        split="train",
    )

    assert isinstance(dataset, datasets.Dataset)

    dataset = dataset.select_columns(["question", "gpt_answer"])
    dataset = dataset.rename_column("gpt_answer", "answer")

    dataset.to_csv(output_file)


if __name__ == "__main__":
    load("data/raw_qa/llava_bench_in_the_wild.csv")
