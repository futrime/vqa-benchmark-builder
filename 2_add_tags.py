from typing import List

import pandas as pd
import tqdm
import vllm

TAG_LIST: List[str] = [
    "color",
    "counting",
    "emotion",
    "object",
    "shape",
    "spatial relationship",
    "texture",
]

PROMPT_TEMPLATE = """
Please select the tags that best describe the question and answer pairs. \
You can select multiple tags by separating them with commas. \
Possible tags: {}
Question: {}
Answer: {}
"""


def main() -> None:
    qa_dataframe = pd.read_csv("data/qa.csv")

    llm = vllm.LLM("mistralai/Mistral-7B-Instruct-v0.2", max_model_len=1024)

    tag_list_str = ", ".join(TAG_LIST)

    for i, row in tqdm.tqdm(
        qa_dataframe.iterrows(), total=len(qa_dataframe), desc="Adding tags"
    ):
        question = row["question"]
        answer = row["answer"]
        assert isinstance(question, str)
        assert isinstance(answer, str)

        prompt = PROMPT_TEMPLATE.format(tag_list_str, question, answer)

        genetated_text = llm.generate(prompt, use_tqdm=False)[0].outputs[0].text

        tags = genetated_text.split(", ")

        qa_dataframe.at[i, "tags"] = tags

    qa_dataframe.to_csv("data/qat.csv")


if __name__ == "__main__":
    main()
