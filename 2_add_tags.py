import logging
import re
from typing import List

import pandas as pd
import tqdm
import vllm

POSSIBLE_TAG_LIST: List[str] = [
    "color",
    "counting",
    "emotion",
    "object",
    "shape",
    "spatial_relationship",
    "texture",
]

PROMPT_TEMPLATE = """
Please select the tags may describe the content. \
Leave blank if none of the tags are suitable. \
Only output the answer, do not explain and reason.
<content>
{} {}
</content>
<available_tags>
{}
</available_tags>

You should follow this answer format:
<answer>
tag1,tag2,tag3,...
</answer>
"""

ANSWER_MATCH_REGEX = r"<answer>(.*?)</answer>"


def main() -> None:
    possible_tag_list_str = ",".join(POSSIBLE_TAG_LIST)
    answer_matcher = re.compile(ANSWER_MATCH_REGEX, re.DOTALL)

    llm = vllm.LLM("mistralai/Mistral-7B-Instruct-v0.2", max_model_len=16384)
    sampling_params = vllm.SamplingParams(max_tokens=128)

    qa_dataframe = pd.read_csv("data/qa.csv")

    for index, row in tqdm.tqdm(
        qa_dataframe.iterrows(), total=len(qa_dataframe), desc="Adding tags"
    ):
        question = str(row["question"])
        answer = str(row["answer"])

        prompt = PROMPT_TEMPLATE.format(possible_tag_list_str, question, answer)

        output = llm.generate(
            prompt,
            sampling_params=sampling_params,
            use_tqdm=False,
        )[0]

        genetated_text = output.outputs[0].text

        match = answer_matcher.search(genetated_text)
        if match is None:
            logging.error(
                f"failed to find answer in generated text for {question} {answer}: the generated text is {genetated_text}"
            )
            continue

        tags_str = match.group(1).replace("\n", "")

        qa_dataframe.at[index, "tags"] = tags_str

    qa_dataframe.to_csv("data/qat.csv", index=False)


if __name__ == "__main__":
    main()
