import re

import torch
import torch.nn
import torch.utils.data
import tqdm
import transformers

from config import IMAGE_TEST_DIR, QUESTION_TEST_FILE_PATH
from lora_dataset import LoRADataset
from utils import get_first_available_gpu


def main() -> None:
    device = get_first_available_gpu()

    dataset = LoRADataset(
        question_file_path=QUESTION_TEST_FILE_PATH, image_dir=IMAGE_TEST_DIR
    )

    processor = transformers.LlavaNextProcessor.from_pretrained(
        "llava-hf/llava-v1.6-mistral-7b-hf"
    )
    assert isinstance(processor, transformers.LlavaNextProcessor)

    model = transformers.LlavaNextForConditionalGeneration.from_pretrained(
        "llava-hf/llava-v1.6-mistral-7b-hf",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    assert isinstance(model, transformers.LlavaNextForConditionalGeneration)

    model.to(device)  # type: ignore

    correct_count = 0

    progress_bar = tqdm.tqdm(total=len(dataset))
    for i in range(len(dataset)):
        image, question, answer = dataset[i]

        q_prompt = f"[INST] <image>\n{question} Think in steps and answer in short form. [/INST] Let's think step by step:\n<steps>\n1."
        q_inputs = processor(q_prompt, image, return_tensors="pt").to(device)

        qs_outputs = model.generate(
            **q_inputs,
            max_new_tokens=2000,
            pad_token_id=model.pad_token_id,
        )
        qs_predicted = processor.decode(qs_outputs[0], skip_special_tokens=True)
        assert isinstance(qs_predicted, str)

        if qs_predicted.lower().find("</steps>") == -1:
            qs_predicted += "</steps>\n"

        qs_prompt = f"{qs_predicted}"
        qs_inputs = processor(qs_prompt, image, return_tensors="pt").to(device)

        qsa_outputs = model.generate(
            **qs_inputs,
            max_new_tokens=20,
            pad_token_id=model.pad_token_id,
        )
        qsa = processor.decode(qsa_outputs[0], skip_special_tokens=True)
        assert isinstance(qsa, str)

        if check_answer(qsa, answer):
            correct_count += 1

        progress_bar.desc = f"Accuracy: {correct_count / (i+1):.2%}"
        progress_bar.update()

    print(f"Accuracy: {correct_count / len(dataset):.2%}")


def check_answer(predicted: str, answer: str) -> bool:
    if predicted.lower().find("</answer>") == -1:
        predicted += "</answer>"

    match = re.search(
        r"<answer>(.*)</answer>",
        predicted,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match is None:
        return False

    predicted_answer = match.group(1)

    if predicted_answer.lower().find(answer.lower()) != -1:
        return True

    return False


if __name__ == "__main__":
    main()
