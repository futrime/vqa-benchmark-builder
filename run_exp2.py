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

        prompt_q = f"[INST] <image> {question} [/INST] Let's think step by step: <steps> First,"
        inputs_q = processor(prompt_q, image, return_tensors="pt").to(device)

        outputs_q = model.generate(
            **inputs_q,
            max_new_tokens=1024,
            pad_token_id=model.pad_token_id,
        )
        predicted_q = processor.decode(outputs_q[0], skip_special_tokens=True)
        assert isinstance(predicted_q, str)

        predicted_steps = extract_steps(predicted_q)

        prompt_qs = f"[INST] <image> {question} [/INST] Let's think step by step: <steps>{predicted_steps}</steps> To answer in one word: <answer>"
        inputs_qs = processor(prompt_qs, image, return_tensors="pt").to(device)

        outputs_qs = model.generate(
            **inputs_qs,
            max_new_tokens=1024,
            pad_token_id=model.pad_token_id,
        )
        predicted_qs = processor.decode(outputs_qs[0], skip_special_tokens=True)
        assert isinstance(predicted_qs, str)

        predicted_answer = extract_answer(predicted_qs)

        if check_answer(predicted_answer, answer):
            correct_count += 1

        progress_bar.desc = f"Accuracy: {correct_count / (i+1):.2%}"
        progress_bar.update()

    print(f"Accuracy: {correct_count / len(dataset):.2%}")


def check_answer(predicted_answer: str, answer: str) -> bool:
    if predicted_answer.lower().find(answer.lower()) != -1:
        return True

    return False


def extract_answer(predicted: str) -> str:
    if predicted.lower().find("</answer>") == -1:
        predicted += "</answer>"

    match = re.search(
        r"<answer>(.*)</answer>",
        predicted,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match is None:
        raise ValueError("answer not found in predicted text")

    return match.group(1)


def extract_steps(predicted: str) -> str:
    if predicted.lower().find("</steps>") == -1:
        predicted += "</steps>"

    match = re.search(
        r"<steps>(.*)</steps>",
        predicted,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match is None:
        raise ValueError("steps not found in predicted text")

    return match.group(1)


if __name__ == "__main__":
    main()
