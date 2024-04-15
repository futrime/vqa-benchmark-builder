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

        prompt = f"[INST] <image>\n{question} Answer in short form. [/INST]"
        inputs = processor(prompt, image, return_tensors="pt").to(device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            pad_token_id=model.pad_token_id,
        )
        predicted = processor.decode(outputs[0], skip_special_tokens=True)

        if check_answer(predicted, answer):
            correct_count += 1

        progress_bar.desc = f"Accuracy: {correct_count / (i+1):.2%}"
        progress_bar.update()

    print(f"Accuracy: {correct_count / len(dataset):.2%}")


def check_answer(predicted: str, answer: str):
    predicted = re.sub(
        r"\[INST\].*\[/INST\]",
        "",
        predicted,
        flags=re.DOTALL,
    )

    if predicted.lower().find(answer.lower()) != -1:
        return True

    return False


if __name__ == "__main__":
    main()
