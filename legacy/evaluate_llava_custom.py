import re

import GPUtil
import torch
import tqdm
import transformers

from custom_dataset import CustomDataset

IMAGE_DIR = "data/custom/images"
QA_FILE = "data/custom/qa.json"


def main():
    device = get_first_available_gpu()

    dataset = CustomDataset(
        metadata_file_path=QA_FILE,
        image_dir=IMAGE_DIR,
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
        entry = dataset[i]
        image = entry["image"]
        question = entry["question"]
        answer = entry["answer"]

        prompt = f"[INST] <image> {question} [/INST] To answer in one word: <answer>"
        inputs = processor(prompt, image, return_tensors="pt").to(device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            pad_token_id=model.pad_token_id,
        )
        predicted = processor.decode(outputs[0], skip_special_tokens=True)
        assert isinstance(predicted, str)

        predicted_answer = extract_answer(predicted)

        if check_answer(predicted_answer, answer):
            correct_count += 1
        else:
            pass

        progress_bar.update()
        progress_bar.set_postfix(
            correctness=f"{correct_count}/{i+1} ({correct_count / (i+1):.2%})",
        )

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


def get_first_available_gpu() -> torch.device:
    gpu_id_list = GPUtil.getFirstAvailable(
        order="first",
        maxLoad=0.5,
        maxMemory=0.5,
        attempts=1,
        interval=900,
        verbose=False,
    )

    gpu_id = gpu_id_list[0]

    device = torch.device(f"cuda:{gpu_id}")

    return device


if __name__ == "__main__":
    main()
