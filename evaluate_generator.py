import json
import os
import re
from typing import TypedDict

import PIL.Image
import torch
import tqdm
from llava.constants import IMAGE_TOKEN_INDEX
from llava.conversation import conv_llava_v1
from llava.mm_utils import (
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
)
from llava.model.builder import load_pretrained_model
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
from transformers import CLIPImageProcessor, LlamaTokenizer

from custom_dataset import CustomDataset

MODEL_PATH = "./data/models/llava-v1.5-7b-task-lora-generator"

IMAGE_DIR = "./data/dataset/images"
VAL_METADATA_FILE = "./data/dataset/val.json"
OUTPUT_GENERATED_RESULTS = "./data/dataset/generation_results.json"

GENERATION_NUM_PER_QA = 10


class GeneratedResultEntry(TypedDict):
    id: int
    qa_id: int
    predicted: str
    correctness: bool


class IdGenerator:
    def __init__(self):
        self._id = 0

    def generate(self) -> int:
        self._id += 1
        return self._id - 1


def main():
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=MODEL_PATH,
        model_base=None,
        model_name=get_model_name_from_path(MODEL_PATH),
    )
    assert isinstance(tokenizer, LlamaTokenizer)
    assert isinstance(model, LlavaLlamaForCausalLM)
    assert isinstance(image_processor, CLIPImageProcessor)
    assert isinstance(context_len, int)

    dataset = CustomDataset(
        metadata_file_path=VAL_METADATA_FILE,
    )

    result_entries: list[GeneratedResultEntry] = []
    id_generator = IdGenerator()

    with torch.inference_mode():
        correct_count = 0
        wrong_count = 0
        progress_bar = tqdm.tqdm(total=len(dataset) * GENERATION_NUM_PER_QA)
        for entry in dataset:
            image_id = entry["image_id"]
            question = entry["question"]
            answer = entry["answer"]

            image_path = os.path.join(IMAGE_DIR, f"{image_id}.png")
            image = PIL.Image.open(image_path).convert("RGB")

            image_tensor = process_images(
                [image],
                image_processor,
                model.config,
            )
            assert isinstance(image_tensor, torch.Tensor)
            image_tensor = image_tensor.to(model.device, dtype=torch.float16)

            conversation = conv_llava_v1.copy()
            conversation.append_message(
                conversation.roles[0],
                f"<image>\n{question}",
            )
            conversation.append_message(conversation.roles[1], None)
            prompt = conversation.get_prompt()

            input_ids = tokenizer_image_token(
                prompt,
                tokenizer,
                IMAGE_TOKEN_INDEX,
                return_tensors="pt",
            )
            assert isinstance(input_ids, torch.Tensor)
            input_ids = input_ids.unsqueeze(0).to(model.device)

            for i in range(GENERATION_NUM_PER_QA):
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    image_sizes=torch.Tensor([image.size]),
                    do_sample=True,
                    temperature=0.8,
                    max_new_tokens=1024,
                    use_cache=True,
                )

                outputs = tokenizer.decode(output_ids[0]).strip()

                predicted_answer = extract_answer(outputs)

                correct = check_answer(predicted_answer, answer)
                if correct:
                    correct_count += 1
                else:
                    wrong_count += 1

                result_entry: GeneratedResultEntry = {
                    "id": id_generator.generate(),
                    "qa_id": entry["id"],
                    "predicted": outputs,
                    "correctness": correct,
                }

                result_entries.append(result_entry)

                progress_bar.update()
                progress_bar.set_postfix(
                    acc=f"{correct_count / (correct_count+wrong_count):.2%}"
                )

            with open(OUTPUT_GENERATED_RESULTS, "w") as f:
                json.dump(result_entries, f, indent=4)

        progress_bar.close()


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


if __name__ == "__main__":
    main()
