import re

import torch
import tqdm
from llava.constants import IMAGE_TOKEN_INDEX
from llava.conversation import conv_llava_v1
from llava.eval.run_llava import eval_model
from llava.mm_utils import (
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
)
from llava.model.builder import load_pretrained_model
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
from transformers import CLIPImageProcessor, LlamaTokenizer

from custom_dataset import CustomDataset

MODEL_PATH = "./data/models/llava-v1.5-7b-task-lora"

IMAGE_DIR = "./data/dataset/images"
TEST_METADATA_FILE = "./data/dataset/test.json"


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
        metadata_file_path=TEST_METADATA_FILE,
        image_dir=IMAGE_DIR,
    )

    with torch.inference_mode():
        correct_count = 0
        progress_bar = tqdm.tqdm(total=len(dataset))
        for i in range(len(dataset)):
            entry = dataset[i]
            image = entry["image"]
            question = entry["question"]
            answer = entry["answer"]

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

            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=torch.Tensor([image.size]),
                do_sample=False,
                temperature=0,
                max_new_tokens=1024,
                use_cache=True,
            )

            outputs = tokenizer.decode(output_ids[0]).strip()

            predicted_answer = extract_answer(outputs)

            if check_answer(predicted_answer, answer):
                correct_count += 1

            progress_bar.update()
            progress_bar.set_postfix(acc=f"{correct_count / (i + 1):.2%}")


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
