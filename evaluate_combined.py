import os
import re

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

GENERATOR_MODEL_PATH = "./data/models/llava-v1.5-7b-task-lora-generator"
VERIFIER_MODEL_PATH = "./data/models/llava-v1.5-7b-task-lora-verifier"

IMAGE_DIR = "./data/dataset/images"
TEST_METADATA_FILE = "./data/dataset/test.json"

MAX_GENERATION_TRIES = 10


def main():
    gen_tokenizer, gen_model, gen_image_processor, gen_context_len = (
        load_pretrained_model(
            model_path=GENERATOR_MODEL_PATH,
            model_base=None,
            model_name=get_model_name_from_path(GENERATOR_MODEL_PATH),
        )
    )
    assert isinstance(gen_tokenizer, LlamaTokenizer)
    assert isinstance(gen_model, LlavaLlamaForCausalLM)
    assert isinstance(gen_image_processor, CLIPImageProcessor)
    assert isinstance(gen_context_len, int)

    veri_tokenizer, veri_model, veri_image_processor, veri_context_len = (
        load_pretrained_model(
            model_path=VERIFIER_MODEL_PATH,
            model_base=None,
            model_name=get_model_name_from_path(VERIFIER_MODEL_PATH),
        )
    )
    assert isinstance(veri_tokenizer, LlamaTokenizer)
    assert isinstance(veri_model, LlavaLlamaForCausalLM)
    assert isinstance(veri_image_processor, CLIPImageProcessor)
    assert isinstance(veri_context_len, int)

    dataset = CustomDataset(
        metadata_file_path=TEST_METADATA_FILE,
    )

    with torch.inference_mode():
        correct_count = 0
        verification_fail_count = 0
        wrong_count = 0
        progress_bar = tqdm.tqdm(total=len(dataset))
        for entry in dataset:
            image_id = entry["image_id"]
            question = entry["question"]
            answer = entry["answer"]

            image_path = os.path.join(IMAGE_DIR, f"{image_id}.png")
            image = PIL.Image.open(image_path).convert("RGB")

            gen_image_tensor = process_images(
                [image],
                gen_image_processor,
                gen_model.config,
            )
            assert isinstance(gen_image_tensor, torch.Tensor)
            gen_image_tensor = gen_image_tensor.to(
                gen_model.device, dtype=torch.float16
            )

            veri_image_tensor = process_images(
                [image],
                veri_image_processor,
                veri_model.config,
            )
            assert isinstance(veri_image_tensor, torch.Tensor)
            veri_image_tensor = veri_image_tensor.to(
                veri_model.device, dtype=torch.float16
            )

            gen_conversation = conv_llava_v1.copy()
            gen_conversation.append_message(
                gen_conversation.roles[0],
                f"<image>\n{question}",
            )
            gen_conversation.append_message(gen_conversation.roles[1], None)
            gen_prompt = gen_conversation.get_prompt()

            gen_input_ids = tokenizer_image_token(
                gen_prompt,
                gen_tokenizer,
                IMAGE_TOKEN_INDEX,
                return_tensors="pt",
            )
            assert isinstance(gen_input_ids, torch.Tensor)
            gen_input_ids = gen_input_ids.unsqueeze(0).to(gen_model.device)

            is_verified = False
            predicted_answer = ""

            for _ in range(MAX_GENERATION_TRIES):
                gen_output_ids = gen_model.generate(
                    gen_input_ids,
                    images=gen_image_tensor,
                    image_sizes=torch.Tensor([image.size]),
                    do_sample=True,
                    temperature=0.8,
                    max_new_tokens=1024,
                    use_cache=True,
                )

                gen_outputs = gen_tokenizer.decode(gen_output_ids[0])
                gen_outputs = gen_outputs.removeprefix("<s>")
                gen_outputs = gen_outputs.removesuffix("</s>")

                veri_conversation = conv_llava_v1.copy()
                veri_conversation.append_message(
                    veri_conversation.roles[0],
                    f"<image>\n{question}\n{gen_outputs}",
                )
                veri_conversation.append_message(veri_conversation.roles[1], None)
                veri_prompt = veri_conversation.get_prompt()

                veri_input_ids = tokenizer_image_token(
                    veri_prompt,
                    veri_tokenizer,
                    IMAGE_TOKEN_INDEX,
                    return_tensors="pt",
                )
                assert isinstance(veri_input_ids, torch.Tensor)
                veri_input_ids = veri_input_ids.unsqueeze(0).to(veri_model.device)

                veri_output_ids = veri_model.generate(
                    veri_input_ids,
                    images=veri_image_tensor,
                    image_sizes=torch.Tensor([image.size]),
                    do_sample=True,
                    temperature=0.8,
                    max_new_tokens=1024,
                    use_cache=True,
                )

                veri_outputs = veri_tokenizer.decode(veri_output_ids[0])

                is_verified = check_veri_answer(veri_outputs)

                if is_verified:
                    predicted_answer = extract_gen_answer(gen_outputs)
                    break

            if not is_verified:
                verification_fail_count += 1

            else:
                correct = check_answer(predicted_answer, answer)
                if correct:
                    correct_count += 1
                else:
                    wrong_count += 1

            progress_bar.update()
            progress_bar.set_postfix(
                acc=f"{correct_count / (correct_count+wrong_count+verification_fail_count):.2%}",
                fail=f"{verification_fail_count / (correct_count+wrong_count+verification_fail_count):.2%}",
            )

        progress_bar.close()


def check_answer(predicted_answer: str, answer: str) -> bool:
    if predicted_answer.lower().find(answer.lower()) != -1:
        return True

    return False


def extract_gen_answer(predicted: str) -> str:
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


def check_veri_answer(predicted: str) -> bool:
    if predicted.lower().find("yes") != -1:
        return True

    return False


if __name__ == "__main__":
    main()
