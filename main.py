import torch
import transformers

from lora_dataset import LoRADataset

IMAGE_DIR = "data/LoRA/valid"
QUESTION_FILE_PATH = "data/LoRA/Questions/lora_vqa_valid.json"


def main() -> None:
    dataset = LoRADataset(
        question_file_path=QUESTION_FILE_PATH,
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

    model.to("cuda:0")  # type: ignore

    for i in range(0, 10):
        image, question = dataset[i]
        prompt = f"[INST] <image>\n{question['question']} Answer in short form. [/INST]"

        inputs = processor(prompt, image, return_tensors="pt").to("cuda:0")
        outputs = model.generate(**inputs, max_new_tokens=100)

        answer = processor.decode(outputs[0], skip_special_tokens=True)

        print(answer)
        print()


if __name__ == "__main__":
    main()
