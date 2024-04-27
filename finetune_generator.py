import GPUtil
import torch
from custom_dataset import CustomDataset

IMAGE_DIR = "data/custom/images"
QA_FILE = "data/custom/qa.json"

FINETUNING_ENTRY_COUNT = 10000


def main():
    device = get_first_available_gpu()

    dataset = CustomDataset(
        qa_file_path=QA_FILE,
        image_dir=IMAGE_DIR,
    )


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
