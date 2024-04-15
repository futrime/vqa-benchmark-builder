import GPUtil
import torch.cuda


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
