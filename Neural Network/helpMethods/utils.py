import torch
from torch._prims_common import DeviceLikeType


def get_device() -> DeviceLikeType:
    """Function to determine whether to run the training on GPU or CPU."""

    # Device will determine whether to run the training on GPU or CPU.
    if torch.backends.mps.is_available():  # GPU on MacOS
        device = "mps"
    elif torch.cuda.is_available():  # GPU on Linux/Windows
        device = "cuda"
    else:  # default to CPU if no GPU is available
        device = "cpu"

    device = torch.device(device)
    print(f"Running pytorch version ({torch.__version__}) with backend = {device}")

    return device
