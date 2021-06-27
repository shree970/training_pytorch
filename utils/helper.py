import random

import numpy as np
import torch
from torchsummary import summary


def model_summary(model, input_size):
    """
    Summary of the model.
    """
    summary(model, input_size=input_size)


def get_default_device():
    """
    Pick GPU if available, else CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def seed_everything(seed: int):
    """
    Seed everything for reproducibility and deterministic behaviour.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
