import torch

## File that does all the preperations for reproducability


def prepare():
    torch.manual_seed(0)

    # When running on the CuDNN backend two further options must be set for reproducibility
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False