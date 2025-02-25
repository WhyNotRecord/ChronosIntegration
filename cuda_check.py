import torch


def check_available():
  if torch.cuda.is_available():
    print("cuda is available")
    print(torch.cuda.get_device_name(0))
