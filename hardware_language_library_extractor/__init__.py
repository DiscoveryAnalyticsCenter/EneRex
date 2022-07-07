import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
device_id = -1
if device.type == "cuda":
    torch.cuda.set_device(device)
    device_id = 0
