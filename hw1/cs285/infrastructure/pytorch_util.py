import torch

device = None


def init_gpu(use_gpu=True, gpu_id=0):
    global device
    device = torch.device("mps")
    print("using device:", device)


def set_device(gpu_id):
    assert False


def from_numpy(*args, **kwargs):
    return torch.from_numpy(*args, **kwargs).float().to(device)


def to_numpy(tensor):
    return tensor.to('cpu').detach().numpy()
