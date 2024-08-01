import torch
import torch.backends.cudnn as cudnn


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
if use_cuda:
    cudnn.benchmark = True
    print('Using CUDA ..')


def from_numpy(*args, **kwargs):
    return torch.from_numpy(*args, **kwargs).float().to(device, non_blocking=True)


def to_numpy(x):
    return x.detach().float().cpu().numpy()


def dict_from_numpy(np_dict):
    return {
        k: from_numpy(v) for k, v in np_dict.items()
    }


def zeros(sizes, **kwargs):
    return torch.zeros(sizes, **kwargs).float().to(device, non_blocking=True)


def zeros_like(*args, **kwargs):
    return torch.zeros_like(*args, **kwargs).float().to(device, non_blocking=True)


def ones(sizes, **kwargs):
    return torch.ones(sizes, **kwargs).float().to(device, non_blocking=True)


def ones_like(*args, **kwargs):
    return torch.ones_like(*args, **kwargs).float().to(device, non_blocking=True)


def tensor(*args, **kwargs):
    return torch.tensor(*args, **kwargs).to(device, non_blocking=True)


def dict_to_numpy(tensor_dict):
    return {
        k: to_numpy(v) for k, v in tensor_dict.items()
    }


def dict_detach_to_numpy(tensor_dict):
    return {
        k: to_numpy(v) for k, v in tensor_dict.items()
    }


def dict_to_tensor(np_dict):
    return {
        k: to_tensor(v) for k, v in np_dict.items()
    }


def to_tensor(*args, **kwargs):
    return torch.as_tensor(*args, **kwargs).float().to(device, non_blocking=True)
