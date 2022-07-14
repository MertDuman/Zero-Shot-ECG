import numpy as np
import torch


def to_numpy(X: torch.Tensor, clone=True):
    """
    Safely convert from PyTorch tensor to numpy.
    ``clone`` is set to True by default to mitigate side-effects that this function might cause.
    For instance:
        ``torch.Tensor.cpu`` will clone the object if it is in GPU, but won't if it is in CPU.
        ``clone`` allows this function to clone the input always.
    """
    if isinstance(X, np.ndarray):
        if clone:
            return X.copy()
        else:
            return X

    old_memory = get_memory_loc(X)
    if X.requires_grad:
        X = X.detach()
    if X.is_cuda:
        X = X.cpu()
    if clone and old_memory == get_memory_loc(X):
        X = X.clone()
    return X.numpy()


def to_tensor(X: np.ndarray, device=None, dtype=None, clone=True):
    """
    Converts the given input to ``torch.Tensor`` and optionally clones it (True by default).
    If ``clone`` is False, this function may still clone the input, read ``torch.as_tensor``.
    """
    old_memory = get_memory_loc(X)
    X = torch.as_tensor(X, device=device, dtype=dtype)
    if clone and old_memory == get_memory_loc(X):
        X = X.clone()
    return X


def to_safe_tensor(X: torch.Tensor, clone=True):
    """
    Convert the given ``torch.Tensor`` to another one that is detached and is in cpu.
    ``clone`` is set to True by default to mitigate side-effects that this function might cause.
    For instance:
        ``torch.Tensor.cpu`` will clone the object if it is in GPU, but won't if it is in CPU.
        ``clone`` allows this function to clone the input always.
    """
    old_memory = get_memory_loc(X)
    if X.requires_grad:
        X = X.detach()
    if X.is_cuda:
        X = X.cpu()
    if clone and old_memory == get_memory_loc(X):
        X = X.clone()
    return X


def get_memory_loc(X):
    if isinstance(X, np.ndarray):
        return X.__array_interface__['data'][0]
    if isinstance(X, torch.Tensor):
        return X.data_ptr()
    return -1
    # raise TypeError("Cannot get memory location of this data type.")
