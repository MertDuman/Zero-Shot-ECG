import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from pytorch_sklearn.utils.func_utils import to_tensor


class DefaultDataset(Dataset):
    """
    A simple dataset for convenience.
    """
    def __init__(self, X, y=None):
        X = to_tensor(X, clone=False)  # Convert to tensor if it is not already.
        if y is not None:
            y = to_tensor(y, clone=False)

        self.X = X
        self.y = y
        self.n = X.shape[0]

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        if self.y is None:
            return self.X[index, ...]
        else:
            return self.X[index, ...], self.y[index, ...]


class CUDADataset(Dataset):
    """
    A simple CUDA dataset, where the data will be sent to the cuda device. This is useful when the data is
    small and can be stored entirely in GPU memory, as this will speed up training.
    """
    def __init__(self, X, y=None):
        device = "cuda"
        X = to_tensor(X, device=device, clone=False)
        if y is not None:
            y = to_tensor(y, device=device, clone=False)

        self.X = X
        self.y = y
        self.n = X.shape[0]

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        if self.y is None:
            return self.X[index, ...]
        else:
            return self.X[index, ...], self.y[index, ...]
