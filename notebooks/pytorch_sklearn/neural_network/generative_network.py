import pickle
import copy
import warnings

import torch
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer as _Optimizer
from torch.utils.data import DataLoader, Dataset

from pytorch_sklearn.utils import DefaultDataset, CUDADataset
from pytorch_sklearn.callbacks import CallbackManager
from pytorch_sklearn.utils.class_utils import set_properties_hidden
from pytorch_sklearn.utils.func_utils import to_tensor, to_safe_tensor


class CycleGAN:
    """
    Implements CycleGAN from the paper: https://github.com/junyanz/CycleGAN
    Follows similar implementation to: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

    Parameters
    ----------
    generator_A : PyTorch module
        Generator that tries to convert class A to class B
    generator_B : PyTorch module
        Generator that tries to convert class B to class A
    discriminator_A : PyTorch module
        Discriminator that classifies input as "from class A" or "not from class A"
    discriminator_B : PyTorch module
        Discriminator that classifies input as "from class B" or "not from class B"
    optimizer_Gen : PyTorch optimizer
        Updates the weights of the generators.
    optimizer_Disc : PyTorch optimizer
        Updates the weights of the discriminators.
    criterion : PyTorch loss
        GAN loss that will be applied to discriminator outputs.
    """
    def __init__(self, generator_A, generator_B, discriminator_A, discriminator_B, optimizer_Gen, optimizer_Disc, criterion):
        # Base parameters
        self.generator_A = generator_A
        self.generator_B = generator_B
        self.discriminator_A = discriminator_A
        self.discriminator_B = discriminator_B
        self.optimizer_Gen = optimizer_Gen
        self.optimizer_Disc = optimizer_Disc
        self.criterion = criterion
        self.cbmanager = CallbackManager()  # SAVED
        self.keep_training = True

        # Maintenance parameters
        self._using_original = True  # SAVED
        self._original_state_dict = None  # SAVED

        # Fit function parameters
        self._train_X = None
        self._train_y = None
        self._validate = None
        self._val_X = None
        self._val_y = None
        self._max_epochs = None
        self._batch_size = None
        self._use_cuda = None
        self._fits_gpu = None
        self._device = None
        self._callbacks = None
        self._metrics = None

        # Fit runtime parameters
        self._epoch = None
        self._batch = None
        self._batch_X = None
        self._batch_y = None
        self._batch_out = None
        self._batch_loss = None
        self._pass_type = None
        self._num_batches = None
        self._train_loader = None
        self._val_loader = None

        # Predict function parameters
        self._test_X = None
        self._decision_func = None
        self._decision_func_kw = None

        # Predict runtime parameters
        self._predict_loader = None
        self._pred_y = None
        self._batch = None
        self._batch_X = None

        # Predict Proba function parameters
        self._test_X = None

        # Predict Proba runtime parameters
        self._predict_proba_loader = None
        self._proba = None
        self._batch = None
        self._batch_X = None

        # Score function parameters
        self._test_X = None
        self._test_y = None
        self._score_func = None
        self._score_func_kw = None

        # Score runtime parameters
        self._score_loader = None
        self._out = None
        self._score = None
        self._batch = None
        self._batch_X = None
        self._batch_y = None

    @property
    def callbacks(self):
        return self.cbmanager.callbacks

    @property
    def history(self):
        return self.cbmanager.history

    # Model Training Core Functions
    def forward(self, X: torch.Tensor):
        return self.module(X)

    def get_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        return self.criterion(y_pred, y_true)

    def zero_grad(self):
        self.optimizer.zero_grad(set_to_none=True)

    def compute_grad(self, loss: torch.Tensor):
        loss.backward()

    def step_grad(self):
        self.optimizer.step()

    def backward(self, loss: torch.Tensor):
        self.zero_grad()
        self._notify(f"on_grad_compute_begin")
        self.compute_grad(loss)
        self._notify(f"on_grad_compute_end")
        self.step_grad()

    # Model Modes
    def train(self):
        self.module.train()
        self._pass_type = "train"

    def val(self):
        self.module.eval()
        self._pass_type = "val"

    def test(self):
        self.module.eval()
        self._pass_type = "test"

    # Model Training Main Functions
    def fit(
            self,
            train_X=None,
            train_y=None,
            validate=False,
            val_X=None,
            val_y=None,
            max_epochs=10,
            batch_size=32,
            use_cuda=True,
            fits_gpu=False,
            callbacks=None,
            metrics=None
    ):
        # Handle None inputs.
        callbacks = [] if callbacks is None else callbacks
        metrics = {} if metrics is None else metrics
        device = "cuda" if use_cuda else "cpu"
        if not use_cuda and fits_gpu:
            fits_gpu = False
            warnings.warn("Fits gpu is true, but not using CUDA.")

        if max_epochs == -1:
            max_epochs = float("inf")
            warnings.warn("max_epochs is set to -1. Make sure to pass an early stopping method.")

        #  Set fit class parameters
        fit_params = locals().copy()
        set_properties_hidden(**fit_params)

        # Handle CallbackManager
        self.cbmanager.callbacks = callbacks

        # Define DataLoaders
        self._train_X = self._to_tensor(self._train_X)
        self._train_y = self._to_tensor(self._train_y)
        self._train_loader = self.get_dataloader(self._train_X, self._train_y, self._batch_size, shuffle=True)
        if self._validate:
            self._val_X = self._to_tensor(self._val_X)
            self._val_y = self._to_tensor(self._val_y)
            self._val_loader = self.get_dataloader(self._val_X, self._val_y, self._batch_size, shuffle=True)

        # Begin Fit
        self.module = self.module.to(self._device)
        self._notify("on_fit_begin")
        self._epoch = 1
        while self._epoch < self._max_epochs + 1:
            if not self.keep_training:
                self._notify("on_fit_interrupted")
                break
            self.train()
            self.fit_epoch(self._train_loader)
            if self._validate:
                with torch.no_grad():
                    self.val()
                    self.fit_epoch(self._val_loader)

            if self._epoch == self._max_epochs:
                break  # so that self._epoch == self._max_epochs when loop exits.
            self._epoch += 1
        self._notify("on_fit_end")

    def fit_epoch(self, data_loader):
        self._num_batches = len(data_loader)
        self._notify(f"on_{self._pass_type}_epoch_begin")
        for self._batch, (self._batch_X, self._batch_y) in enumerate(data_loader, start=1):
            self._batch_X = self._batch_X.to(self._device, non_blocking=True)
            self._batch_y = self._batch_y.to(self._device, non_blocking=True)
            self.fit_batch(self._batch_X, self._batch_y)
        self._notify(f"on_{self._pass_type}_epoch_end")

    def fit_batch(self, X, y):
        self._notify(f"on_{self._pass_type}_batch_begin")
        self._batch_out = self.forward(X)
        self._batch_loss = self.get_loss(self._batch_out, y)
        if self._pass_type == "train":
            self.backward(self._batch_loss)
        self._notify(f"on_{self._pass_type}_batch_end")

    def get_dataloader(self, X, y, batch_size, shuffle):
        dataset = self.get_dataset(X, y)
        if self._fits_gpu:
            return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=True)

    def get_dataset(self, X, y):
        if isinstance(X, Dataset):
            return X
        if self._fits_gpu:
            return CUDADataset(X, y)
        return DefaultDataset(X, y)

    def predict(self, test_X, decision_func=None, **decision_func_kw):
        #  Set predict class parameters
        predict_params = locals().copy()
        set_properties_hidden(**predict_params)

        self._test_X = self._to_tensor(self._test_X)
        self._predict_loader = self.get_dataloader(self._test_X, None, self._batch_size, shuffle=False)

        with torch.no_grad():
            self.test()
            self._notify("on_predict_begin")
            self._pred_y = []
            for self._batch, (self._batch_X) in enumerate(self._predict_loader, start=1):
                self._batch_X = self._batch_X.to(self._device, non_blocking=True)
                pred_y = self.forward(self._batch_X)
                if self._decision_func is not None:
                    pred_y = self._decision_func(pred_y, **self._decision_func_kw)
                self._pred_y.append(pred_y)
            self._pred_y = torch.cat(self._pred_y)
            self._notify("on_predict_end")
        return self._pred_y

    def predict_proba(self, test_X):
        #  Set predict_proba class parameters
        proba_params = locals().copy()
        set_properties_hidden(**proba_params)

        self._test_X = self._to_tensor(self._test_X)
        self._predict_proba_loader = self.get_dataloader(self._test_X, None, self._batch_size, shuffle=False)

        with torch.no_grad():
            self.test()
            self._notify("on_predict_proba_begin")
            self._proba = []
            for self._batch, (self._batch_X) in enumerate(self._predict_proba_loader, start=1):
                self._batch_X = self._batch_X.to(self._device, non_blocking=True)
                self._proba.append(self.forward(self._batch_X))
            self._proba = torch.cat(self._proba)
            self._notify("on_predict_proba_end")
        return self._proba

    def score(self, test_X, test_y, score_func=None, **score_func_kw):
        #  Set score class parameters
        score_params = locals().copy()
        set_properties_hidden(**score_params)

        self._test_X = self._to_tensor(self._test_X)
        self._test_y = self._to_tensor(self._test_y)
        self._score_loader = self.get_dataloader(self._test_X, self._test_y, self._batch_size, shuffle=False)

        with torch.no_grad():
            self.test()
            self._out = []
            self._score = []
            for self._batch, (self._batch_X, self._batch_y) in enumerate(self._score_loader, start=1):
                self._batch_X = self._batch_X.to(self._device, non_blocking=True)
                self._batch_y = self._batch_y.to(self._device, non_blocking=True)
                batch_out = self.forward(self._batch_X)
                if self._score_func is None:
                    batch_loss = self.get_loss(batch_out, self._batch_y).item()
                else:
                    batch_loss = self._score_func(self._to_safe_tensor(batch_out), self._to_safe_tensor(self._batch_y),
                                                  **self._score_func_kw)
                self._out.append(batch_out)
                self._score.append(batch_loss)
            self._out = torch.cat(self._out)
            self._score = torch.Tensor(self._score).mean()
        return self._score

    def _notify(self, method_name, **cb_kwargs):
        for callback in self.cbmanager.callbacks:
            if method_name in callback.__class__.__dict__:  # check if method is overridden
                getattr(callback, method_name)(self, **cb_kwargs)

    def _to_tensor(self, X):
        return to_tensor(X, clone=False)

    def _to_safe_tensor(self, X):
        return to_safe_tensor(X, clone=False)

    def load_weights(self, weight_checkpoint):
        if self._using_original:
            self._original_state_dict = copy.deepcopy(self.module.state_dict())
        self.module.load_state_dict(weight_checkpoint.best_weights)
        self._using_original = False

    def load_weights_from_path(self, weight_path):
        if self._using_original:
            self._original_state_dict = copy.deepcopy(self.module.state_dict())
        self.module.load_state_dict(torch.load(weight_path))
        self._using_original = False

    def load_original_weights(self):
        if not self._using_original:
            self.module.load_state_dict(self._original_state_dict)
            self._using_original = True
            self._original_state_dict = None

    @classmethod
    def save_class(cls, net, savepath):
        d = {
            "module_state": net.module.state_dict(),
            "original_module_state": net._original_state_dict,
            "using_original": net._using_original,
            "optimizer_state": net.optimizer.state_dict(),
            "criterion_state": net.criterion.state_dict(),
            "cbmanager": net.cbmanager
        }
        with open(savepath, "wb") as f:
            pickle.dump(d, f)

    @classmethod
    def load_class(cls, loadpath, module=None, optimizer=None, criterion=None):
        with open(loadpath, "rb") as f:
            d = pickle.load(f)

        if module is not None:
            module.load_state_dict(d["module_state"])
        if optimizer is not None:
            optimizer.load_state_dict(d["optimizer_state"])
        if criterion is not None:
            criterion.load_state_dict(d["criterion_state"])
        net = NeuralNetwork(module, optimizer, criterion)
        net.cbmanager = d["cbmanager"]
        net._using_original = d["using_original"]
        net._original_state_dict = d["original_module_state"]
        return net