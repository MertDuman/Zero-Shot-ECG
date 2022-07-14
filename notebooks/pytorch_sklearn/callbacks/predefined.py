from pytorch_sklearn.callbacks import Callback
from pytorch_sklearn.utils.func_utils import to_safe_tensor

import torch

import matplotlib.pyplot as plt
import numpy as np
import copy

# WeightCheckpoint
import os

# Verbose
import time

# LossPlot
from IPython import get_ipython
import matplotlib as mpl
import sys
import importlib

from progress_bar import print_progress


class History(Callback):
    def __init__(self):
        super(History, self).__init__()
        self.name = "History"
        self.track = {}
        self.sessions = []
        self.epoch_metrics = None
        self.num_metrics = -1
        self.session = -1

    def init_training(self, net):
        self.track["train_loss"] = []
        for name in net._metrics.keys():
            self.track[f"train_{name}"] = []

    def init_validation(self, net):
        self.track["val_loss"] = []
        for name in net._metrics.keys():
            self.track[f"val_{name}"] = []

    def on_fit_begin(self, net):
        self.session += 1
        self.num_metrics = len(net._metrics) + 1
        if self.session == 0:
            self.init_training(net)
            if net._validate:
                self.init_validation(net)
        self.sessions.append(len(self.track["train_loss"]) + 1)  # new session starts at epoch = len(train_loss) + 1

    def on_train_epoch_begin(self, net):
        self.epoch_metrics = np.zeros(self.num_metrics)  # reset epoch metrics

    def on_val_epoch_begin(self, net):
        self.epoch_metrics = np.zeros(self.num_metrics)  # reset epoch metrics

    def on_train_epoch_end(self, net):
        self._save_metrics(net)

    def on_val_epoch_end(self, net):
        self._save_metrics(net)

    def _save_metrics(self, net):
        self.epoch_metrics = self.epoch_metrics / net._num_batches
        self.track[f"{net._pass_type}_loss"].append(self.epoch_metrics[0])
        for i, name in enumerate(net._metrics.keys(), start=1):
            self.track[f"{net._pass_type}_{name}"].append(self.epoch_metrics[i])

    def on_train_batch_end(self, net):
        self._calculate_metrics(net)

    def on_val_batch_end(self, net):
        self._calculate_metrics(net)

    def _calculate_metrics(self, net):
        batch_out = to_safe_tensor(net._batch_out)
        batch_y = to_safe_tensor(net._batch_y)
        self.epoch_metrics[0] += net._batch_loss.item()
        for i, metric in enumerate(net._metrics.values(), start=1):
            self.epoch_metrics[i] += metric(batch_out, batch_y)


class Verbose(Callback):
    def __init__(self, verbose=4):
        """
        Prints the following training information:
            - Current Epoch / Total Epochs
            - Current Batch / Total Batches
            - Loss
            - Metrics
            - Total Time + ETA

        Parameters
        ----------
        verbose : int
            Controls how much is printed. Higher levels include the info from lower levels.
            The following levels are valid:
                0: Current Epoch / Total Epochs
                1: Current Batch / Total Batches
                2: Loss
                3: Metrics
                4: Total Time + ETA
        """
        super(Verbose, self).__init__()
        self.name = "Verbose"
        self.verbose = verbose

        # Time info
        self.total_time = 0
        self.rem_time = 0
        self.start_time = 0
        self.end_time = 0

    def on_train_epoch_begin(self, net):
        if self.verbose == 0:
            print(f"Epoch {net._epoch}/{net._max_epochs}", end='\x1b[2k\r', flush=True)
        else:
            print(f"Epoch {net._epoch}/{net._max_epochs}")
        self.total_time = 0

    def on_val_epoch_begin(self, net):
        self.total_time = 0

    def on_train_batch_begin(self, net):
        if self.verbose >= 4:
            self.start_time = time.perf_counter()

    def on_train_batch_end(self, net):
        if self.verbose >= 1:
            self._print(net)

    def on_val_batch_begin(self, net):
        if self.verbose >= 4:
            self.start_time = time.perf_counter()

    def on_val_batch_end(self, net):
        if self.verbose >= 1:
            self._print(net)

    def _print(self, net):
        # Calculate data
        epoch_metrics = net.cbmanager.history.epoch_metrics / net._batch  # mean batch loss
        if self.verbose >= 4:
            self.end_time = time.perf_counter()
            self.total_time += self.end_time - self.start_time
            self.rem_time = ((net._num_batches - net._batch) * self.total_time) / net._batch

        # Fill print data
        opt = None
        if self.verbose >= 2:
            opt = [f"{net._pass_type}_loss: {epoch_metrics[0]:.3f}"]
        if self.verbose >= 3:
            opt.extend([f"{net._pass_type}_{name}: {epoch_metrics[i]:.3f}" for i, name in enumerate(net._metrics.keys(), start=1)])
        if self.verbose >= 4:
            opt.extend([f"Time: {self.total_time:.2f}", f"ETA: {self.rem_time:.2f}"])
        print_progress(net._batch, net._num_batches, opt=opt)


class LossPlot(Callback):
    def __init__(self,
                 new_backend="Qt5Agg",
                 pyplot_name="matplotlib.pyplot",
                 max_col=1,
                 block_on_finish=False,
                 savefig=False,
                 savename=None,
                 figure_kw=None):
        super(LossPlot, self).__init__()

        self.new_backend = new_backend
        self.old_backend = mpl.get_backend()
        self.pyplot_name = pyplot_name
        self.max_col = max_col
        self.is_ipython = get_ipython() is not None
        self.block_on_finish = block_on_finish
        self.savefig = savefig
        self.savename = savename
        if self.savefig:
            assert self.savename is not None, "You must provide a savename."
        self.figure_kw = {} if figure_kw is None else figure_kw

        # on_fit_begin
        self.fig = None
        self.axes = None

    def on_fit_begin(self, net):
        self.switch_qt5()
        plt.ion()  # turn on interactive mode

        num_metrics = net.cbmanager.history.num_metrics
        nrows = int(np.ceil(num_metrics / self.max_col))

        self.fig, self.axes = plt.subplots(nrows, self.max_col, sharex="all", squeeze=False, **self.figure_kw)

        # Delete unused subplots
        for i in range(num_metrics, nrows * self.max_col):
            self.fig.delaxes(self.axes[i // self.max_col, i % self.max_col])
            # self.axes[i // self.max_col, i % self.max_col].set_visible(False)

        # Define empty lines for loss line
        self.axes[0, 0].set_title(f"{net.criterion.__class__.__name__}")
        self.axes[0, 0].plot([], [], "-o", label="train loss")
        if net._validate:
            self.axes[0, 0].plot([], [], "-o", label="val loss")
        self.axes[0, 0].legend()

        # Define empty lines for other metric lines
        for i, name in enumerate(net._metrics.keys(), start=1):
            ax = self.axes[i // self.max_col, i % self.max_col]
            ax.set_title(f"{name.capitalize()}")
            ax.plot([], [], "-o", label=f"train {name}")
            if net._validate:
                ax.plot([], [], "-o", label=f"val {name}")
            ax.legend()

        # h, l = self.axes[0, 0].get_legend_handles_labels()
        # self.fig.legend(l)

    def on_train_epoch_end(self, net):
        self.plot_metrics(net)

    def on_val_epoch_end(self, net):
        self.plot_metrics(net)

    def plot_metrics(self, net):
        track = net.cbmanager.history.track
        line_idx = 0 if net._pass_type == "train" else 1

        # Change plot for loss line
        data = track[f"{net._pass_type}_loss"]
        self.axes[0, 0].lines[line_idx].set_data(np.arange(len(data)), data)
        self.axes[0, 0].relim()
        self.axes[0, 0].autoscale_view()

        # Change plot for other metric lines
        for i, name in enumerate(net._metrics.keys(), start=1):
            data = track[f"{net._pass_type}_{name}"]
            ax = self.axes[i // self.max_col, i % self.max_col]
            ax.lines[line_idx].set_data(np.arange(len(data)), data)
            ax.relim()
            ax.autoscale_view()

        self.force_draw()

    def on_fit_end(self, net):
        if self.savefig:
            self.fig.savefig(self.savename, bbox_inches="tight")
        if self.block_on_finish:
            plt.show(block=True)
        self.switch_normal(self.old_backend)

    def force_draw(self):
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def switch_qt5(self):
        if self.is_ipython:
            self.switch_magic("qt")
        else:
            self.switch_normal("Qt5Agg")

    def switch_normal(self, backend):
        mpl.use(backend, force=True)
        importlib.reload(sys.modules[self.pyplot_name])

    def switch_magic(self, backend):
        get_ipython().run_line_magic("matplotlib", backend)


class WeightCheckpoint(Callback):
    @property
    def tracked(self):
        return self._tally.recorded

    @property
    def mode(self):
        return self._tally.mode

    @property
    def best_epoch(self):
        return self._tally.best_epoch

    @property
    def savefile(self):
        return os.path.basename(self.savepath)

    @property
    def best_weights(self):
        if self._tally.best_weights is not None:
            return self._tally.best_weights
        elif self.savepath is not None:
            return torch.load(self.savepath)
        else:
            raise RuntimeError("There are no best_weights loaded in RAM or saved to disk.")  # TODO: Return None instead?

    def __init__(self, tracked, mode, savepath=None):
        super(WeightCheckpoint, self).__init__()
        self._tally = Tally(recorded=tracked, mode=mode, best_epoch=-1, best_weights=None)
        self.savepath = savepath

    def on_train_epoch_end(self, net):
        if not net._validate:
            self._track(net)

    def on_val_epoch_end(self, net):
        if net._validate:
            self._track(net)

    def _track(self, net):
        track = net.cbmanager.history.track
        new_record = track[self._tally.recorded][-1]
        self._tally.evaluate_record(new_record=new_record,
                                    best_epoch=net._epoch,
                                    best_weights=copy.deepcopy(net.module.state_dict()))

    def on_fit_end(self, net):
        if self.savepath is not None:
            if self._tally.best_weights is None:
                self._tally.best_weights = self.best_weights  # This can happen if we train the net a second time.
            torch.save(self._tally.best_weights, self.savepath)
            self._tally.best_weights = None  # No need to keep it in memory after saving.


class EarlyStopping(Callback):
    """
    Implements early stopping functionality to the added NeuralNetwork.
    It will monitor the given metric as `monitor` and if that metric does not improve
    `patience` times in a row, training will be stopped early.
    """
    def __init__(self, tracked: str, mode: str, patience: int = 20):
        super(EarlyStopping, self).__init__()
        self._tally = Tally(recorded=tracked, mode=mode, best_epoch=-1, best_weights=None)
        self.patience = patience
        self.current_patience = 0

    @property
    def tracked(self):
        return self._tally.recorded

    @property
    def mode(self):
        return self._tally.mode

    @property
    def best_epoch(self):
        return self._tally.best_epoch

    @property
    def best_weights(self):
        return self._tally.best_weights

    def on_train_epoch_end(self, net):
        if not net._validate:
            self._track(net)

    def on_val_epoch_end(self, net):
        if net._validate:
            self._track(net)

    def _track(self, net):
        track = net.cbmanager.history.track
        new_record = track[self._tally.recorded][-1]
        is_better_record = self._tally.is_better_record(new_record)

        if is_better_record:
            self._tally.evaluate_record(new_record=new_record,
                                        best_epoch=net._epoch,
                                        best_weights=copy.deepcopy(net.module.state_dict()))

            self.current_patience = 0
        else:
            self.current_patience += 1
            if self.current_patience >= self.patience:
                net.keep_training = False


class Tracker(Callback):
    @property
    def tracked(self):
        return self._tally.recorded

    @property
    def mode(self):
        return self._tally.mode

    @property
    def best_epoch(self):
        return self._tally.best_epoch

    """
    Tracks the given metric as `tracked`.
    """
    def __init__(self, tracked: str, mode: str):
        super(Tracker, self).__init__()
        self._tally = Tally(recorded=tracked, mode=mode, best_epoch=-1)

    def on_train_epoch_end(self, net):
        if not net._validate:
            self._track(net)

    def on_val_epoch_end(self, net):
        if net._validate:
            self._track(net)

    def _track(self, net):
        track = net.cbmanager.history.track
        self._tally.evaluate_record(track[self._tally.recorded][-1], best_epoch=net._epoch)


class Tally:
    """
    Tally a given record's best state.
    """
    def __init__(self, recorded: str, mode: str, **kwargs):
        self.__dict__.update(kwargs)
        self.recorded = recorded
        self.mode = mode
        self.best_record = -np.Inf if mode == "max" else np.Inf

    def is_better_record(self, new_record):
        if self.mode == "max":
            return new_record > self.best_record
        return new_record < self.best_record

    def evaluate_record(self, new_record, **kwargs):
        """
        Check if ``new_metric`` is better than ``self.best_metric``.
        If it is, update this class's properties with ``**kwargs``.
        """
        if self.is_better_record(new_record):
            self.best_record = new_record
            self.__dict__.update(**kwargs)


class CallbackInfo(Callback):
    """
    Collects and prints the ``neural_network`` parameters at the first time the callback function is called.
    Use this to get an intuition on which parameters will be available on each callback function.
    """
    def __init__(self):
        self.name = "CallbackInfo"
        self.called = np.zeros(15, dtype=bool)
        self.parameters = {}

    def on_fit_begin(self, net):
        if not self.called[0]:
            self.parameters["on_fit_begin"] = [k for k, v in net.__dict__.items() if v is not None]
            print("on_fit_begin: [")
            for k, v in net.__dict__.items():
                if v is not None:
                    print("\t", k)
            print("]")
            self.called[0] = True

    def on_fit_end(self, net):
        if not self.called[1]:
            self.parameters["on_fit_end"] = [k for k, v in net.__dict__.items() if v is not None]
            print("on_fit_end: [")
            for k, v in net.__dict__.items():
                if v is not None:
                    print("\t", k)
            print("]")
            self.called[1] = True

    def on_fit_interrupted(self, net):
        if not self.called[2]:
            self.parameters["on_fit_interrupted"] = [k for k, v in net.__dict__.items() if v is not None]
            print("on_fit_interrupted: [")
            for k, v in net.__dict__.items():
                if v is not None:
                    print("\t", k)
            print("]")
            self.called[2] = True

    def on_train_epoch_begin(self, net):
        if not self.called[3]:
            self.parameters["on_train_epoch_begin"] = [k for k, v in net.__dict__.items() if v is not None]
            print("on_train_epoch_begin: [")
            for k, v in net.__dict__.items():
                if v is not None:
                    print("\t", k)
            print("]")
            self.called[3] = True

    def on_train_epoch_end(self, net):
        if not self.called[4]:
            self.parameters["on_train_epoch_end"] = [k for k, v in net.__dict__.items() if v is not None]
            print("on_train_epoch_end: [")
            for k, v in net.__dict__.items():
                if v is not None:
                    print("\t", k)
            print("]")
            self.called[4] = True

    def on_train_batch_begin(self, net):
        if not self.called[5]:
            self.parameters["on_train_batch_begin"] = [k for k, v in net.__dict__.items() if v is not None]
            print("on_train_batch_begin: [")
            for k, v in net.__dict__.items():
                if v is not None:
                    print("\t", k)
            print("]")
            self.called[5] = True

    def on_train_batch_end(self, net):
        if not self.called[6]:
            self.parameters["on_train_batch_end"] = [k for k, v in net.__dict__.items() if v is not None]
            print("on_train_batch_end: [")
            for k, v in net.__dict__.items():
                if v is not None:
                    print("\t", k)
            print("]")
            self.called[6] = True

    def on_val_epoch_begin(self, net):
        if not self.called[7]:
            self.parameters["on_val_epoch_begin"] = [k for k, v in net.__dict__.items() if v is not None]
            print("on_val_epoch_begin: [")
            for k, v in net.__dict__.items():
                if v is not None:
                    print("\t", k)
            print("]")
            self.called[7] = True

    def on_val_epoch_end(self, net):
        if not self.called[8]:
            self.parameters["on_val_epoch_end"] = [k for k, v in net.__dict__.items() if v is not None]
            print("on_val_epoch_end: [")
            for k, v in net.__dict__.items():
                if v is not None:
                    print("\t", k)
            print("]")
            self.called[8] = True

    def on_val_batch_begin(self, net):
        if not self.called[9]:
            self.parameters["on_val_batch_begin"] = [k for k, v in net.__dict__.items() if v is not None]
            print("on_val_batch_begin: [")
            for k, v in net.__dict__.items():
                if v is not None:
                    print("\t", k)
            print("]")
            self.called[9] = True

    def on_val_batch_end(self, net):
        if not self.called[10]:
            self.parameters["on_val_batch_end"] = [k for k, v in net.__dict__.items() if v is not None]
            print("on_val_batch_end: [")
            for k, v in net.__dict__.items():
                if v is not None:
                    print("\t", k)
            print("]")
            self.called[10] = True

    def on_predict_begin(self, net):
        if not self.called[11]:
            self.parameters["on_predict_begin"] = [k for k, v in net.__dict__.items() if v is not None]
            print("on_predict_begin: [")
            for k, v in net.__dict__.items():
                if v is not None:
                    print("\t", k)
            print("]")
            self.called[11] = True

    def on_predict_end(self, net):
        if not self.called[12]:
            self.parameters["on_predict_end"] = [k for k, v in net.__dict__.items() if v is not None]
            print("on_predict_end: [")
            for k, v in net.__dict__.items():
                if v is not None:
                    print("\t", k)
            print("]")
            self.called[12] = True

    def on_predict_proba_begin(self, net):
        if not self.called[13]:
            self.parameters["on_predict_proba_begin"] = [k for k, v in net.__dict__.items() if v is not None]
            print("on_predict_proba_begin: [")
            for k, v in net.__dict__.items():
                if v is not None:
                    print("\t", k)
            print("]")
            self.called[13] = True

    def on_predict_proba_end(self, net):
        if not self.called[14]:
            self.parameters["on_predict_proba_end"] = [k for k, v in net.__dict__.items() if v is not None]
            print("on_predict_proba_end: [")
            for k, v in net.__dict__.items():
                if v is not None:
                    print("\t", k)
            print("]")
            self.called[14] = True