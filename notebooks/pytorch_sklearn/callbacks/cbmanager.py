from pytorch_sklearn.callbacks import Verbose, History, CallbackInfo
from list_utilities import make_unique_fast


class CallbackManager:
    def __init__(self, callbacks=None):
        self._callbacks = None
        self._history = History()
        self._default_callbacks = [
            self._history,  # History must come first, as other callbacks can depend on it.
        ]

        if callbacks is not None:
            self.callbacks = callbacks

    @property
    def history(self):
        return self._history

    @property
    def callbacks(self):
        if self._callbacks is not None:
            return self._default_callbacks + self._callbacks
        else:
            return self._default_callbacks

    @callbacks.setter
    def callbacks(self, cbs):
        self._callbacks = cbs
        self._name_unique(self._callbacks)

    def _name_unique(self, cbs):
        names = []
        for callback in cbs:
            if callback.name is None:
                names.append(type(callback).__name__)
            else:
                names.append(callback.name)

        names = make_unique_fast(names)
        i = 0
        for callback in cbs:
            callback.name = names[i]
            i += 1
