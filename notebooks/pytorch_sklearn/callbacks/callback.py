class Callback:
    """
    Base class for all ``Callback``s.
    Every callback gets the neural network it is attached to as the first parameter of its callback functions.
    """
    def __init__(self):
        self.name = None  # provide an informative name

    # Fit
    def on_fit_begin(self, net):
        pass

    def on_fit_end(self, net):
        pass

    def on_fit_interrupted(self, net):
        pass

    # Gradient
    def on_grad_compute_begin(self, net):
        pass

    def on_grad_compute_end(self, net):
        pass

    # Train
    def on_train_epoch_begin(self, net):
        pass

    def on_train_epoch_end(self, net):
        pass

    def on_train_batch_begin(self, net):
        pass

    def on_train_batch_end(self, net):
        pass

    # Validation
    def on_val_epoch_begin(self, net):
        pass

    def on_val_epoch_end(self, net):
        pass

    def on_val_batch_begin(self, net):
        pass

    def on_val_batch_end(self, net):
        pass

    # Prediction
    def on_predict_begin(self, net):
        pass

    def on_predict_end(self, net):
        pass

    def on_predict_proba_begin(self, net):
        pass

    def on_predict_proba_end(self, net):
        pass
