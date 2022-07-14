from pytorch_sklearn.neural_network import NeuralNetwork
import torch
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer as _Optimizer


class L1L2NeuralNet(NeuralNetwork):
    def __init__(self, module: torch.nn.Module, optimizer: _Optimizer, criterion: _Loss, lambda1: float, lambda2: float):
        super(L1L2NeuralNet, self).__init__(module, optimizer, criterion)
        self.lambda1 = lambda1
        self.lambda2 = lambda2

    def get_loss(self, y_pred, y_true):
        crit_loss = self.criterion(y_pred, y_true)
        params = torch.cat([x.view(-1) for x in self.module.parameters()])
        l1_reg = self.lambda1 * torch.norm(params, 1)
        l2_reg = self.lambda2 * torch.sum(params ** 2)
        return crit_loss + l1_reg + l2_reg
