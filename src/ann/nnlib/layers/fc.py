from typing import Callable
import numpy as np

from ..base import NeuralLayer, NeuralLayerFactory
from ..activation import ActivationFunc


class FC(NeuralLayer):
    def __init__(self, isize: int, osize: int,
                 activation: ActivationFunc,
                 eta: float) -> None:
        #self.W = np.random.rand(isize+1, osize) * 2 - 1
        self.W = np.random.randn(isize+1, osize)
        self.dW = np.zeros_like(self.W)
        self.activation = activation
        self.eta = eta

    def isize(self) -> np.ndarray:
        return np.array([self.W.shape[0]-1])

    def osize(self) -> np.ndarray:
        return np.array([self.W.shape[1]])

    def activate(self, net: np.ndarray) -> np.ndarray:
        return self.activation.f(net)

    def update_eta(self, update: Callable[[float], float]) -> None:
        self.eta = update(self.eta)

    def net(self, X: np.ndarray) -> np.ndarray:
        assert len(X.shape) == 2
        X = np.column_stack((X, np.ones(len(X))))
        return X.dot(self.W)

    def train(self, dE: np.ndarray, out: np.ndarray, net: np.ndarray, X: np.ndarray) -> np.ndarray:
        delta = self.activation.delta(dE, net, out=out)
        dE_in = delta.dot(self.W[:-1].T)
        X = np.column_stack((X, np.ones(len(X))))
        X.T.dot(delta, out=self.dW)
        np.multiply(self.dW, self.eta / len(X), out=self.dW)
        self.W += self.dW
        return dE_in


class FCFactory(NeuralLayerFactory):
    def __init__(self, osize: int,
                 activation: ActivationFunc,
                 eta: float) -> None:
        self.osz = osize
        self.f = activation
        self.eta = eta

    def set_isize(self, isize: np.ndarray) -> FC:
        return FC(isize=isize.prod(), osize=self.osz,
                  activation=self.f,
                  eta=self.eta)
