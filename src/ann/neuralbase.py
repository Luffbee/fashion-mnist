from typing import Tuple
from abc import ABCMeta, abstractmethod

import numpy as np
from numpy import ndarray

from activationfunc import ActivationFunc


class NeuralLayer(metaclass=ABCMeta):
    @abstractmethod
    def isize(self) -> int:
        pass

    @abstractmethod
    def osize(self) -> int:
        pass

    @abstractmethod
    def weight(self) -> ndarray:
        pass

    @abstractmethod
    def activate(self, net: ndarray) -> ndarray:
        pass

    @abstractmethod
    def dactivate(self, net: ndarray) -> ndarray:
        pass

    @abstractmethod
    def set_eta(self, eta: float) -> None:
        pass

    @abstractmethod
    def adjust(self, dW: ndarray) -> None:
        pass

    def net(self, X: ndarray) -> ndarray:
        W = self.weight()
        assert len(X.shape) == 2
        return W.dot(X)

    def calc(self, X: ndarray) -> Tuple[ndarray, ndarray]:
        net = self.net(X)
        out = self.activate(net)
        return (out, net)

    def delta(self, dE: ndarray, net: ndarray) -> ndarray:
        assert dE.shape == net.shape
        return dE * self.dactivate(net)

    def dweight(self, delta: ndarray, X: ndarray) -> ndarray:
        return delta.dot(X.T)

    def derror(self, delta: ndarray) -> ndarray:
        W = self.weight()
        return W.T.dot(delta)


class NeuralNet(metaclass=ABCMeta):
    @abstractmethod
    def isize(self) -> int:
        pass

    @abstractmethod
    def osize(self) -> int:
        pass

    @abstractmethod
    def calc(self, X: ndarray) -> ndarray:
        pass

    @abstractmethod
    def set_eta(self, eta: float) -> None:
        pass

    @abstractmethod
    def train(self, X: ndarray, Y: ndarray) -> None:
        pass

    def trainBGD(self, X: ndarray, Y: ndarray, batch: int) -> None:
        for i in range(0, X.shape[1], batch):
            self.train(X[:, i:(i+batch)], Y[:, i:(i+batch)])

    def trainSGD(self, X: ndarray, Y: ndarray) -> None:
        self.trainBGD(X, Y, 1)


class FullyConnectedNL(NeuralLayer):
    def __init__(self, isize: int, osize: int, f: ActivationFunc, eta: float) -> None:
        self.W = np.random.rand(osize, isize) * 2 - 1
        self.f = f
        self.eta = eta

    def isize(self) -> int:
        return self.W.shape[1]

    def osize(self) -> int:
        return self.W.shape[0]

    def weight(self) -> ndarray:
        return self.W.copy()

    def activate(self, net: ndarray) -> ndarray:
        return self.f.call(net)

    def dactivate(self, net: ndarray) -> ndarray:
        return self.f.derivative(net)
    
    def set_eta(self, eta: float) -> None:
        self.eta = eta

    def adjust(self, dW: ndarray) -> None:
        assert self.W.shape == dW.shape
        self.W += self.eta * dW


