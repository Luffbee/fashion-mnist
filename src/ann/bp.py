from typing import List

import numpy as np
from numpy import ndarray

from neuralbase import NeuralNet, FullyConnectedNL
from activationfunc import Sigmoid

class BP(NeuralNet):
    def __init__(self, isize: int, layers_size: List[int], eta: float) -> None:
        self.layers = []
        isize = isize+1  # 1 bias
        for osize in layers_size:
            layer = FullyConnectedNL(isize, osize, Sigmoid(), eta)
            self.layers.append(layer)
            isize = osize + 1  # 1 bias

    def isize(self) -> int:
        return self.layers[0].isize()-1

    def osize(self) -> int:
        return self.layers[-1].osize()

    def calc(self, X: ndarray) -> ndarray:
        ones = np.repeat(1.0, X.shape[1])
        for layer in self.layers:
            X = np.row_stack((X, ones))
            X, _ = layer.calc(X)
        return X

    def set_eta(self, eta: float) -> None:
        for layer in self.layers:
            layer.set_eta(eta)

    def train(self, X: ndarray, Y: ndarray) -> None:
        res = []
        ones = np.repeat(1.0, X.shape[1])
        for layer in self.layers:
            X = np.row_stack((X, ones))
            out, net = layer.calc(X)
            res.append((X, net))
            X = out
        assert Y.shape == X.shape
        dE = Y - X
        for layer in reversed(self.layers):
            X, net = res.pop()
            delta = layer.delta(dE, net)
            dW = layer.dweight(delta, X)
            dE = layer.derror(delta)[:-1]
            layer.adjust(dW)
