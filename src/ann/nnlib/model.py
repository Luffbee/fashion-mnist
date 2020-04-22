from typing import List
import numpy as np

from .base import NeuralNet, NeuralLayerFactory, NeuralLayer


class Sequential(NeuralNet):
    def __init__(self, isize: np.ndarray) -> None:
        self.isz = isize
        self.layers: List[NeuralLayer] = []

    def add(self, factory: NeuralLayerFactory) -> None:
        isize = self.isz
        if len(self.layers) > 0:
            isize = self.layers[-1].osize()
        layer = factory.set_isize(isize)
        self.layers.append(layer)

    def isize(self) -> np.ndarray:
        return self.isz

    def osize(self) -> np.ndarray:
        if len(self.layers) == 0:
            return np.array([0])
        else:
            return self.layers[-1].osize()

    def calc(self, X: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            X, _ = layer.calc(X)
        return X

    def set_eta(self, eta: float):
        for layer in self.layers:
            layer.set_eta(eta)

    def train(self, X: np.ndarray, Y: np.ndarray) -> None:
        save = []
        for layer in self.layers:
            out, net = layer.calc(X)
            save.append((X, net))
            X = out
        assert Y.shape == X.shape
        dE = Y - X
        for layer in reversed(self.layers):
            X, net = save.pop()
            dE = layer.train(dE, net, X)
