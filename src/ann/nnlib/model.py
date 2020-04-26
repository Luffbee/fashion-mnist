from typing import List, Callable
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

    def update_eta(self, update: Callable[[float], float]):
        for layer in self.layers:
            layer.update_eta(update)

    def train(self, X: np.ndarray, Y: np.ndarray) -> None:
        save = []
        for layer in self.layers:
            out, net = layer.calc(X)
            assert np.isfinite(out).all()
            assert np.isfinite(net).all()
            save.append((X, net, out))
            X = out
        assert Y.shape == X.shape
        dE = Y - X
        for layer in reversed(self.layers):
            X, net, out = save.pop()
            dE = layer.train(dE, out, net, X)
            assert np.isfinite(dE).all()
