from typing import Tuple, Callable
from abc import ABCMeta, abstractmethod

import numpy as np


class NeuralLayer(metaclass=ABCMeta):
    @abstractmethod
    def isize(self) -> np.ndarray:
        pass

    @abstractmethod
    def osize(self) -> np.ndarray:
        pass

    @abstractmethod
    def activate(self, net: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def update_eta(self, update: Callable[[float], float]) -> None:
        pass

    @abstractmethod
    def net(self, X: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def train(self, dE: np.ndarray, out: np.ndarray, net: np.ndarray, X: np.ndarray) -> np.ndarray:
        pass

    def calc(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        assert np.isfinite(X).all()
        net = self.net(X)
        assert np.isfinite(net).all()
        out = self.activate(net)
        return (out, net)


class NeuralNet(metaclass=ABCMeta):
    @abstractmethod
    def isize(self) -> np.ndarray:
        pass

    @abstractmethod
    def osize(self) -> np.ndarray:
        pass

    @abstractmethod
    def calc(self, X: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def update_eta(self, update: Callable[[float], float]) -> None:
        pass

    @abstractmethod
    def train(self, X: np.ndarray, Y: np.ndarray) -> None:
        pass

    def trainBGD(self, X: np.ndarray, Y: np.ndarray, batch: int) -> None:
        for i in range(0, len(X), batch):
            self.train(X[i:(i+batch)], Y[i:(i+batch)])

    def trainSGD(self, X: np.ndarray, Y: np.ndarray) -> None:
        self.trainBGD(X, Y, 1)


class NeuralLayerFactory(metaclass=ABCMeta):
    @abstractmethod
    def set_isize(self, isize: Tuple[int, int]) -> NeuralLayer:
        pass
