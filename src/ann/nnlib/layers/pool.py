from typing import Tuple, Dict
from math import ceil

import numpy as np

from ..base import NeuralLayer, NeuralLayerFactory


class MaxPool2D(NeuralLayer):
    def __init__(self, isize: Tuple[int, int, int], pool_size: Tuple[int, int], kind: str) -> None:
        self.isz = isize
        self.osz = (ceil(isize[0] / pool_size[0]),
                    ceil(isize[1] / pool_size[1]),
                    isize[2])
        self.k = pool_size
        if kind not in ['max', 'mean']:
            raise NotImplementedError('invalid type of pooling')
        self.kind = kind
        self.cache: Dict[str, np.ndarray] = {}

    def isize(self) -> np.ndarray:
        return np.array(self.isz)

    def osize(self) -> np.ndarray:
        return np.array(self.osz)

    def activate(self, net: np.ndarray) -> np.ndarray:
        return net

    def dactivate(self, net: np.ndarray) -> np.ndarray:
        return np.ones(net.shape)

    def set_eta(self, eta: float) -> None:
        pass

    def net(self, X: np.ndarray) -> np.ndarray:
        X = X.reshape((len(X), *self.isz))
