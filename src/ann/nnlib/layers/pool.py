from typing import Tuple, Callable

import numpy as np
from skimage.util import view_as_blocks

from ..base import NeuralLayer, NeuralLayerFactory


class Pool2D(NeuralLayer):
    def __init__(self, isize: Tuple[int, int, int], pool_size: Tuple[int, int], kind: str) -> None:
        self.isz = isize
        assert isize[0] % pool_size[0] == 0
        assert isize[1] % pool_size[1] == 0
        self.osz = (isize[0] // pool_size[0],
                    isize[1] // pool_size[1],
                    isize[2])
        self.k = pool_size
        self.kind = kind
        if self.kind == 'max':
            self.func = np.max
            self.train_func = self.train_max
        elif self.kind == 'mean':
            self.func = np.mean
            self.train_func = self.train_mean
        else:
            raise NotImplementedError('invalid type of pooling')

    def isize(self) -> np.ndarray:
        return np.array(self.isz)

    def osize(self) -> np.ndarray:
        return np.array(self.osz)

    def activate(self, net: np.ndarray) -> np.ndarray:
        return net

    def update_eta(self, update: Callable[[float], float]) -> None:
        pass

    def net(self, X: np.ndarray) -> np.ndarray:
        X = X.reshape((len(X), *self.isz))
        blocks = view_as_blocks(X, (1, *self.k, 1))
        net = self.func(blocks, axis=(4, 5, 6, 7))
        ro, co, cho = self.osz
        return net.reshape((len(X), ro*co*cho))

    def train_max(self, dE: np.ndarray, X: np.ndarray) -> np.ndarray:
        dE = dE.reshape((len(X), *self.osz))
        X = X.reshape((len(X), *self.isz))
        dE_in = np.zeros((len(X), *self.isz))
        np.multiply(self.max_mask(view_as_blocks(X, (1, *self.k, 1))),
                    view_as_blocks(dE, (1, 1, 1, 1)),
                    out=view_as_blocks(dE_in, (1, *self.k, 1)))
        return dE_in

    @staticmethod
    def max_mask(X: np.ndarray) -> np.ndarray:
        blk, rb, cb, ch, _, k0, k1, _ = X.shape
        r, c = blk*rb*cb*ch, k0*k1
        idx = np.argmax(X.reshape((r, c)), axis=1)
        mask = np.zeros((r, c))
        mask[np.arange(r), idx] = 1
        return mask.reshape(X.shape)

    def train_mean(self, dE: np.ndarray, X: np.ndarray) -> np.ndarray: # pylint: disable=W0613
        dE = dE.reshape((len(dE), *self.osz))
        dE_in = np.zeros((len(dE), *self.isz))
        np.divide(view_as_blocks(dE, (1, 1, 1, 1)),
                  self.k[0]*self.k[1],
                  out=view_as_blocks(dE_in, (1, *self.k, 1)))
        return dE_in

    def train(self, dE: np.ndarray, out: np.ndarray, net: np.ndarray, X: np.ndarray) -> np.ndarray:
        ri, ci, chi = self.isz
        return self.train_func(dE, X).reshape((len(X), ri*ci*chi))


class Pool2DFactory(NeuralLayerFactory):
    def __init__(self, pool_size: Tuple[int, int], kind: str) -> None:
        self.k = pool_size
        self.kind = kind

    def set_isize(self, isize: np.ndarray) -> NeuralLayer:
        assert len(isize.shape) == 1
        assert len(isize) == 3
        assert isize.dtype == int
        r, c, ch = isize
        return Pool2D(isize=(r, c, ch),
                      pool_size=self.k,
                      kind=self.kind)
