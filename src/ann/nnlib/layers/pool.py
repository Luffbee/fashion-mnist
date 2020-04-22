from typing import Tuple, Dict
from math import ceil

import numpy as np

from ..base import NeuralLayer, NeuralLayerFactory


class Pool2D(NeuralLayer):
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
        net = np.zeros((len(X), *self.osz))
        func = np.max if self.kind == 'max' else np.mean

        ro, co, cho = self.osz
        for i in range(ro):
            r_start = i*self.k[0]
            r_end = r_start+self.k[0]
            for j in range(co):
                c_start = j*self.k[1]
                c_end = c_start+self.k[1]
                net[:, i, j, :] = \
                    func(X[:, r_start:r_end, c_start:c_end, :], axis=(1, 2))

        return net.reshape((len(X), ro*co*cho))

    def train(self, dE: np.ndarray, net: np.ndarray, X: np.ndarray) -> np.ndarray:
        dE = dE.reshape((len(X), *self.osz))
        X = X.reshape((len(X), *self.isz))
        dE_in = np.zeros((len(X), *self.isz))

        ro, co = self.osz[:2]
        for i in range(ro):
            r_start = i*self.k[0]
            r_end = r_start+self.k[0]
            for j in range(co):
                c_start = j*self.k[1]
                c_end = c_start+self.k[1]
                block = dE_in[:, r_start:r_end, c_start:c_end, :]
                if self.kind == 'max':
                    block[:, :, :, :] = \
                        self.max_mask(X[:, r_start:r_end, c_start:c_end, :]) * \
                        dE[:, i:i+1, j:j+1, :]
                elif self.kind == 'mean':
                    block[:, :, :, :] = \
                        dE[:, i:i+1, j:j+1, :] / \
                        (block.shape[1] * block.shape[2])

        ri, ci, chi = self.isz
        return dE_in.reshape((len(X), ri*ci*chi))

    @staticmethod
    def max_mask(X: np.ndarray) -> np.ndarray:
        blk, r, c, ch = X.shape
        mask = np.zeros((blk, r, c, ch))
        idx = np.argmax(X.reshape((blk, r*c, ch)), axis=1)
        ax0, ax3 = np.indices((blk, ch))
        mask.reshape((blk, r*c, ch))[ax0, idx, ax3] = 1
        return mask


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
