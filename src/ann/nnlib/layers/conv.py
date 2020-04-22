from typing import Tuple
import numpy as np
from scipy.signal import correlate2d

from ..base import NeuralLayer, NeuralLayerFactory
from ..activation import ActivationFunc


class Conv2D(NeuralLayer):
    def __init__(self,
                 isize: Tuple[int, int, int],
                 kernel_size: Tuple[int, int],
                 channel: int,
                 activation: ActivationFunc,
                 eta: float) -> None:
        self.isz = isize
        self.osz = (isize[0] - kernel_size[0] + 1,
                    isize[1] - kernel_size[1] + 1,
                    channel)
        self.k = kernel_size
        self.f = activation
        self.eta = eta
        self.W = np.random.rand(
            self.k[0], self.k[1], isize[2], channel) * 2 - 1
        self.bias = np.random.rand(1, 1, 1, channel)

    def isize(self) -> np.ndarray:
        return np.array(self.isz)

    def osize(self) -> np.ndarray:
        return np.array(self.osz)

    def activate(self, net: np.ndarray) -> np.ndarray:
        return self.f.call(net)

    def dactivate(self, net: np.ndarray) -> np.ndarray:
        return self.f.derivative(net)

    def set_eta(self, eta: float) -> None:
        self.eta = eta

    def net(self, X: np.ndarray) -> np.ndarray:
        X = X.reshape((len(X), *self.isz))
        net = np.zeros((len(X), *self.osz))
        chi = self.isz[2]
        ro, co, cho = self.osz
        if len(X)*cho*chi < ro * co:
            for i in range(len(X)):
                for j in range(cho):
                    for k in range(chi):
                        net[i, :, :, j] += correlate2d(
                            X[i, :, :, k],
                            self.W[:, :, k, j],
                            mode='valid')
        else:
            for i in range(ro):
                r_start = i
                r_end = i+self.k[0]
                for j in range(co):
                    c_start = j
                    c_end = j+self.k[1]
                    net[:, i, j, :] = np.sum(
                        X[:, r_start:r_end, c_start:c_end, :, np.newaxis] *
                        self.W[np.newaxis, :, :, :, :],
                        axis=(1, 2, 3))
        net += self.bias
        return net.reshape((len(X), ro*co*cho))

    def train(self, dE: np.ndarray, net: np.ndarray, X: np.ndarray) -> np.ndarray:
        delta = self.delta(dE, net).reshape((len(X), *self.osz))
        X = X.reshape((len(X), *self.isz))

        dE_in = np.zeros((len(X), *self.isz))
        dW = np.zeros(self.W.shape)

        ri, ci, chi = self.isz
        ro, co, cho = self.osz

        if len(X) * chi * cho < ro * co:
            Wrot180 = np.rot90(self.W, 2, axes=(0, 1))
            for i in range(len(X)):
                for j in range(chi):
                    for k in range(cho):
                        dE_in[i, :, :, j] += \
                            correlate2d(
                                delta[i, :, :, k],
                                Wrot180[:, :, j, k],
                                mode='full')
                        dW[:, :, j, k] += \
                            correlate2d(
                                X[i, :, :, j],
                                delta[i, :, :, k],
                                mode='valid')
        else:
            for i in range(ro):
                r_start = i
                r_end = i+self.k[0]
                for j in range(co):
                    c_start = j
                    c_end = j+self.k[1]
                    dE_in[:, r_start:r_end, c_start:c_end, :] += \
                        np.sum(self.W[np.newaxis, :, :, :, :] *
                               delta[:, i:i+1, j:j+1, np.newaxis, :],
                               axis=4)
                    dW += np.sum(
                        X[:, r_start:r_end, c_start:c_end, :, np.newaxis] *
                        delta[:, i:i+1, j:j+1, np.newaxis, :],
                        axis=0)

        self.W += self.eta * dW
        self.bias += self.eta * delta.sum(axis=(0, 1, 2))
        return dE_in.reshape((len(X), ri*ci*chi))


class Conv2DFactory(NeuralLayerFactory):
    def __init__(self, kernel_size: Tuple[int, int], channel: int,
                 activation: ActivationFunc,
                 eta: float) -> None:
        self.k = kernel_size
        self.ch = channel
        self.f = activation
        self.eta = eta

    def set_isize(self, isize: np.ndarray) -> NeuralLayer:
        assert len(isize.shape) == 1
        assert len(isize) == 3
        assert isize.dtype == int
        r, c, ch = isize
        return Conv2D(isize=(r, c, ch),
                      kernel_size=self.k, channel=self.ch,
                      activation=self.f,
                      eta=self.eta)
