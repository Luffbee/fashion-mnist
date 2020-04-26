from typing import Tuple, Callable
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
        self.activation = activation
        self.eta = eta
        # self.W = np.random.rand(
        #    self.k[0], self.k[1], isize[2], channel) * 2 - 1
        #self.bias = np.random.rand(1, 1, 1, channel) * 2 - 1
        self.W = np.random.randn(
            self.k[0], self.k[1], isize[2], channel)
        self.dW = np.zeros_like(self.W)
        self.dW_tmp = np.zeros_like(self.W)
        self.bias = np.random.randn(1, 1, 1, channel)
        self.dbias = np.zeros_like(self.bias)

    def isize(self) -> np.ndarray:
        return np.array(self.isz)

    def osize(self) -> np.ndarray:
        return np.array(self.osz)

    def activate(self, net: np.ndarray) -> np.ndarray:
        assert np.isfinite(net).all()
        return self.activation.f(net)

    def update_eta(self, update: Callable[[float], float]) -> None:
        self.eta = update(self.eta)

    def net(self, X: np.ndarray) -> np.ndarray:
        assert np.isfinite(X).all()
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
            net_mult = np.zeros((len(X), *self.W.shape))
            for i in range(ro):
                r_start = i
                r_end = i+self.k[0]
                for j in range(co):
                    c_start = j
                    c_end = j+self.k[1]
                    np.multiply(
                        X[:, r_start:r_end, c_start:c_end, :, np.newaxis],
                        self.W[np.newaxis, :, :, :, :],
                        out=net_mult)
                    np.sum(net_mult, axis=(1, 2, 3), out=net[:, i, j, :])
        assert np.isfinite(net).all()
        net += self.bias
        assert np.isfinite(net).all()
        return net.reshape((len(X), ro*co*cho))

    def train(self, dE: np.ndarray, out: np.ndarray, net: np.ndarray, X: np.ndarray) -> np.ndarray:
        delta = self.activation.delta(
            dE, net, out=out).reshape((len(X), *self.osz))
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
            dE_mult = np.zeros((len(X), *self.W.shape))
            dE_tmp = np.zeros((len(X), *self.k, self.isz[2]))
            self.dW.fill(0)
            dW_mult = np.zeros((len(X), *self.W.shape))
            for i in range(ro):
                r_start = i
                r_end = i+self.k[0]
                for j in range(co):
                    c_start = j
                    c_end = j+self.k[1]

                    np.multiply(
                        self.W[np.newaxis, :, :, :, :],
                        delta[:, i:i+1, j:j+1, np.newaxis, :],
                        out=dE_mult)
                    np.sum(dE_mult, axis=4, out=dE_tmp)
                    dE_in[:, r_start:r_end, c_start:c_end, :] += dE_tmp

                    np.multiply(
                        X[:, r_start:r_end, c_start:c_end, :, np.newaxis],
                        delta[:, i:i+1, j:j+1, np.newaxis, :],
                        out=dW_mult)
                    np.sum(dW_mult, axis=0, out=self.dW_tmp)
                    self.dW += self.dW_tmp

        eta = self.eta / len(X)
        np.multiply(self.dW, eta, out=self.dW)
        self.W += self.dW

        np.sum(delta, axis=(0, 1, 2), out=self.dbias.reshape(delta.shape[3:]))
        np.multiply(self.dbias, eta, out=self.dbias)
        self.bias += self.dbias

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
