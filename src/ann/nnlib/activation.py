from typing import Optional
from abc import ABCMeta, abstractmethod
import random

import numpy as np
from numpy import ndarray
from scipy.special import softmax, expit  # pylint: disable=E0611


class ActivationFunc(metaclass=ABCMeta):
    @abstractmethod
    def f(self, net: ndarray) -> ndarray:
        pass

    @abstractmethod
    def delta(self, dE: ndarray, net: ndarray, out: Optional[ndarray] = None) -> ndarray:
        pass


class SimpleActivation(ActivationFunc):
    @abstractmethod
    def f(self, net: ndarray) -> ndarray:
        pass

    @abstractmethod
    def df(self, net: ndarray, out: Optional[ndarray] = None) -> ndarray:
        pass

    def delta(self, dE: ndarray, net: ndarray, out: Optional[ndarray] = None) -> ndarray:
        return dE * self.df(net, out=out)


class Sigmoid(SimpleActivation):
    def f(self, net: ndarray) -> ndarray:
        return expit(net)

    def df(self, net: ndarray, out: Optional[ndarray] = None) -> ndarray:
        if out is None:
            out = expit(net)
        return out * (1 - out)


class ReLU(SimpleActivation):
    def f(self, net: ndarray) -> ndarray:
        return np.maximum(net, 0)

    def df(self, net: ndarray, out: Optional[ndarray] = None) -> ndarray:
        return np.where(net > 0, 1, 0)


class LeakyReLU(SimpleActivation):
    def __init__(self, a: float) -> None:
        self.a = a

    def f(self, net: ndarray) -> ndarray:
        return np.where(net > 0, net, net*self.a)

    def df(self, net: ndarray, out: Optional[ndarray] = None) -> ndarray:
        return np.where(net > 0, 1, self.a)


class RReLU(SimpleActivation):
    def __init__(self, l: float, u: float) -> None:
        self.a = random.random() * (u - l) + l

    def f(self, net: ndarray) -> ndarray:
        return np.where(net > 0, net, net*self.a)

    def df(self, net: ndarray, out: Optional[ndarray] = None) -> ndarray:
        return np.where(net > 0, 1, self.a)


class Tanh(SimpleActivation):
    def f(self, net: ndarray) -> ndarray:
        return np.tanh(net)

    def df(self, net: ndarray, out: Optional[ndarray] = None) -> ndarray:
        if out is None:
            out = np.tanh(net)
        return 1 - out**2


class Softmax(ActivationFunc):
    def f(self, net: ndarray) -> ndarray:
        out = softmax(net, axis=1)
        return out

    def delta(self, dE: ndarray, net: ndarray, out: Optional[ndarray] = None) -> ndarray:
        # if out is None:
        #    out = self.f(net)
        # only consider softmax cross-entropy loss
        #y = -dE * out
        # return y - out
        return dE
