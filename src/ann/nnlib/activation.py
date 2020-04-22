from abc import ABCMeta, abstractmethod

import numpy as np
from numpy import ndarray

class ActivationFunc(metaclass=ABCMeta):
    @abstractmethod
    def call(self, X: ndarray) -> ndarray:
        pass

    @abstractmethod
    def derivative(self, X: ndarray) -> ndarray:
        pass

class Sigmoid(ActivationFunc):
    def call(self, X: ndarray) -> ndarray:
        return 1 / (1 + np.exp(-X))

    def derivative(self, X: ndarray) -> ndarray:
        exp = np.exp(-X)
        exp1 = exp + 1
        return exp / (exp1 * exp1)
