from typing import List

import numpy as np

from .base import NeuralNet
from .model import Sequential
from .layers import FCFactory
from .activation import Sigmoid


def buildBP(isize: int, layers_size: List[int], eta: float) -> NeuralNet:
    bp = Sequential(np.array([isize]))
    for sz in layers_size:
        bp.add(FCFactory(sz, Sigmoid(), eta))
    return bp
