from typing import List, Dict, Callable
import time

import numpy as np
from numpy import ndarray
import pandas as pd
from utils import mnist_reader

from nnlib.base import NeuralNet
from nnlib.bp import buildBP  # pylint: disable=W0611
from nnlib.model import Sequential  # pylint: disable=W0611
from nnlib.layers import FCFactory, Conv2DFactory, Pool2DFactory  # pylint: disable=W0611
from nnlib.activation import Softmax, Sigmoid, ReLU, RReLU, Tanh  # pylint: disable=W0611

CLS_LEN = 10
CLASSES = list(range(CLS_LEN))
CLASSE_NAMES = [
    'T-shirt/top',
    'Trouser',
    'Pullover',
    'Dress',
    'Coat',
    'Sandal',
    'Shirt',
    'Sneaker',
    'Bag',
    'Ankle boot',
]


def accuracy(res: np.ndarray) -> float:
    good = res.diagonal().sum()
    tot = res.sum()
    if tot == 0:
        return -1
    return good / tot


def precision(cls: int, res: np.ndarray) -> float:
    good = res[cls][cls]
    tot = res[:, cls].sum()
    if tot == 0:
        return -1
    return good / tot


def recall(cls: int, res: np.ndarray) -> float:
    good = res[cls][cls]
    tot = res[cls].sum()
    if tot == 0:
        return -1
    return good / tot


def verify(clsfer: NeuralNet, X: ndarray, Y: ndarray, detail: bool = False):
    run_start = time.time()
    r_poss = clsfer.calc(X)
    run_end = time.time()
    run_time = run_end - run_start
    r = np.argmax(r_poss, axis=1)
    result = np.zeros((10, 10))
    for i in range(len(Y)):
        result[Y[i]][r[i]] += 1

    print('============================')
    print(f'data set size: {len(X)}, classify time: {run_time}s')
    print('accuracy: ', accuracy(result))
    if detail:
        analysis: Dict[str, List[float]] = {
            'precision': [],
            'recall': [],
        }
        for cls in range(len(CLASSES)):
            analysis['precision'].append(precision(cls, result))
            analysis['recall'].append(recall(cls, result))
        print(pd.DataFrame(data=analysis, columns=[
            'precision', 'recall'], index=CLASSES))
        print('')
        print('row name: tag, colum name: result')
        print(pd.DataFrame(result, columns=CLASSES, index=[
            f'{CLASSES[i]} {CLASSE_NAMES[i]}' for i in range(len(CLASSES))]))


def label2Y(label: ndarray) -> ndarray:
    Yo = np.zeros((len(label), CLS_LEN))
    Yo[np.arange(len(label)), label] = 1
    return Yo


def preprocess(X: ndarray) -> ndarray:
    mx = np.max(X, axis=1, keepdims=True)
    return X / mx


def exp_eta(a: float) -> Callable[[float], float]:
    return lambda x: x * a


def main() -> None:
    Tx, Ty = mnist_reader.load_mnist('../../data/fashion', kind='train')
    Vx, Vy = mnist_reader.load_mnist('../../data/fashion', kind='t10k')

    Tx = preprocess(Tx)
    Tyy = label2Y(Ty)
    Vx = preprocess(Vx)
    # model = buildBP(Tx.shape[1], [128, 256, 64, CLS_LEN],
    #                4e-1)  # pylint: disable=E1136
    model = Sequential(np.array([28, 28, 1]))
    model.add(Conv2DFactory((5, 5), 4, RReLU(0.02, 0.02), 1e-3))
    model.add(Pool2DFactory((2, 2), 'max'))
    model.add(FCFactory(128, Sigmoid(), 4e-1))
    model.add(FCFactory(256, Sigmoid(), 4e-1))
    model.add(FCFactory(64, Sigmoid(), 4e-1))
    model.add(FCFactory(CLS_LEN, Softmax(), 1e-2))

    for layer in model.layers:
        print(layer.isize().prod(), layer.osize().prod())

    train_time = 0.0
    #sample_size = len(Tx)
    blk = 1
    for i in range(50):
        idx = np.random.permutation(len(Tx))
        #idx = np.random.randint(len(Tx), size=sample_size)
        # idx.sort()
        train_start = time.time()
        model.trainBGD(Tx[idx], Tyy[idx], blk)
        train_end = time.time()
        print(f'------ epoch {i} -------')
        print(f'blk: {blk}')
        print(f'training time: {train_end-train_start}s')
        # print(model.calc(Tx[:5]).round(2))
        # print(Tyy[:5])
        train_time += train_end - train_start
        verify(model, Vx, Vy)  # , True)
        model.update_eta(exp_eta(0.9))
        blk = min(blk + 2, 100)
    print(f'training time: {train_time}s')
    print('verify on Tx')
    verify(model, Tx, Ty)
    print('verify on Vx')
    verify(model, Vx, Vy)


if __name__ == '__main__':
    main()
