from typing import List, Dict
import time

import numpy as np
from numpy import ndarray
import pandas as pd
from utils import mnist_reader

from nnlib.base import NeuralNet
from nnlib.bp import buildBP  # pylint: disable=W0611
from nnlib.model import Sequential
from nnlib.layers import FCFactory, Conv2DFactory
from nnlib.activation import Sigmoid

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


def verify(clsfer: NeuralNet, X: ndarray, Y: ndarray):
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
    idx = np.arange(len(label)) * CLS_LEN + label
    np.put(Yo, idx, 1)
    return Yo


def preprocess(X: ndarray) -> ndarray:
    mx = np.max(X, axis=1, keepdims=True)
    return X / mx


def main() -> None:
    Tx, Ty = mnist_reader.load_mnist('../../data/fashion', kind='train')
    Vx, Vy = mnist_reader.load_mnist('../../data/fashion', kind='t10k')
    Tx = Tx[:2000]
    Ty = Ty[:2000]

    Tx = preprocess(Tx)
    Tyy = label2Y(Ty)
    Vx = preprocess(Vx)
    train_time = 0.0
    eta = 0.4
    # model = buildBP(Tx.shape[1], [128, 256, 64, CLS_LEN],
    #                eta)  # pylint: disable=E1136
    model = Sequential(np.array([28, 28, 1]))
    model.add(Conv2DFactory((5, 5), 3, Sigmoid(), eta))
    model.add(FCFactory(200, Sigmoid(), eta))
    model.add(FCFactory(64, Sigmoid(), eta))
    model.add(FCFactory(CLS_LEN, Sigmoid(), eta))

    for layer in model.layers:
        print(layer.isize().prod(), layer.osize().prod())

    for i in range(50):
        blk = min(i*10+1, 5000)
        train_start = time.time()
        model.trainBGD(Tx, Tyy, blk)
        train_end = time.time()
        print(f'------ epoch {i} -------')
        print(f'eta: {eta}, blk: {blk}')
        print(f'training time: {train_end-train_start}s')
        train_time += train_end - train_start
        verify(model, Vx, Vy)
        eta = max(eta * 0.9, 0.001)
        model.set_eta(eta)
    print(f'training time: {train_time}s')
    print('verify on Tx')
    verify(model, Tx, Ty)
    print('verify on Vx')
    verify(model, Vx, Vy)


if __name__ == '__main__':
    main()
