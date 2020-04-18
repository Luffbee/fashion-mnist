from typing import List, Dict, Tuple
import time

import numpy as np
from numpy import ndarray
import pandas as pd

from utils import mnist_reader
from classifier import Classifier, NaiveBayes, KNN, RKNN, RandomClassifier, randRKNN # pylint: disable=W0611

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


def verify(clsfer: Classifier, X: ndarray, Y: ndarray):
    run_start = time.time()
    r_poss = clsfer.classify(X, choose_one=False)
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


Point = Tuple[int, int]


def block_index(p0: Point, p1: Point, shape: Point) -> List[int]:
    _, C = shape
    r0, c0 = p0
    r1, c1 = p1
    ret = []
    for i in range(r0, r1):
        for j in range(c0, c1):
            ret.append(i*C+j)
    return ret


def preprocess(X: ndarray) -> ndarray:
    mx = np.max(X, axis=1, keepdims=True)
    nX = ((X / mx) * 255)
    ret = nX.astype(np.uint8)
    step = 14
    for r0 in range(0, 28-step+1):
        for c0 in range(0, 28-step+1):
            idx = block_index((r0, c0), (r0+step, c0+step), (28, 28))
            ret = np.column_stack(
                (ret, np.count_nonzero(X[:, idx], axis=1).astype(np.uint8)))
    return ret


def main() -> None:
    clsfer_kind = 'KNN'
    Tx, Ty = mnist_reader.load_mnist('../data/fashion', kind='train')
    Vx, Vy = mnist_reader.load_mnist('../data/fashion', kind='t10k')
    pre_start = time.time()
    Tx = preprocess(Tx)
    Vx = preprocess(Vx)
    pre_end = time.time()
    print(f'preprocessed {pre_end-pre_start}s')
    print(f'training data set size: {len(Tx)}')
    train_start = time.time()
    clsfer: Classifier
    if clsfer_kind == 'NaiveBayes':
        clsfer = NaiveBayes(Tx, 255, Ty, CLS_LEN)
    elif clsfer_kind == 'KNN':
        clsfer = KNN(Tx, Ty, CLS_LEN, 5)
        #clsfer = randRKNN(Tx, Ty, CLS_LEN, 5, 100, 10000, 30, 10)
    else:
        assert False
    train_end = time.time()
    print(f'training time: {train_end - train_start}s')
    print('verify on Tx')
    #verify(clsfer, Tx, Ty)
    #Vx = Vx[:1000]
    #Vy = Vy[:1000]
    print('verify on Vx')
    verify(clsfer, Vx, Vy)


if __name__ == '__main__':
    main()
