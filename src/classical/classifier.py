from typing import List
from abc import ABCMeta, abstractmethod
import random
from multiprocessing import Pool, cpu_count
import sys

import numpy as np
from numpy import ndarray
from scipy.spatial import KDTree  # pylint: disable=E0611


def _cat_cnt(label: ndarray, cls_len: int, cat_m: float) -> ndarray:
    cnt = np.repeat(cat_m, cls_len)
    for y in label:
        cnt[y] += 1
    return cnt


def choose_by_poss(poss: np.ndarray) -> np.ndarray:
    cat = random.choices(range(len(poss)), weights=poss)
    poss = np.zeros(len(poss))
    poss[cat] = 1.0
    return poss


def _argmax(poss: np.ndarray) -> int:
    is_mx = poss >= poss.max()
    mx_idx = [i for i in range(len(poss)) if is_mx[i]]
    return random.choices(mx_idx)[0]


def choose_max(poss: np.ndarray) -> np.ndarray:
    cat = _argmax(poss)
    poss = np.zeros(len(poss))
    poss[cat] = 1.0
    return poss


class Classifier(metaclass=ABCMeta):
    @abstractmethod
    def classify(self, data: ndarray, choose_one: bool = True) -> ndarray:
        return np.zeros(1)


class NaiveBayes(Classifier):
    def __init__(self, data: ndarray, v_mx: int, label: ndarray, cls_len: int,
                 cat_m: float = 0.01, kw_m: float = 0.001) -> None:
        self.kw_m = kw_m
        self.cls_len = cls_len
        cat_cnt = _cat_cnt(label, cls_len, cat_m)
        self.cat_poss = cat_cnt / cat_cnt.sum()

        r, c = data.shape[:2]
        self.attr_poss: ndarray = np.zeros((c, v_mx+1, cls_len)) + kw_m
        for i in range(r):
            x = data[i]
            y = label[i]
            for ai in range(c):
                a = x[ai]
                self.attr_poss[ai, a, y] += 1
        cat_cnt += kw_m * (v_mx+1)
        self.attr_poss /= cat_cnt

    def classify(self, data: ndarray, choose_one: bool = True) -> ndarray:
        r, c = data.shape[:2]
        ret = np.repeat([self.cat_poss], r, axis=0)
        for ai in range(c):
            ap = self.attr_poss[ai, data[:, ai]]
            assert ret.shape == ap.shape
            ret *= ap
            ret = ret / np.sum(ret, axis=1, keepdims=True)
        if choose_one:
            for i in range(r):
                ret[i, :] = choose_max(ret[i, :])
        return ret


class KNN(Classifier):
    def __init__(self, data: ndarray, label: ndarray, cls_len: int, k: int, cat_m: float = 0.0) -> None:
        v = data.var(axis=0)
        col_size = int(len(v) * 0.9)
        idx = np.argpartition(v, len(v) - col_size)[-col_size:]
        self.data = data[:, idx].copy()
        self.idx = idx
        self.label = label
        self.cls_len = cls_len
        self.k = k
        self.cat_m = cat_m

    def classify(self, data: ndarray, choose_one: bool = True) -> ndarray:
        data = data[:, self.idx].copy()
        r = len(data)
        cpu_n = cpu_count() - 1
        blk = (r + cpu_n - 1) // cpu_n
        jobs = [(self.data, self.label, self.cls_len, self.k, data[i:i+blk])
                for i in range(0, r, blk)]
        ret: ndarray
        with Pool(processes=cpu_n) as pool:
            ret = np.row_stack(pool.map(self.do_classify, jobs))
        ret += self.cat_m
        ret = ret / np.sum(ret, axis=1, keepdims=True)
        if choose_one:
            for i in range(r):
                ret[i, :] = choose_max(ret[i, :])
        return ret

    @staticmethod
    def do_classify(args) -> ndarray:
        mdata, mlabel, cls_len, k, data = args
        r = len(data)
        ret = np.zeros((r, cls_len))
        for i in range(r):
            mx = np.maximum(mdata, data[i])
            mn = np.minimum(mdata, data[i])
            diff = (mx - mn).astype(np.uint16)
            dis = np.sum(diff**2, axis=1)
            idx = np.argpartition(dis, k)[:k]
            for j in idx:
                ret[i, mlabel[j]] += 1.0/np.max([dis[j], 1])
        return ret / np.sum(ret, axis=1, keepdims=True)


class RKNN(Classifier):
    def __init__(self, data: ndarray, label: ndarray, cls_len: int, k: int, idx_pool: int = 16, idx_sample: int = 16, cat_m: float = 0.0) -> None:
        sys.setrecursionlimit(10000)
        v = data.var(axis=0)
        idx = np.argpartition(v, len(v) - idx_pool)[-idx_pool:]
        idx = random.sample(list(idx), idx_sample)
        idx.sort()
        self.kdtree = KDTree(
            data[:, idx].astype(np.float16), leafsize=16)
        self.idx = idx
        self.label = label
        self.cls_len = cls_len
        self.k = k
        self.cat_m = cat_m

    def classify(self, data: ndarray, choose_one: bool = True) -> ndarray:
        r, _ = data.shape[:2]
        dis, idx = self.kdtree.query(
            data[:, self.idx].astype(np.float16), self.k, eps=0.1)
        ret = np.zeros((r, self.cls_len))
        for i in range(r):
            if self.k == 1:
                y = self.label[idx[i]]
                ret[i, y] += 1
                continue
            for j in range(self.k):
                y = self.label[idx[i, j]]
                ret[i, y] += 1.0/np.max([dis[i, j], 1.0])
        ret /= np.sum(ret, axis=1, keepdims=True)
        ret += self.cat_m
        ret /= np.sum(ret, axis=1, keepdims=True)
        if choose_one:
            for i in range(r):
                ret[i, :] = choose_max(ret[i, :])
        return ret


class RandomClassifier(Classifier):
    def __init__(self, cls_len: int, clsfers: List[Classifier]) -> None:
        self.clsfers = clsfers
        self.cls_len = cls_len

    def classify(self, data: ndarray, choose_one: bool = True) -> ndarray:
        r = len(data)
        cpu_n = cpu_count()
        blk = (len(self.clsfers) + cpu_n - 1) // cpu_n
        jobs = [(self.clsfers[i:i+blk], self.cls_len, data)
                for i in range(0, len(self.clsfers), blk)]
        ret = np.zeros((r, self.cls_len))
        with Pool(processes=cpu_n) as pool:
            for r in pool.map(self.do_classify, jobs):
                ret += r
        ret /= np.sum(ret, axis=1, keepdims=True)
        if choose_one:
            for i in range(r):
                ret[i, :] = choose_by_poss(ret[i, :])
        return ret

    @staticmethod
    def do_classify(args) -> ndarray:
        clsfers, cls_len, data = args
        r = len(data)
        ret = np.zeros((r, cls_len))
        for clsfer in clsfers:
            ret += clsfer.classify(data, choose_one=False)
        return ret


def randRKNN(X, Y, cls_len: int, k: int, n: int, size: int, idx_pool: int, idx_sample: int) -> RandomClassifier:
    clsfers: List[Classifier] = []
    for _ in range(n):
        idx = random.sample(range(len(X)), size)
        clsfers.append(RKNN(X[idx, :], Y[idx], cls_len, k,
                            idx_pool=idx_pool, idx_sample=idx_sample))
    return RandomClassifier(cls_len, clsfers)
