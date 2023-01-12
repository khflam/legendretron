import os
from collections import Counter

import datasets
import numpy as np
import scipy.io as sio
from sklearn.datasets import load_svmlight_file

from lib import helpers

_mnists = ["mnist", "fmnist", "kmnist"]
_path = os.path.expanduser("~/data/")


def raw_digit_data(ypos, yneg, datatype, train, seed):

    assert datatype in _mnists

    mndata = datasets.MNIST("%s/%s" % (_path, datatype))

    if train:
        Xall, Yall = map(np.array, mndata.load_training())
    else:
        Xall, Yall = map(np.array, mndata.load_testing())

    print("raw_digit_data len = %i" % len(Yall))

    ineg = np.zeros_like(Yall, dtype=bool)
    ipos = np.zeros_like(Yall, dtype=bool)
    for yy in yneg:
        ineg[Yall == yy] = 1
    for yy in ypos:
        ipos[Yall == yy] = 1
    assert not np.any(ipos & ineg)
    iany = (ipos | ineg) == 1
    Ybinary = np.zeros_like(Yall)
    Ybinary[ipos] = 1
    Y = Ybinary[iany]
    Ymulti = Yall[iany]
    Xall = Xall[iany, :]

    ishuffle = np.array(tuple(helpers.randomly(range(len(Y)), seed))).astype(
        int
    )
    X = Xall[ishuffle, :]
    Y = Y[ishuffle]
    Ymulti = Ymulti[ishuffle]

    print("y counts b", "pos", (Y == 1).sum(), "neg", (Y == 0).sum())

    print(set(Ymulti[Y == 1]), set(ypos))
    print(set(Ymulti[Y.flatten() == 0].flatten()), set(yneg))

    return (
        X.astype(np.double),
        Y.astype(int).reshape(-1, 1),
        Ymulti.astype(int).reshape(-1, 1),
    )


# preprocessed and split *nist data
def digit_data(pos, neg, ntrain, ntest, datatype, seed):

    if neg is None:
        assert pos is not None
        neg = tuple(set(range(10)).difference(pos))
    if pos is None:
        pos = tuple(set(range(10)).difference(neg))

    print("pos", pos)
    print("neg", neg)
    print("ntrain", ntrain)
    print("ntest", ntest)
    print("datatype", datatype)

    assert datatype in ("mnist", "fmnist", "kmnist")

    Xtrain, Ytrain, Ytrainmulti = raw_digit_data(
        pos, neg, datatype, train=True, seed=seed
    )
    mux = np.mean(Xtrain, axis=0)
    stdx = np.std(Xtrain, axis=0)
    standardise = lambda X: (X - mux) / np.mean(stdx)
    Xtrain = standardise(Xtrain)
    Xtrain = Xtrain[:ntrain, :]
    Ytrain = Ytrain[:ntrain, :]
    Ytrainmulti = Ytrainmulti[:ntrain, :]
    Xtest, Ytest, Ytestmulti = raw_digit_data(
        pos, neg, datatype, train=False, seed=seed
    )
    Xtest = standardise(Xtest)
    Xtest = Xtest[:ntest, :]
    Ytest = Ytest[:ntest, :]
    Ytestmulti = Ytestmulti[:ntest, :]

    print("%i train %i test" % (len(Ytrain), len(Ytest)))
    print("train counts", str(Counter(Ytrain.flatten())))
    print("test counts", str(Counter(Ytest.flatten())))
    print("dimension", Xtrain.shape[1])

    return Xtrain, Ytrain, Ytrainmulti, Xtest, Ytest, Ytestmulti


def ytest_dict(seed):
    d = dict()
    for datatype in _mnists:
        X, Y, d[datatype] = raw_digit_data(
            [0], list(range(1, 10)), datatype, train=False, seed=seed
        )
        assert all(Y == (d[datatype] != 0)), list(
            zip(Y.flatten(), d[datatype].flatten())
        )
    return d


# load libsvm datasets
def load_libsvm(data_path, data_name):
    if data_name == "svmguide2":
        path = os.path.join(data_path, data_name, data_name + ".txt")
    elif data_name == "usps":
        path = os.path.join(data_path, data_name, data_name)
    else:
        path = os.path.join(data_path, data_name, data_name + ".scale")
    data = load_svmlight_file(path)
    X, y = data[0].toarray(), data[1].reshape(len(data[1]), 1)
    y = y - y.min()
    if data_name == "glass" or data_name == "svmguide4":
        y = np.array([(x - 1 if x > 3 else x) for x in y])
    return X.astype(np.double), y.astype(int).reshape(-1, 1)


def noisify(y, K=9, eta=0.1, random_state=0):
    """Flip labels with symmetric probability p."""

    m = y.shape[0]
    new_y = y.copy()
    label_flipper = np.random.RandomState(random_state)

    for idx in np.arange(m):
        p_vals = eta / K * np.ones(K + 1)
        p_vals[y[idx]] = 1 - eta
        p_vals = p_vals / sum(p_vals)
        new_y[idx] = label_flipper.multinomial(n=1, pvals=p_vals).argmax()

    return new_y
