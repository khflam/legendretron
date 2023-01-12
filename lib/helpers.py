import os
import random

import matplotlib as mpl
import numpy as np

latex_preamble = r'''
\usepackage{amsmath,bm,amssymb}
\newcommand*\intd{\mathop{}\!\mathrm{d}}
'''
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = latex_preamble
os.environ['PATH'] = '%s:/Library/TeX/texbin/' % os.environ['PATH']


# memoize a function
def memodict_d(f):
    class memodict(dict):
        def __missing__(self, key):
            ret = self[key] = f(key)
            return ret
    d = memodict()
    f.memodict_dict = d
    return d.__getitem__, d


# iterate over a sequence in random order
def randomly(seq, seed=None):
    if seed is not None:
        state = random.getstate()
        random.seed(seed)
    shuffled = list(seq)
    random.shuffle(shuffled)
    if seed is not None:
        random.setstate(state)
    return iter(shuffled)


# sample from a multivariate normal
def sample_mvn(mu, sigma, n, seed=None):

    mu = mu.reshape(-1, 1)
    d = sigma.shape[0]
    assert sigma.shape[1] == d
    assert len(mu) == d, (len(mu), d)
    try:
        c = np.linalg.cholesky(sigma)
    except:
        raise Exception('eig(sigma)', sorted(np.linalg.eigvals(sigma)))

    rval = mu + np.dot(c, np.random.randn(d * n).reshape((d, n)))

    return rval


# simple moving average
def sma(x, t, n):
    isort = np.argsort(x)
    x = x[isort]
    t = t[isort]
    t = t.flatten()
    xc = np.cumsum(t)
    i = np.maximum(np.arange(len(t)) - n, 0)
    j = np.minimum(np.arange(len(t)) + n, len(t) - 1)
    count = j-i
    return x, (xc[j]-xc[i]) / count
