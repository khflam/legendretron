"""
Definition of Log(Softmax^{+}) and other helper functions for ICNNs
Modified from:
https://github.com/CW-Huang/CP-Flow/blob/main/lib/functional.py
"""
import numpy as np
import torch
import torch.nn as nn

DELTA = 1e-7
torch.set_default_dtype(torch.float64)


def softplus(x):
    return nn.functional.softplus(x) + DELTA


def sigmoid(x):
    return torch.sigmoid(x) * (1 - DELTA) + 0.5 * DELTA


def logsigmoid(x):
    return -softplus(-x)


def log(x):
    return torch.log(x * 1e2) - np.log(1e2)


def logit(x):
    return log(x) - log(1 - x)


def act_a(x):
    return nn.functional.softplus(x) + DELTA


def act_b(x):
    return x


def act_w(x):
    return nn.functional.softmax(x, dim=2)


def oper(array, operetor, axis=-1, keepdims=False):
    a_oper = operetor(array)
    if keepdims:
        shape = []
        for j, s in enumerate(array.size()):
            shape.append(s)
        shape[axis] = -1
        a_oper = a_oper.view(*shape)
    return a_oper


def log_sum_exp(a, axis=-1, sum_op=torch.sum):
    def maximum(x):
        return x.max(axis)[0]

    a_max = oper(a, maximum, axis, True)

    def summation(x):
        return sum_op(torch.exp(x - a_max), axis)

    b = torch.log(oper(a, summation, axis, True)) + a_max
    return b


def log_softmax_plus(a, axis=-1, sum_op=torch.sum):
    gpu_maybe = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n = a.shape[0]

    def maximum(x):
        return x.max(axis)[0]

    a_max = oper(a, maximum, axis, True)

    def summation(x):
        return sum_op(torch.exp(x - a_max), axis)

    b = torch.cat((a, torch.zeros(n, 1).to(gpu_maybe)), axis)
    return b - a_max - torch.log(oper(b, summation, axis, True))
