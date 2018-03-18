from __future__ import division
from __future__ import print_function
import numpy as np
from numba import jit
import matplotlib.pyplot as plt


@jit(nopython=True)
def _seed_numba(seed):
    np.random.seed(seed)

def seed(seed):
    np.random.seed(seed)
    _seed_numba(seed)

@jit(nopython=True)
def log_mean_prob(log_Prob):
    max_log_prob = np.max(log_Prob)
    return np.log(np.mean(np.exp(log_Prob - max_log_prob))) + max_log_prob

@jit(nopython=True)
def normalize_prob(log_Prob):
    Prob = np.exp(log_Prob - np.max(log_Prob))
    Prob /= np.sum(Prob)
    return Prob

@jit(nopython=True)
def residual_resample(Prob):
    M = Prob.size
    res_M = M
    res_Prob = Prob * M
    count = np.zeros(M, dtype=np.int64)
    for i in range(M):
        count[i] = int(res_Prob[i])
        res_Prob[i] -= count[i]
        res_M -= count[i]
    if res_M > 0:
        j = 0
        cum_prob = 0.0
        res_Prob /= res_M
        sampled_float = np.sort(np.random.rand(res_M))
        for i in range(M):
            cum_prob += res_Prob[i]
            while j < res_M and sampled_float[j] <= cum_prob:
                count[i] += 1
                j += 1
    j = 0
    sampled_int = np.arange(M)
    for i in range(M):
        if count[i] == 0:
            while count[j] <= 1: j += 1
            sampled_int[i] = j
            count[j] -= 1
    return sampled_int

@jit(nopython=True)
def resample(s_post,s_pre,probability):
    sampled_index = residual_resample(probability)
    for j in range(s_pre.shape[0]):
        s_post[j] = s_pre[sampled_index[j]]