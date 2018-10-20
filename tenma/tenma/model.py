# coding:utf-8
import numpy as np

def cutoff(x, threshold=0.5):
    if x < threshold:
        return 0
    else:
        return 1

"""
params: {
    "mu_01": xx,
    "sigma_01": xx,
}
"""
COL_MU_TEMP = "mu_%02d"
COL_SIGMA_TEMP = "sigma_%02d"
def ability(X, N, params):        
    sigmoid = lambda  x: 1.0 / (1.0 + np.exp(-x))
    length = len(X)
    p = np.zeros(length)
    for i in range(1, N + 1):
        # i は 1開始
        X_col = list(map(lambda x: x[i - 1], X))
        col_mu = COL_MU_TEMP % i
        col_sigma = COL_SIGMA_TEMP % i
        mu = params[col_mu]
        sigma = params[col_sigma]
        before = lambda x: np.random.normal(mu[x], sigma[x])
        p += np.array(list(map(before, X_col)))

    return sigmoid(p)
