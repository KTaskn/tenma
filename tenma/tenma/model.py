# coding:utf-8
import numpy as np

"""
cutoff
実数値を閾値によって２値に変換する
input: float x
input: float threshold 閾値
output int cutoff
"""
def cutoff(x, threshold=0.5):
    if x < threshold:
        return 0
    else:
        return 1

COL_MU_TEMP = "mu_%02d"
COL_SIGMA_TEMP = "sigma_%02d"
"""
ability
個々の能力を計測する
input: list<list <int>> X 着順
input: int D 入力の次元
input: dict params 係数パラメータ
    {
        "mu_01": [1, 1],
        "sigma_01": [1, 1],
    }
output: list<float> 確率p
"""
def ability(X, D, params):        
    sigmoid = lambda  x: 1.0 / (1.0 + np.exp(-x))
    length = len(X)
    p = np.zeros(length)
    for i in range(1, D + 1):
        # i は 1開始
        X_col = list(map(lambda x: x[i - 1], X))
        col_mu = COL_MU_TEMP % i
        col_sigma = COL_SIGMA_TEMP % i
        mu = params[col_mu]
        sigma = params[col_sigma]
        before = lambda x: np.random.normal(mu[x], sigma[x])
        p += np.array(list(map(before, X_col)))

    return sigmoid(p)
