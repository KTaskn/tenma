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

COL_BF_JYUNI_TEMP = "bf_jyuni_%02d"
def x_for_ability(df, D):
    cols = []
    for num in range(1, D + 1):
        a_col = COL_BF_JYUNI_TEMP % num
        cols.append(a_col)
        df[a_col] = df.groupby('kettonum')['kakuteijyuni'].shift(num).fillna('09').replace('00', '09')
    
    return df[cols].astype(np.int32).values

COL_BF_RANK_TEMP = "bf_rank_%02d"
def x_bfrank(df, D):
    cols = []
    for num in range(1, D + 1):
        a_col = COL_BF_RANK_TEMP % num
        cols.append(a_col)
        df[a_col] = df.groupby('kettonum')['racerank'].shift(num).fillna(4)
    
    return df[cols].astype(np.int32).values + 1

def x_nowrank(df):
    return df['racerank'].astype(np.int32).values + 1

PARAM_P_MU = "p_mu"
PARAM_P_SIGMA = "p_sigma"
PARAM_BIAS_MU = "bias_mu"
PARAM_BIAS_SIGMA = "bias_sigma"
PARAM_R_A_MU = "r_a_mu"
PARAM_R_A_SIGMA = "r_a_sigma"
PARAM_R_BIAS_MU = "r_bias_mu"
PARAM_R_BIAS_SIGMA = "r_bias_sigma"
SAMPLE_N = 100
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
def ability(X_R, X_G, X_RACE, params):
    inv_logit = lambda  x: 1.0 / (1.0 + np.exp(-x))
    length = len(X_R)
    p = np.zeros(length)
    for n in range(length):
        p_tmp = np.zeros(SAMPLE_N)
        for d in range(len(X_R[0])):
            p_tmp += (1 / X_R[n][d]) * np.random.normal(
                params[PARAM_P_MU][d],
                params[PARAM_P_SIGMA][d],
                SAMPLE_N
            ) * inv_logit(
                np.random.normal(
                    params[PARAM_R_A_MU],
                    params[PARAM_R_A_SIGMA],
                    SAMPLE_N
                ) * (X_RACE[n] - X_G[n][d]) + np.random.normal(
                    params[PARAM_R_BIAS_MU],
                    params[PARAM_R_BIAS_SIGMA],
                    SAMPLE_N
                )
            )
        p_tmp += np.random.normal(
                params[PARAM_BIAS_MU],
                params[PARAM_BIAS_SIGMA],
                SAMPLE_N
            )
        p[n] += np.mean(p_tmp)

    return inv_logit(p)
