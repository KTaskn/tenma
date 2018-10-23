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
        df[a_col] = df.groupby('kettonum')['kakuteijyuni'].shift(num).fillna('19').replace('00', '19')
    
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

COL_P_MU = "p_mu"
COL_P_SIGMA = "p_sigma"
COL_RACE_P_MU = "race_p_mu"
COL_RACE_P_SIGMA = "race_p_sigma"
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
        p_tmp = np.zeros(100)
        for d in range(len(X_R[0])):
            p_tmp += np.random.normal(
                params[COL_P_MU][
                        X_R[n][d] - 1
                    ][
                        X_G[n][d] - 1
                    ],
                params[COL_P_SIGMA][
                        X_R[n][d] - 1
                    ][
                        X_G[n][d] - 1
                    ],
                100
            )
        p_tmp += np.random.normal(
                params[COL_RACE_P_MU][X_RACE[n] - 1],
                params[COL_RACE_P_SIGMA][X_RACE[n] - 1],
                100
            )
        p[n] += np.mean(p_tmp)

    return inv_logit(p)
