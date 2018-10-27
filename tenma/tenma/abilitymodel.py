# coding:utf-8
import os
import json
from pprint import pprint
import pandas as pd
import numpy as np
import pystan

BF_N = 4

def normal(mu, sigma):
    if 'env' in os.environ.keys() and os.environ['env'] == "test":
        return mu
    return np.random.normal(mu, sigma)

def inv_logit(x):
    return 1.0 / (1.0 + np.exp(-x))

P_MU = "p_mu"
P_SIGMA = "p_sigma"
def calc_order(order, params):
    return normal(params[P_MU], params[P_SIGMA]) * (1.0 / order)

G_A_MU = "g_a_mu"
G_A_SIGMA = "g_a_sigma"
G_B_MU = "g_b_mu"
G_B_SIGMA = "g_b_sigma"
def calc_grade(bf_grade, now_grade, params):
    return inv_logit(
        normal(params[G_A_MU], params[G_A_SIGMA])
        * (now_grade - bf_grade)
        + normal(params[G_B_MU], params[G_B_SIGMA])
    )

def calc_bfrace(order, bf_grade, now_grade, params):
    return calc_order(order, params) * calc_grade(bf_grade, now_grade, params)


BIAS_MU = "bias_mu"
BIAS_SIGMA = "bias_sigma"
def predict(l_order, l_bf_grade, now_grade, all_params):
    ret = 0
    for num in range(BF_N):
        params = {
            P_MU : all_params[P_MU][num],
            P_SIGMA : all_params[P_SIGMA][num],
            G_A_MU : all_params[G_A_MU],
            G_A_SIGMA : all_params[G_A_SIGMA],
            G_B_MU : all_params[G_B_MU],
            G_B_SIGMA : all_params[G_B_SIGMA]
        }
        params[P_MU] = all_params[P_MU][num]
        params[P_SIGMA] = all_params[P_SIGMA][num]
        ret += calc_bfrace(
            l_order[num],
            l_bf_grade[num],
            now_grade,
            params
        )
    ret += normal(
        all_params[BIAS_MU],
        all_params[BIAS_SIGMA]
    )

    return inv_logit(ret)


COL_BF_JYUNI_TEMP = "bf_jyuni_%02d"
def x_for_ability(df):
    cols = []
    for num in range(1, BF_N + 1):
        a_col = COL_BF_JYUNI_TEMP % num
        cols.append(a_col)
        df[a_col] = df.groupby('kettonum')['kakuteijyuni'].shift(num).fillna('09').replace('00', '09')
    
    return df[cols].astype(np.int32).values

COL_BF_RANK_TEMP = "bf_rank_%02d"
def x_bfrank(df):
    cols = []
    for num in range(1, BF_N + 1):
        a_col = COL_BF_RANK_TEMP % num
        cols.append(a_col)
        df[a_col] = df.groupby('kettonum')['racerank'].shift(num).fillna(4)
    
    return df[cols].astype(np.int32).values + 1

def x_nowrank(df):
    return df['racerank'].astype(np.int32).values + 1

def get_learn_df(df):
    learn_mask = (df['year'].astype(int) < 2018)

    for num in range(1, BF_N + 1):
        learn_mask = learn_mask & ~(df.groupby(['kettonum'])['kakuteijyuni'].shift(num).isna())
    
    mask = df["kakuteijyuni"] == '01'
    target = df.pipe(lambda df: df[mask & learn_mask])
    non_target = df.pipe(lambda df: df[~mask & learn_mask])
    N = len(target.index)
    df_ret = pd.concat([
            target.assign(answer=1).sample(N),
            # 対象と同じ数だけサンプリング
            non_target.sample(N).assign(answer=0),
        ],
        ignore_index=True
    )
    return df_ret, df_ret['answer']

P = "p"
BIAS = "bias"
G_A = "g_a"
G_B = "g_b"
PARAM_FILE_PATH = "model/ability_params.json"
def learn(df):
    df_learn, Y = get_learn_df(df)
    X_R = x_for_ability(df_learn)
    X_G = x_bfrank(df_learn)
    X_RACE = x_nowrank(df_learn)

    print(df_learn['answer'].value_counts())


    data = {
        "N": len(df_learn.index),
        "R": 19,
        "G": 10,
        "X_R": X_R,
        "X_G": X_G,
        "X_RACE": X_RACE,
        "Y": Y
    }

    model = pystan.StanModel(file="stanmodel/abilitymodel.stan")
    fit_vb = model.vb(data=data, pars=[P, BIAS, G_A, G_B])
    ms = pd.read_csv(fit_vb['args']['sample_file'].decode('utf-8'), comment='#')

    # pの変数リストを作成
    l_p = []
    for num in range(1, BF_N + 1):
        l_p.append("%s.%d" % (P, num))

    params = {
        P_MU : ms[l_p].mean(axis=0).tolist(),
        P_SIGMA : ms[l_p].std(axis=0).tolist(),
        BIAS_MU : ms[BIAS].mean(axis=0).tolist(),
        BIAS_SIGMA : ms[BIAS].std(axis=0).tolist(),
        G_A_MU : ms[G_A].mean(axis=0).tolist(),
        G_A_SIGMA : ms[G_A].std(axis=0).tolist(),
        G_B_MU : ms[G_B].mean(axis=0).tolist(),
        G_B_SIGMA : ms[G_B].std(axis=0).tolist(),
    }

    with open(PARAM_FILE_PATH, 'w') as outfile:
        json.dump(params, outfile)
