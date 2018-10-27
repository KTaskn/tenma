# coding:utf-8
import os
import json
import numpy as np
import pandas as pd
import pystan
from tenma import abilitymodel as am

def normal(mu, sigma):
    if 'env' in os.environ.keys() and os.environ['env'] == "test":
        return mu
    return np.random.normal(mu, sigma)

def inv_logit(x):
    return 1.0 / (1.0 + np.exp(-x))

A_MU = "a_mu"
A_SIGMA = "a_sigma"
B_MU = "b_mu"
B_SIGMA = "b_sigma"
def calc(x, params):
    return inv_logit(
        x * normal(params[A_MU], params[A_SIGMA])
        + normal(params[B_MU], params[B_SIGMA])
    )

SCORE = "score"
def _predict(df, score, params):
    cal_zscore = lambda x: (x - x.mean()) / x.std()
    df[SCORE] = score
    zscore = df.groupby(
        ['year', 'monthday', 'jyocd', 'racenum']
    )[SCORE].transform(cal_zscore).fillna(0.).values

    return calc(zscore, params)

def predict(df):
    score = SCORE_MODEL.predict(df)

    with open(PARAM_FILE_PATH) as f:
        params = json.loads(f.read())
        
    return _predict(df, score, params)

def get_learn_df(df, model):
    # scoreを利用するモデルのを流用する
    return model.get_learn_df(df)

A = "a"
B = "b"
STAN_MODEL_PATH = "stanmodel/comparemodel.stan"
PARAM_FILE_PATH = "model/compare_params.json"
SCORE_MODEL = am
def learn(df):
    df_learn, Y = get_learn_df(df, SCORE_MODEL)
    score = SCORE_MODEL.predict(df_learn)
   
    model = pystan.StanModel(file=STAN_MODEL_PATH)
    data = {
        "N": len(df_learn.index),
        "X": score,
        "Y": Y
    }
    fit_vb = model.vb(data=data, pars=[A, B])
    ms = pd.read_csv(fit_vb['args']['sample_file'].decode('utf-8'), comment='#')
    params = {
        A_MU : ms[A].mean(axis=0).tolist(),
        A_SIGMA : ms[A].std(axis=0).tolist(),
        B_MU : ms[B].mean(axis=0).tolist(),
        B_SIGMA : ms[B].std(axis=0).tolist(),
    }

    with open(PARAM_FILE_PATH, 'w') as outfile:
        json.dump(params, outfile)
