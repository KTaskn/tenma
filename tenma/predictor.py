# coding:utf-8
import json
import numpy as np
import pandas as pd
from tenma import dataload, abilitymodel

YEAR = "2018"
MONTHDAY = "1028"

if __name__ == "__main__":
    df = dataload.load()
    df = pd.merge(
        df.pipe(lambda df: df[
            (df['year'] == YEAR) & (df['monthday'] == MONTHDAY)
        ])[['kettonum']].drop_duplicates(),
        df
    ).sort_values(['year', 'monthday'])
    df = df[(df.groupby(['kettonum'])['kakuteijyuni'].shift(1) == df.groupby(['kettonum'])['kakuteijyuni'].shift(1))
        & (df.groupby(['kettonum'])['kakuteijyuni'].shift(2) == df.groupby(['kettonum'])['kakuteijyuni'].shift(2))
        & (df.groupby(['kettonum'])['kakuteijyuni'].shift(3) == df.groupby(['kettonum'])['kakuteijyuni'].shift(3))
        & (df.groupby(['kettonum'])['kakuteijyuni'].shift(4) == df.groupby(['kettonum'])['kakuteijyuni'].shift(4))]
    FILE_PATH = "model/ability_params.json"
    with open(FILE_PATH) as f:
        params = json.loads(f.read())
        print(params.keys())
    D = 4
    X_R = abilitymodel.x_for_ability(df, D)
    X_G = abilitymodel.x_bfrank(df, D)
    X_RACE = abilitymodel.x_nowrank(df)
    result = abilitymodel.ability(X_R, X_G, X_RACE, params)
    df['predict'] = result

    cal_zscore = lambda x: (x - x.mean()) / x.std()
    df['predict_comp'] = df.groupby(['year', 'monthday', 'jyocd', 'racenum'])['predict'].transform(cal_zscore).fillna(0.)

    FILE_PATH = "model/compare_params.json"
    with open(FILE_PATH) as f:
        params = json.loads(f.read())

    inv_logit = lambda  x: 1.0 / (1.0 + np.exp(-x))
    a = np.random.normal(params['a_mu'], params['a_sigma'], 100)
    b = np.random.normal(params['b_mu'], params['b_sigma'], 100)
    l = []
    X = df['predict_comp'].values
    for n in range(len(df.index)):
        l.append(inv_logit(a * X[n] + b).mean())
    df['predict_comp'] = l



    df[['odds', 'predict', 'predict_comp']].to_csv('result.csv')
    print(df.pipe(lambda df: df[
            (df['year'] == YEAR)
            & (df['monthday'] == MONTHDAY)
            & (df['racenum'] == '11')
        ])[['jyocd', 'racenum', 'bamei', 'predict']].sort_values(['jyocd', 'racenum', 'predict']))