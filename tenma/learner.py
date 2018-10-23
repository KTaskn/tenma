# coding:utf-8
import json
from pprint import pprint
import numpy as np
import pystan
import pandas as pd
from tenma import dataload, abilitymodel


def get_dataset(df):
    learn_mask = df['year'].astype(int) < 2018
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

if __name__ == "__main__":
    df = dataload.load()
    # 学習用サンプルを取得
    df_learn, Y = get_dataset(df)
    del(df)
    D = 4
    X_R = abilitymodel.x_for_ability(df_learn, D)
    X_G = abilitymodel.x_bfrank(df_learn, D)
    X_RACE = abilitymodel.x_nowrank(df_learn)

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
    pprint(data)

    model = pystan.StanModel(file="stanmodel/abilitymodel.stan")
    print("compiled")
    fit = model.sampling(data=data, pars=['p', 'race_p'])
    params = {
        abilitymodel.COL_P_MU : np.mean(fit.extract()['p'], axis=0).tolist(),
        abilitymodel.COL_P_SIGMA : np.std(fit.extract()['p'], axis=0).tolist(),
        abilitymodel.COL_RACE_P_MU : np.mean(fit.extract()['race_p'], axis=0).tolist(),
        abilitymodel.COL_RACE_P_SIGMA : np.std(fit.extract()['race_p'], axis=0).tolist(),
    }


    FILE_PATH = "model/ability_params.json"
    with open(FILE_PATH, 'w') as outfile:
        json.dump(params, outfile)