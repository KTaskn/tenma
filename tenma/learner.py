# coding:utf-8
import json
from pprint import pprint
import numpy as np
import pystan
import pandas as pd
from tenma import dataload, abilitymodel


def get_dataset(df):
    learn_mask = ((df['year'].astype(int) < 2018)
        & (df.groupby(['kettonum'])['kakuteijyuni'].shift(1) == df.groupby(['kettonum'])['kakuteijyuni'].shift(1))
        & (df.groupby(['kettonum'])['kakuteijyuni'].shift(2) == df.groupby(['kettonum'])['kakuteijyuni'].shift(2))
        & (df.groupby(['kettonum'])['kakuteijyuni'].shift(3) == df.groupby(['kettonum'])['kakuteijyuni'].shift(3))
        & (df.groupby(['kettonum'])['kakuteijyuni'].shift(4) == df.groupby(['kettonum'])['kakuteijyuni'].shift(4)))
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
    # df = dataload.load()
    # # 学習用サンプルを取得
    # df_learn, Y = get_dataset(df)
    # del(df)
    # D = 4
    # X_R = abilitymodel.x_for_ability(df_learn, D)
    # X_G = abilitymodel.x_bfrank(df_learn, D)
    # X_RACE = abilitymodel.x_nowrank(df_learn)

    # print(df_learn['answer'].value_counts())


    # data = {
    #     "N": len(df_learn.index),
    #     "R": 19,
    #     "G": 10,
    #     "X_R": X_R,
    #     "X_G": X_G,
    #     "X_RACE": X_RACE,
    #     "Y": Y
    # }
    # pprint(data)

    # model = pystan.StanModel(file="stanmodel/abilitymodel.stan")
    # print("compiled")
    # fit_vb = model.vb(data=data, pars=['p', 'bias', 'r_a', 'r_bias'])
    # ms = pd.read_csv(fit_vb['args']['sample_file'].decode('utf-8'), comment='#')
    # params = {
    #     abilitymodel.PARAM_P_MU : ms[['p.1', 'p.2', 'p.3', 'p.4']].mean(axis=0).tolist(),
    #     abilitymodel.PARAM_P_SIGMA : ms[['p.1', 'p.2', 'p.3', 'p.4']].std(axis=0).tolist(),
    #     abilitymodel.PARAM_BIAS_MU : ms['bias'].mean(axis=0).tolist(),
    #     abilitymodel.PARAM_BIAS_SIGMA : ms['bias'].std(axis=0).tolist(),
    #     abilitymodel.PARAM_R_A_MU : ms['r_a'].mean(axis=0).tolist(),
    #     abilitymodel.PARAM_R_A_SIGMA : ms['r_a'].std(axis=0).tolist(),
    #     abilitymodel.PARAM_R_BIAS_MU : ms['r_bias'].mean(axis=0).tolist(),
    #     abilitymodel.PARAM_R_BIAS_SIGMA : ms['r_bias'].std(axis=0).tolist(),
    # }

    # FILE_PATH = "model/ability_params.json"
    # with open(FILE_PATH, 'w') as outfile:
    #     json.dump(params, outfile)

    FILE_PATH = "model/ability_params.json"
    with open(FILE_PATH) as f:
        params = json.loads(f.read())
        print(params.keys())


    # 比較モデル
    cal_zscore = lambda x: (x - x.mean()) / x.std()
    df = dataload.load()
    D = 4
    X_R = abilitymodel.x_for_ability(df, D)
    X_G = abilitymodel.x_bfrank(df, D)
    X_RACE = abilitymodel.x_nowrank(df)
    df['predict'] = abilitymodel.ability(X_R, X_G, X_RACE, params)
    df['zscore'] = df.groupby(['year', 'monthday', 'jyocd', 'racenum'])['predict'].transform(cal_zscore).fillna(0.)

    df_learn, Y = get_dataset(df)

    
    model = pystan.StanModel(file="stanmodel/comparemodel.stan")
    print("compiled")
    data = {
        "N": len(df_learn.index),
        "X": df_learn['zscore'],
        "Y": Y
    }
    fit_vb = model.vb(data=data, pars=['a', 'b'])
    ms = pd.read_csv(fit_vb['args']['sample_file'].decode('utf-8'), comment='#')
    params = {
        "a_mu" : ms['a'].mean(axis=0).tolist(),
        "a_sigma" : ms['a'].std(axis=0).tolist(),
        "b_mu" : ms['b'].mean(axis=0).tolist(),
        "b_sigma" : ms['b'].std(axis=0).tolist(),
    }

    FILE_PATH = "model/compare_params.json"
    with open(FILE_PATH, 'w') as outfile:
        json.dump(params, outfile)