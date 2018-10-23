# coding:utf-8
import json
import pandas as pd
from tenma import dataload, abilitymodel

YEAR = "2018"
MONTHDAY = "1021"

if __name__ == "__main__":
    df = dataload.load()
    df = pd.merge(
        df.pipe(lambda df: df[
            (df['year'] == YEAR) & (df['monthday'] == MONTHDAY)
        ])[['kettonum']].drop_duplicates(),
        df
    ).sort_values(['year', 'monthday'])
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
    print(df.pipe(lambda df: df[
            (df['year'] == YEAR)
            & (df['monthday'] == MONTHDAY)
            & (df['racenum'] == '11')
        ])[['jyocd', 'racenum', 'bamei', 'predict']].sort_values(['jyocd', 'racenum', 'predict']))