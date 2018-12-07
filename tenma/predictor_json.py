# coding:utf-8
import json
import numpy as np
import pandas as pd
from tenma import dataload, abilitymodel as am, comparemodel as cm

YEAR = "2018"
MONTHDAY = ["1201", "1202"]

dic_jyo = {
    '01': '札幌',
    '02': '函館',
    '03': '福島',
    '04': '新潟',
    '05': '東京',
    '06': '中山',
    '07': '中京',
    '08': '京都',
    '09': '阪神',
    '10': '小倉'
}

def func(x):
    if x < 1:
        return 1
    else:
        return int(x)


def get_jyo(x):
    if x in dic_jyo.keys():
        return dic_jyo[x]
    else:
        return ''

if __name__ == "__main__":
    df = dataload.load()
    df['predict'] = am.predict(df)
    df = df.pipe(lambda df: df[
        (df['year'] == '2018')
    &   (df['monthday'].map(lambda x: x in MONTHDAY))
    ])

    df['ranking'] = df.groupby(['year', 'monthday', 'jyocd', 'racenum'])['predict'].rank(ascending=False)

    df[['year', 'monthday', 'jyocd', 'racenum', 'kettonum', 'ranking']].to_csv('output.csv', index=False)
    df[['kettonum', 'bamei']].to_csv('bamei.csv', index=False)