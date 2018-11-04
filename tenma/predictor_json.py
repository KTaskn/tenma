# coding:utf-8
import json
import numpy as np
import pandas as pd
from tenma import dataload, abilitymodel as am, comparemodel as cm

YEAR = "2018"
MONTH = "11"
DAY = "04"

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
    df['jyoname'] = df['jyocd'].map(get_jyo)

    json_raceset = {}

    for year, monthday, jyocd, racenum in df.pipe(lambda df: df[
                (df['year'] == '2018') & 
                (df['monthday'].astype(int) >= 1000)
            ])[['year', 'monthday', 'jyocd', 'racenum']].drop_duplicates().values:

        if year not in json_raceset.keys():
            json_raceset[year] = {}
        
        if monthday not in json_raceset[year].keys():
            json_raceset[year][monthday] = {}

        if jyocd not in json_raceset[year][monthday].keys():
            json_raceset[year][monthday][jyocd] = []

        json_raceset[year][monthday][jyocd].append(racenum)

        json_race = {"children": []}

        for bamei, predict in df.pipe(lambda df: df[
                (df['year'] == year)
                & (df['monthday'] == monthday)
                & (df['jyocd'] == jyocd)
                & (df['racenum'] == racenum)
            ])[["bamei", "predict"]].values:
            json_race['children'].append(
                {
                    "name": bamei,
                    "val": predict
                }
            )
        with open("json/%s%s%s%s.json" % (year, monthday, jyocd, racenum), "w") as f:
            json.dump(json_race, f)
    
    with open("raceset.json", "w") as f:
        json.dump(json_raceset, f)