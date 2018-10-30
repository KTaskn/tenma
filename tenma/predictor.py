# coding:utf-8
import json
import numpy as np
import pandas as pd
from tenma import dataload, abilitymodel as am, comparemodel as cm
from tenma.rulemodel import Rule

YEAR = "2018"
MONTHDAY = "1028"
JYOCD = "05"
RACENUM = "11"

if __name__ == "__main__":
    df = dataload.load()
    df['odds'] = df['odds'].astype(np.int32) / 10.0
    df['reward'] = ((df['kakuteijyuni'] == "01") * df['odds']) + ((df['kakuteijyuni'] != "01") * -1.0)
    df['predict'] = am.predict(df)
    df = df.pipe(lambda df: df[df['year'] == "2018"])

    result = []
    rule = Rule("rule.csv")
    for x, y, reward in df[['odds', 'predict', 'reward']].values:
        action = rule.get_nearly([x, y])
        if action:
            result.append(reward)
        else:
            result.append(0)

    print(result)
    print(sum(result))