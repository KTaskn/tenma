# coding:utf-8
import json
import numpy as np
import pandas as pd
from tenma import dataload, abilitymodel as am, comparemodel as cm

YEAR = "2018"
MONTHDAY = "1028"
JYOCD = "05"
RACENUM = "11"

if __name__ == "__main__":
    df = dataload.load()
    df['predict_comp'] = cm.predict(df)
    print(df.pipe(lambda df: df[
            (df['year'] == YEAR)
            & (df['monthday'] == MONTHDAY)
            & (df['jyocd'] == JYOCD)
            & (df['racenum'] == RACENUM)
        ])[['jyocd', 'racenum', 'bamei', 'predict']].sort_values(['jyocd', 'racenum', 'predict']))