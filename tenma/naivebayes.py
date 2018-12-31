# coding:utf-8
import pandas as pd
from datetime import datetime
from tenma import dataload
from tenma import naivebayes as nb

if __name__ == "__main__":
    df = dataload.load()

    def to_unixtime(x):
        return int(datetime.strptime(x, '%Y%m%d').timestamp())

    df['unixtime'] = (df['year'] + df['monthday']).map(to_unixtime)
    df['kakuteijyuni_bf1'] = df.groupby('kettonum')['kakuteijyuni'].shift(1).fillna(-1)
    df['kakuteijyuni_bf2'] = df.groupby('kettonum')['kakuteijyuni'].shift(2).fillna(-1)
    df['kakuteijyuni_bf3'] = df.groupby('kettonum')['kakuteijyuni'].shift(3).fillna(-1)
    df['kakuteijyuni_bf4'] = df.groupby('kettonum')['kakuteijyuni'].shift(4).fillna(-1)
    df = df.pipe(lambda df: df[df['unixtime'] > int(datetime.now().timestamp()) - 24 * 60 * 60 * 90])
    result = nb.NaiveBayesModel(df)
    df['result'] = result

    df.to_csv('output.csv', index=False)