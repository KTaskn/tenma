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

    df_text = df.pipe(lambda df: df[
        (df['year'] == '2018')
    &   (df['monthday'] == MONTH + DAY)
    ]).sort_values(['jyocd', 'racenum', 'predict']).groupby(
        ['jyocd', 'racenum']
    ).tail(1)[['jyoname', 'racenum', 'bamei']]
    df_text.columns = ['競馬場', 'レース番号', '馬名']
    text_table = df_text.to_html(index=False)

    text = """
    {}月{}日（）の競馬予想
    <p>人工知能による競馬予想。{}月{}日の注目馬をピックアップ</p>

    {}
    """.format(MONTH, DAY, MONTH, DAY, text_table)
    with open('blog_text.txt', 'w') as f:
        f.write(text)