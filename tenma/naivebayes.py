# coding:utf-8
import pandas as pd
from datetime import datetime
from tenma import dataload
from tenma import naivebayes as nb

def smile(x):
    if x <= 1300:
        return "s"
    if x > 1300 and x < 1900:
        return "m"
    if x >= 1900 and x <= 2100:
        return "i"
    #if x > 2100 and x <= 2700:
    return "l"
    #return "e"

def get_track(x):
    DIC_TRACK = {
        '10' :'turf',
        '11' :'turf',
        '12' :'turf',
        '13' :'turf',
        '14' :'turf',
        '15' :'turf',
        '16' :'turf',
        '17' :'turf',
        '18' :'turf',
        '19' :'turf',
        '20' :'turf',
        '21' :'turf',
        '22' :'turf',
        '23' :'dirt',
        '24' :'dirt',
        '25' :'dirt',
        '26' :'dirt',
        '27' :'dirt',
        '28' :'dirt',
        '29' :'dirt',
        '51' :'hurdle',
        '52' :'hurdle',
        '53' :'hurdle',
        '54' :'hurdle',
        '55' :'hurdle',
        '56' :'hurdle',
        '57' :'hurdle',
        '58' :'hurdle',
        '59' :'hurdle',
    }
    if x in DIC_TRACK.keys():
        return DIC_TRACK[x]
    else:
        return 'other'


if __name__ == "__main__":
    df = dataload.load()

    def to_unixtime(x):
        return int(datetime.strptime(x, '%Y%m%d').timestamp())

    l_col = [
        "kakuteijyuni_bf1",
        "kakuteijyuni_bf2",
        "kakuteijyuni_bf3",
        "kakuteijyuni_bf4",
        "ketto3infohansyokunum1",
        "kisyucode"
    ]

    df['unixtime'] = (df['year'] + df['monthday']).map(to_unixtime)
    df['smile'] = df['kyori'].astype(int).map(smile)
    df['isturf'] = df['trackcd'].map(get_track)
    df['kakuteijyuni_bf1'] = df.groupby('kettonum')['kakuteijyuni'].shift(1).fillna(-1)
    df['kakuteijyuni_bf2'] = df.groupby('kettonum')['kakuteijyuni'].shift(2).fillna(-1)
    df['kakuteijyuni_bf3'] = df.groupby('kettonum')['kakuteijyuni'].shift(3).fillna(-1)
    df['kakuteijyuni_bf4'] = df.groupby('kettonum')['kakuteijyuni'].shift(4).fillna(-1)

    mask = (df['year'] == '2018') & (df['monthday'] == '1228') & (df['jyocd'] == '06') & (df['racenum'] == '11')
    df_output = nb.NaiveBayesModel(df, df[mask].reset_index(), l_col, 2)

    df_output.sort_values(["year", 'monthday', 'jyocd', 'racenum', 'odds']).to_csv('output_tan.csv', index=False)