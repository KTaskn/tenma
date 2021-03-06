# coding:utf-8
import sys
import pandas as pd
from datetime import datetime
from tenma import dataload
from tenma import naivebayes as nb
import numpy as np
from sklearn.linear_model import RidgeClassifier

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
    
    from sklearn.linear_model import RidgeClassifier, LogisticRegression

    mask = (df['year'] != "2018")
    for col in l_col:
        df = pd.merge(
            df,
            pd.merge(
                df[mask][col].value_counts().reset_index().rename(columns={col: "%s_%s" % (col, "cnt_all"), "index": col}),
                df[mask].pipe(
                    lambda df: df[df['kakuteijyuni'] == "01"]
                )[col].value_counts().reset_index().rename(columns={col: "%s_%s" % (col, "cnt_win"), "index": col}),
                how="left"
            ).fillna(0)
        )

    df_tmp = df[mask].pipe(lambda df: df[df['kakuteijyuni'] == "01"])
    df_learn = pd.concat([
        df_tmp,
        df[mask].pipe(lambda df: df[df['kakuteijyuni'] != "01"]).iloc[:len(df_tmp.index)],
    ], ignore_index=True)


    mask = (df['year'] == "2019") & (df['kakuteijyuni'] != '00')
    df_predict = df[mask].reset_index(drop=True)

    df_X = pd.DataFrame([])
    for col in l_col:
        df_X[col] = np.random.beta(
            df_learn["%s_%s" % (col, "cnt_win")].values + 1.0,
            df_learn["%s_%s" % (col, "cnt_all")].values - df_learn["%s_%s" % (col, "cnt_win")].values + 1.0
        )


    rc = RidgeClassifier()
    rc.fit(df_X.values, (df_learn['kakuteijyuni'] == "01").astype(int))

    df_X_pre = pd.DataFrame([])
    df_predict['proba'] = 0
    for i in range(100):
        for col in l_col:
            df_X_pre[col] = np.random.beta(
                df_predict["%s_%s" % (col, "cnt_win")].values + 1.0,
                df_predict["%s_%s" % (col, "cnt_all")].values - df_predict["%s_%s" % (col, "cnt_win")].values + 1.0
            )
        df_predict['proba'] += 1 / (1 + np.exp(-(np.dot(df_X_pre.values, rc.coef_[0]) + rc.intercept_[0])))
    df_predict['predict'] = df_predict.groupby(['year', 'monthday', 'jyocd', 'racenum'])['proba'].rank(ascending=False)


    from sklearn.metrics import confusion_matrix
    print(confusion_matrix(
        (df_predict['kakuteijyuni'].map(lambda x: x in ["01", "02", "03"])),
        (df_predict['predict'].map(lambda x: x in [1, 2, 3]))
    ))
        