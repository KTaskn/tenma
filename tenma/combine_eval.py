# coding:utf-8
import os
import sys
import psycopg2
import numpy as np
import pandas as pd
from datetime import datetime
from tenma import dataload
from tenma import naivebayes as nb
import json
import pystan
import itertools
from pprint import pprint

def load():
    dbparams = "host={} user={} dbname={} port={}".format(
            os.environ['host'],
            os.environ['user'],
            os.environ['dbname'],
            os.environ['port'],
        )
    
    query = """
    SELECT
    n_uma_race.year,
    n_uma_race.monthday,
    n_uma_race.jyocd,
    n_uma_race.racenum,
    n_uma_race.kettonum,
    n_uma_race.kakuteijyuni,
    n_uma_race.bamei,
    SUM(CASE WHEN t.kakuteijyuni IN ('01', '02', '03') AND n_race.kyori <> t.kyori THEN 1 ELSE 0 END) AS win_other,
    SUM(CASE WHEN n_race.kyori <> t.kyori THEN 1 ELSE 0 END) AS race_other,
    SUM(CASE WHEN t.kakuteijyuni IN ('01', '02', '03') AND n_race.kyori = t.kyori THEN 1 ELSE 0 END) AS win_kyori,
    SUM(CASE WHEN n_race.kyori = t.kyori THEN 1 ELSE 0 END) AS race_kyori,
    SUM(CASE WHEN t.kakuteijyuni IN ('01', '02', '03') AND n_race.gradecd = t.gradecd THEN 1 ELSE 0 END) AS win_grade,
    SUM(CASE WHEN n_race.gradecd = t.gradecd THEN 1 ELSE 0 END) AS race_grade
    FROM n_uma_race
    INNER JOIN n_race
    ON n_uma_race.year = n_race.year
    AND n_uma_race.monthday = n_race.monthday
    AND n_uma_race.jyocd = n_race.jyocd
    AND n_uma_race.racenum = n_race.racenum
    INNER JOIN (
        SELECT
            DATE(CONCAT(
                n_uma_race.year,
                '-',
                SUBSTRING(n_uma_race.monthday, 0, 3),
                '-',
                SUBSTRING(n_uma_race.monthday, 3, 4))
            ) AS _date,
            n_uma_race.kettonum,
            n_uma_race.kakuteijyuni,
            n_race.kyori,
            n_race.gradecd
        FROM n_uma_race
        INNER JOIN n_race
        ON n_uma_race.year = n_race.year
        AND n_uma_race.monthday = n_race.monthday
        AND n_uma_race.jyocd = n_race.jyocd
        AND n_uma_race.racenum = n_race.racenum
    ) AS t
    ON n_uma_race.kettonum = t.kettonum
    AND DATE(
            CONCAT(
                n_uma_race.year,
                '-',
                SUBSTRING(n_uma_race.monthday, 0, 3),
                '-',
                SUBSTRING(n_uma_race.monthday, 3, 4)
            )
        ) > t._date
    WHERE n_uma_race.year in ('2019')
    GROUP BY n_uma_race.year,
    n_uma_race.monthday,
    n_uma_race.jyocd,
    n_uma_race.racenum,
    n_uma_race.kettonum,
    n_uma_race.bamei,
    n_uma_race.kakuteijyuni;
    """

    with psycopg2.connect(dbparams) as conn:
        df = pd.io.sql.read_sql_query(query, conn)
    df = df.sort_values(['year', 'monthday', 'jyocd'])
    print(df.columns)
    return df

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

DIC_SCORE = {
    1: 10.0,
    2: 5.0,
    3: 3.0,
    4: 2.0,
    5: 1.0,
}
def get_score(x):
    if x in DIC_SCORE.keys():
        return DIC_SCORE[x]
    else:
        return 0.0
    
if __name__ == "__main__":
    df = load()

    year = sys.argv[1]
    monthday = sys.argv[2]

    l_col = [
        "other",
        "grade",
        "kyori"
    ]

    mask = (df['year'] == year) & (df['kakuteijyuni'] != '00')
    df = df[mask].reset_index(drop=True)


    def zscore(x):
        return (x - x.mean()) / x.std()

    df['score'] = 0.0
    for i in range(100):
        df['other'] = np.random.beta(
            df['win_other'] + 0.001,
            df['race_other'] - df['win_other'] + 0.001
        )

        df['grade'] = np.random.beta(
            df['win_grade'] + 0.001,
            df['race_grade'] - df['win_grade'] + 0.001
        )

        df['kyori'] = np.random.beta(
            df['win_kyori'] + 0.001,
            df['race_kyori'] - df['win_kyori'] + 0.001
        )

        df_param = pd.read_csv('params.csv')

        for col in l_col:
            df[col] = df.groupby(['year', 'monthday', 'jyocd', 'racenum'])[col].transform(zscore)

        df['score'] += np.dot(df[l_col], df_param[["W.1", "W.2", "W.3"]].ix[i]) + df_param["bias"].ix[i]
    df['predict'] = df.groupby(['year', 'monthday', 'jyocd', 'racenum'])['score'].rank(ascending=False)
    # df[['year', 'monthday', 'jyocd', 'racenum', 'bamei', 'predict', "kakuteijyuni", "score"]].to_csv('evaluate.csv', index=False)

    from sklearn.metrics import confusion_matrix
    print(confusion_matrix((df['kakuteijyuni'] == "01"), (df['predict'] == 1)))
