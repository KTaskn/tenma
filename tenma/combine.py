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
    WHERE n_uma_race.year in ('2018')
    GROUP BY n_uma_race.year,
    n_uma_race.monthday,
    n_uma_race.jyocd,
    n_uma_race.racenum,
    n_uma_race.kettonum,
    n_uma_race.kakuteijyuni;
    """

    PATH = "2018_race_combine.csv"
    if os.path.exists(PATH):
        df = pd.read_csv(PATH, dtype=str).pipe(lambda df: df[df['jyocd'].map(
            lambda x: x in ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
        )])
        df['win_other'] = df['win_other'].astype(int)
        df['race_other'] = df['race_other'].astype(int)
        df['win_kyori'] = df['win_kyori'].astype(int)
        df['race_kyori'] = df['race_kyori'].astype(int)
        df['win_grade'] = df['win_grade'].astype(int)
        df['race_grade'] = df['race_grade'].astype(int)
    else:
        with psycopg2.connect(dbparams) as conn:
            df = pd.io.sql.read_sql_query(query, conn)
            df.to_csv(PATH, index=False)
    df = df.sort_values(['year', 'monthday', 'jyocd'])
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
    3: 1.0
}
def get_score(x):
    if x in DIC_SCORE.keys():
        return DIC_SCORE[x]
    else:
        return 0.0
    
if __name__ == "__main__":
    """
    df = load()

    # year = sys.argv[1]
    # monthday = sys.argv[2]
    year = "2018"
    monthday = "0000"

    l_col = [
        "other",
        "grade",
        "kyori"
    ]

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

    df['score'] = df['kakuteijyuni'].astype(int).map(get_score)


    mask = (df['year'] == year) & (df['kakuteijyuni'] != '00') #& (df['monthday'] == "0106") 
    H = 5
    data_ppd = []
    data_x = []

    def zscore(x):
        return (x - x.mean()) / x.std()

    for col in l_col:
        df[col] = df.groupby(['year', 'monthday', 'jyocd', 'racenum'])[col].transform(zscore)

    for idx_grp, grp in df[mask].groupby(['year', 'monthday', 'jyocd', 'racenum']):
        grp = grp.sort_values('kakuteijyuni')
        grp.index = grp['kakuteijyuni'].values
        l_comb = list(itertools.combinations(grp['kakuteijyuni'].values, H))
        l_comb = np.array(list(filter(lambda tpl: ("01" in tpl) or ("02" in tpl) or ("03" in tpl), l_comb)))
        print(idx_grp, len(l_comb))
        for a_comb in l_comb[np.random.random(len(l_comb)) < 0.05]:
            ppd = 1.0
            row_x = []
            i = 0
            for idx_row, row in grp.T[list(a_comb)].T.iterrows():
                if i >= 5:
                    break
                ppd *= np.exp(row['score']) / grp.ix[idx_row:, "score"].map(np.exp).sum()
                row_x.append(row[l_col].values.tolist())
                i += 1

            data_x.append(row_x)
            data_ppd.append(ppd)

    print(len(data_x))


    data = {
        "N": len(data_x),
        "H": H,
        "D": len(l_col),
        "X": data_x,
        "Y": data_ppd,
    }


    with open("data_dict.json", "w") as f:
        json.dump(data, f, indent="\t")
        del(df)
        del(data_x)
        del(data_ppd)
    """

    with open("data_dict.json") as f:
        data = json.load(f)

    # N = 5000
    # data = {
    #     "N": N,
    #     "H": data['H'],
    #     "D": data['D'],
    #     "X": data['X'][:N],
    #     "Y": data['Y'][:N],
    # }

    STAN_MODEL_PATH = "stanmodel/combine.stan"
    model = pystan.StanModel(file=STAN_MODEL_PATH)

    fit_vb = model.vb(data=data, pars=["W", "bias"],
            iter=3000,tol_rel_obj=0.0001,eval_elbo=100)
    df = pd.read_csv(fit_vb['args']['sample_file'].decode('utf-8'), comment='#')
    print(df)
    df.to_csv('result.csv', index=False)

    # fit = model.sampling(data=data, iter=3000, chains=3, thin=1, pars=["W", "bias"])
    # samples = fit.extract(permuted=True)
    # df = pd.DataFrame(samples)
    # df.to_csv('result.csv', index=False)
    
