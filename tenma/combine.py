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
import pickle

def load():
    dbparams = "host={} user={} dbname={} port={}".format(
            os.environ['host'],
            os.environ['user'],
            os.environ['dbname'],
            os.environ['port'],
        )
    
    query = """
	SELECT
	tbl_1.*,
	tbl_2.kyori,
    tbl_2.trackcd,
    tbl_2.kisyucode,
    tbl_2.chokyosicode,
    tbl_2.win_other,
    tbl_2.race_other,
    tbl_2.win_kyori,
    tbl_2.race_kyori,
    tbl_2.win_grade,
    tbl_2.race_grade
    FROM 
	(SELECT
    n_uma_race.year,
    n_uma_race.monthday,
    n_uma_race.jyocd,
    n_uma_race.racenum,
    n_uma_race.kettonum,
    n_uma_race.kakuteijyuni
    FROM n_uma_race
    WHERE n_uma_race.year in ('2018')) AS tbl_1
    LEFT JOIN
    (SELECT
    n_uma_race.year,
    n_uma_race.monthday,
    n_uma_race.jyocd,
    n_uma_race.racenum,
    n_race.kyori,
    n_race.trackcd,
    n_uma_race.kettonum,
    n_uma_race.kisyucode,
    n_uma_race.chokyosicode,
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
        AND n_uma_race.year::int > 2015
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
    AND n_uma_race.year in ('2018')
    WHERE n_uma_race.year in ('2018')
    GROUP BY n_uma_race.year,
    n_uma_race.monthday,
    n_uma_race.jyocd,
    n_uma_race.racenum,
    n_race.kyori,
    n_race.trackcd,
    n_uma_race.kettonum,
    n_uma_race.kisyucode,
    n_uma_race.chokyosicode,
    n_uma_race.kakuteijyuni) AS tbl_2
    ON tbl_1.year = tbl_2.year
    AND tbl_1.monthday = tbl_2.monthday
    AND tbl_1.jyocd = tbl_2.jyocd
    AND tbl_1.racenum = tbl_2.racenum
    AND tbl_1.kettonum = tbl_2.kettonum;
    """

    PATH = "2018_race_combine.csv"
    if os.path.exists(PATH):
        df = pd.read_csv(PATH, dtype=str).pipe(lambda df: df[df['jyocd'].map(
            lambda x: x in ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
        )])

        df['win_other'] = df['win_other'].fillna("0.0").astype(float).astype(int)
        df['race_other'] = df['race_other'].fillna("0.0").astype(float).astype(int)
        df['win_kyori'] = df['win_kyori'].fillna("0.0").astype(float).astype(int)
        df['race_kyori'] = df['race_kyori'].fillna("0.0").astype(float).astype(int)
        df['win_grade'] = df['win_grade'].fillna("0.0").astype(float).astype(int)
        df['race_grade'] = df['race_grade'].fillna("0.0").astype(float).astype(int)
    else:
        with psycopg2.connect(dbparams) as conn:
            df = pd.io.sql.read_sql_query(query, conn)
            df.to_csv(PATH, index=False)
    df = df.sort_values(['year', 'monthday', 'jyocd'])
    return df

def load_kisyu():
    dbparams = "host={} user={} dbname={} port={}".format(
            os.environ['host'],
            os.environ['user'],
            os.environ['dbname'],
            os.environ['port'],
        )
    
    query = """
    SELECT
    n_uma_race.kisyucode,
     DATE(DATE_TRUNC('month', DATE(
            CONCAT(
                n_uma_race.year,
                '-',
                SUBSTRING(n_uma_race.monthday, 0, 3),
                '-',
                SUBSTRING(n_uma_race.monthday, 3, 4)
            )
        ) - 14)) AS _month,
    SUM(CASE WHEN n_uma_race.kakuteijyuni IN ('01', '02', '03') THEN 1 ELSE 0 END) AS win_kisyu,
    COUNT(n_uma_race.kakuteijyuni) AS race_kisyu
    FROM n_uma_race
    WHERE n_uma_race.year IN ('2017', '2018')
    GROUP BY n_uma_race.kisyucode, _month;
    """


    PATH = "2018_race_kisyu.csv"
    if os.path.exists(PATH):
        df = pd.read_csv(PATH, dtype=str)
        df['win_kisyu'] = df['win_kisyu'].astype(int)
        df['race_kisyu'] = df['race_kisyu'].astype(int)
        df['_month'] = df['_month'].map(lambda x: datetime.strptime(x, '%Y-%m-%d'))
    else:
        with psycopg2.connect(dbparams) as conn:
            df = pd.io.sql.read_sql_query(query, conn)
            df.to_csv(PATH, index=False)
    return df

def load_chokyo():
    dbparams = "host={} user={} dbname={} port={}".format(
            os.environ['host'],
            os.environ['user'],
            os.environ['dbname'],
            os.environ['port'],
        )
    
    query = """
    SELECT
    n_uma_race.chokyosicode,
     DATE(DATE_TRUNC('month', DATE(
            CONCAT(
                n_uma_race.year,
                '-',
                SUBSTRING(n_uma_race.monthday, 0, 3),
                '-',
                SUBSTRING(n_uma_race.monthday, 3, 4)
            )
        ) - 14)) AS _month,
    SUM(CASE WHEN n_uma_race.kakuteijyuni IN ('01', '02', '03') THEN 1 ELSE 0 END) AS win_chokyo,
    COUNT(n_uma_race.kakuteijyuni) AS race_chokyo
    FROM n_uma_race
    WHERE n_uma_race.year IN ('2017', '2018')
    GROUP BY n_uma_race.chokyosicode, _month;
    """


    PATH = "2018_race_chokyo.csv"
    if os.path.exists(PATH):
        df = pd.read_csv(PATH, dtype=str)
        df['win_chokyo'] = df['win_chokyo'].astype(int)
        df['race_chokyo'] = df['race_chokyo'].astype(int)
        df['_month'] = df['_month'].map(lambda x: datetime.strptime(x, '%Y-%m-%d'))
    else:
        with psycopg2.connect(dbparams) as conn:
            df = pd.io.sql.read_sql_query(query, conn)
            df.to_csv(PATH, index=False)
    return df


def load_titiuma():
    dbparams = "host={} user={} dbname={} port={}".format(
            os.environ['host'],
            os.environ['user'],
            os.environ['dbname'],
            os.environ['port'],
        )
    
    query = """
    SELECT
    n_uma.kettonum,
    t.*
    FROM n_uma
    INNER JOIN
    (SELECT
    n_race.trackcd,
    n_race.kyori,
    n_uma.Ketto3InfoHansyokuNum1,
    SUM(CASE WHEN n_uma_race.kakuteijyuni IN ('01', '02', '03') THEN 1 ELSE 0 END) AS win_titiuma,
    COUNT(n_uma_race.kakuteijyuni) AS race_titiuma
    FROM n_uma_race
    INNER JOIN n_uma ON n_uma_race.kettonum = n_uma.kettonum
    INNER JOIN n_race 
    ON n_uma_race.year = n_race.year
    AND n_uma_race.monthday = n_race.monthday
    AND n_uma_race.jyocd = n_race.jyocd
    AND n_uma_race.racenum = n_race.racenum
    GROUP BY n_uma.Ketto3InfoHansyokuNum1, n_race.trackcd, n_race.kyori) AS t
    ON n_uma.Ketto3InfoHansyokuNum1 = t.Ketto3InfoHansyokuNum1;
    """

    PATH = "2018_race_titiuma.csv"
    if os.path.exists(PATH):
        df = pd.read_csv(PATH, dtype=str)
        df['win_titiuma'] = df['win_titiuma'].astype(int)
        df['race_titiuma'] = df['race_titiuma'].astype(int)
    else:
        with psycopg2.connect(dbparams) as conn:
            df = pd.io.sql.read_sql_query(query, conn)
            df.to_csv(PATH, index=False)
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
    5: 1.0
}
def get_score(x):
    if x in DIC_SCORE.keys():
        return DIC_SCORE[x]
    else:
        return 0.0
    
if __name__ == "__main__":
    """
    df = pd.merge(
        pd.merge(
            pd.merge(
                load().pipe(lambda df: df.assign(
                    _month = df['year'] + "-" + df['monthday'].map(lambda x: x[:2])
                )),
                load_kisyu().pipe(lambda df: df.assign(
                    _month = df['_month'].map(lambda x: x.strftime('%Y-%m'))
                )),
                on=['kisyucode', '_month'],
                how="left"
            ),
            load_chokyo().pipe(lambda df: df.assign(
                _month = df['_month'].map(lambda x: x.strftime('%Y-%m'))
            )),
            on=['chokyosicode', '_month'],
            how="left"
        ),
        load_titiuma(),
        on=['kettonum', 'trackcd', 'kyori'],
        how="left"
    ).fillna(0.0)
    print(df.columns)

    # year = sys.argv[1]
    # monthday = sys.argv[2]
    year = "2018"
    monthday = "0000"

    l_col = [
        "other",
        "grade",
        "kyori",
        "kisyu",
        "chokyo",
        "titiuma"
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

    df['kisyu'] = np.random.beta(
        df['win_kisyu'] + 0.001,
        df['race_kisyu'] - df['win_kisyu'] + 0.001
    )

    df['chokyo'] = np.random.beta(
        df['win_chokyo'] + 0.001,
        df['race_chokyo'] - df['win_chokyo'] + 0.001
    )

    df['titiuma'] = np.random.beta(
        df['win_titiuma'] + 0.001,
        df['race_titiuma'] - df['win_titiuma'] + 0.001
    )

    df['score'] = df['kakuteijyuni'].astype(int).map(get_score)


    mask = (df['year'] == year) & (df['kakuteijyuni'] != '00') #& (df['monthday'] == "0106") 
    H = 5
    data_ppd = []
    data_x = []

    def zscore(x):
        v = x.std()
        if v:
            return (x - x.mean()) / v
        else:
            return 0.0

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

    d = pd.Series(data['Y'])
    mask = np.random.random(len(d)) < d.map(lambda x: x ** 1.2) * 25
    print(len(d[mask]))

    data = {
        "N": len(d[mask]),
        "H": data['H'],
        "D": data['D'],
        "X": np.array(data['X'])[mask],
        "Y": np.array(data['Y'])[mask]
    }

    STAN_MODEL_PATH = "stanmodel/combine.stan"
    model = pystan.StanModel(file=STAN_MODEL_PATH)

    # fit_vb = model.vb(data=data, pars=["W", "bias"],
    #         iter=3000,tol_rel_obj=0.0001,eval_elbo=100)
    # df = pd.read_csv(fit_vb['args']['sample_file'].decode('utf-8'), comment='#')
    # print(df)
    # df.to_csv('result.csv', index=False)

    fit = model.sampling(data=data, iter=3000, chains=3, thin=1, pars=["W", "bias"])
    samples = fit.extract(permuted=True)
    
    with open("result.pkl", "wb") as f:
        pickle.dump(samples, f)