# coding:utf-8
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from tenma import dataload
from tenma import naivebayes as nb
import pystan
import itertools

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
    1: 1.0,
    2: 0.8,
    3: 0.5
}
def get_score(x):
    if x in DIC_SCORE.keys():
        return DIC_SCORE[x]
    else:
        return 0.0
    
if __name__ == "__main__":
    df = dataload.load()

    year = sys.argv[1]
    monthday = sys.argv[2]

    def to_unixtime(x):
        return int(datetime.strptime(x, '%Y%m%d').timestamp())

    l_col = [
        "kakuteijyuni_bf1",
        "kakuteijyuni_bf2",
        "kakuteijyuni_bf3",
        "kakuteijyuni_bf4",
        # "ketto3infohansyokunum1",
        # "kisyucode"
    ]

    df['unixtime'] = (df['year'] + df['monthday']).map(to_unixtime)
    df['smile'] = df['kyori'].astype(int).map(smile)
    df['isturf'] = df['trackcd'].map(get_track)
    df['kakuteijyuni_bf1'] = df.groupby('kettonum')['kakuteijyuni'].shift(1).fillna(-1).astype(int)
    df['kakuteijyuni_bf2'] = df.groupby('kettonum')['kakuteijyuni'].shift(2).fillna(-1).astype(int)
    df['kakuteijyuni_bf3'] = df.groupby('kettonum')['kakuteijyuni'].shift(3).fillna(-1).astype(int)
    df['kakuteijyuni_bf4'] = df.groupby('kettonum')['kakuteijyuni'].shift(4).fillna(-1).astype(int)

    df['score'] = df['kakuteijyuni'].astype(int).map(get_score)

    mask = (df['year'] == year) & (df['monthday'] == monthday) & (df['kakuteijyuni'] != '00')

    H = 5
    data_ppd = []
    data_x = []
    for idx_grp, grp in df[mask].groupby(['year', 'monthday', 'jyocd', 'racenum']):
        print(idx_grp)
        grp = grp.sort_values('kakuteijyuni')
        grp.index = grp['kakuteijyuni'].values
        l_comb = np.array(list(itertools.combinations(grp['kakuteijyuni'].values, H)))
        for a_comb in l_comb[np.random.random(len(l_comb)) < 0.05]:
            ppd = 1.0
            row_x = []
            for idx_row, row in grp.T[list(a_comb)].T.iterrows():
                ppd *= np.exp(row['score']) / grp.ix[idx_row:, "score"].map(np.exp).sum()
                row_x.append(row[l_col].values.tolist())
            data_x.append(row_x)
            data_ppd.append(ppd)

    data = {
        "N": len(data_x),
        "H": H,
        "D": len(l_col),
        "X": data_x,
        "Y": data_ppd,
    }

    STAN_MODEL_PATH = "stanmodel/combine.stan"
    model = pystan.StanModel(file=STAN_MODEL_PATH)
    fit_vb = model.vb(data=data, pars=["W", "bias"],
            iter=5000,tol_rel_obj=0.0001,eval_elbo=100)
    
    ms = pd.read_csv(fit_vb['args']['sample_file'].decode('utf-8'), comment='#')

    print(ms)