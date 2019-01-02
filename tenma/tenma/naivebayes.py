# coding:utf-8
import numpy as np
import pandas as pd
from tenma import dataload
from datetime import datetime

# P(A)
def P_A(df, col_A, A):
    if(col_A not in df.columns):
        raise KeyError("%sはデータフレームに存在しない列です" % col_A)

    if(len(df.index) == 0):
        raise RuntimeError("データフレームにデータが存在しません")

    numerator = (df[col_A] == A).astype(int).sum()

    if numerator <= 0 :
        return np.log10(1.0 / (len(df[col_A].index) + len(df.index)))
        #return -50.0
    else:
        return np.log10(numerator / len(df[col_A].index))

# P(A|B)
def P_AB(df, col_A, A, col_B, B):
    if(len(df.index) == 0):
        raise RuntimeError("データフレームにデータが存在しません")

    if(col_A not in df.columns):
        raise KeyError("%sはデータフレームに存在しない列です" % col_A)

    if(col_B not in df.columns):
        raise KeyError("%sはデータフレームに存在しない列です" % col_B)

    mask = (df[col_B] == B)
    numerator = (df[mask][col_A] == A).astype(int).sum()
    if numerator <= 0.0: 
        return np.log10(1.0 / (mask.astype(int).sum() + len(df.index)))
        #return -50.0
    else:
        return np.log10(numerator / mask.astype(int).sum())

# P_BA
def P_BA(df, P_B, col_A, A, col_B, B):
    return P_AB(df, col_A, A, col_B, B) + P_B - P_A(df, col_A, A)

def NaiveBayes(df, P_B, l_col_A, l_A, col_B, B):
    p_b = P_B
    for col_A, A in zip(l_col_A, l_A):
        p_b = P_BA(df, p_b, col_A, A, col_B, B)
    return p_b

def NaiveBayesModel(df):
    l_col = [
        "kakuteijyuni_bf1",
        "kakuteijyuni_bf2",
        "kakuteijyuni_bf3",
        "kakuteijyuni_bf4",
        "ketto3infohansyokunum1",
        "kisyucode"
    ]

    result = [np.nan] * len(df.index)
    df_row = df.pipe(lambda df: df[
        (df['year'] == '2018')
         #& (df['monthday'].astype(int) == 1000)
         & (df['monthday'] == '1223')
    ]).reset_index(drop=True)


    df_output = pd.DataFrame([], columns=['year', 'monthday', 'jyocd', 'racenum', 'bamei1', 'bamei2', 'odds'])
    for _, grp in df_row.groupby(['year', 'monthday', 'jyocd', 'racenum']):
        dic_p = {
            "01": {},
            "02": {},
            "03": {},
        }
        for idx, row_1 in grp.iterrows():
            df_tmp = df.pipe(lambda df: df[
                (df['unixtime'] < row_1['unixtime'])
            & (df['unixtime'] > row_1['unixtime'] - 24 * 60 * 60 * 90)
            & (df['smile'] == row_1['smile'])
            & (df['isturf'] == row_1['isturf'])
            ])
            if len(df_tmp.index) == 0:
                df_tmp = df.pipe(lambda df: df[
                        (df['unixtime'] < row_1['unixtime'])
                    & (df['smile'] == row_1['smile'])
                    & (df['isturf'] == row_1['isturf'])
                    ])

            for jyuni in dic_p.keys():
                p = np.log10(1 / 18.0)
                dic_p[jyuni][row_1['bamei']] = NaiveBayes(
                    df_tmp,
                    p,
                    l_col,
                    row_1[l_col].values,
                    "kakuteijyuni",
                    jyuni
                )
    
        l_tmp = []
        for idx, row_1 in grp.iterrows():
            for idx, row_2 in grp.iterrows():
                    if row_1['bamei'] != row_2['bamei']:
                                l_tmp.append([
                                    row_1['year'],
                                    row_1['monthday'],
                                    row_1['jyocd'],
                                    row_1['racenum'],
                                    row_1['bamei'],
                                    row_2['bamei'],
                                    dic_p["01"][row_1['bamei']]
                                        + dic_p["02"][row_2['bamei']]
                                ])
        df_tmp = pd.DataFrame(l_tmp, columns=['year', 'monthday', 'jyocd', 'racenum', 'bamei1', 'bamei2', 'odds'])
        df_tmp['odds'] = df_tmp['odds'].map(lambda x: 10 ** x)
        df_tmp['odds'] = df_tmp['odds'] / df_tmp['odds'].sum()
        df_tmp['odds'] = 1.0 / df_tmp['odds']
        df_output = pd.concat([
            df_output,
            df_tmp
        ], ignore_index=True)

    return result, df_output.sort_values(['year', 'monthday', 'jyocd', 'racenum', 'odds'])
