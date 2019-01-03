# coding:utf-8
from itertools import permutations
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

def make_dic_p(df_all, df_grp, l_col, NUM):
    dic_p = dict(zip(map(lambda x: "%02d" % x, range(1, NUM + 1)), [{}] * NUM))

    # 順位における確率を計算
    for jyuni in dic_p.keys():
        dic_tmp = {}
        for _, row in df_grp.iterrows():
            # 過去の期間、レース条件を制限する
            df_jyoken = df_all.pipe(lambda df: df[
                (df['unixtime'] < row['unixtime'])
                & (df['unixtime'] > row['unixtime'] - 24 * 60 * 60 * 90)
                & (df['smile'] == row['smile'])
                & (df['isturf'] == row['isturf'])
            ])

            p = np.log10(1 / 18.0)
            dic_tmp[row['kettonum']] = NaiveBayes(
                df_jyoken,
                p,
                l_col,
                row[l_col].values,
                "kakuteijyuni",
                jyuni
            )
        dic_p[jyuni] = dic_tmp
    return dic_p

def get_p_rentan(tpl, dic):
    p = 0.0
    for idx, id in enumerate(tpl):
        p += dic["%02d" % (idx + 1)][id]
    return p

def NaiveBayesModel(df, df_target, l_col, NUM):
    df_output = pd.DataFrame([])
    grp_col = ['year', 'monthday', 'jyocd', 'racenum']
    for _, df_grp in df_target.groupby(grp_col):

        # 確率のリストを作成する
        dic_p = make_dic_p(df, df_grp, l_col, NUM)

        # 中身が重複しているやつは削除
        func = lambda input: len(input) == len(list(set(input)))
        l_tpl = list(filter(func, list(permutations(df_grp['kettonum'].values, NUM))))
        l_p = list(map(lambda row: get_p_rentan(row, dic_p), l_tpl))

        # 結果のデータフレームを作成
        # 血糖番号の順列のデータフレーム
        df_tmp = pd.DataFrame(l_tpl, columns=list(map(lambda x: "kettonum_%d" % x, range(1, NUM + 1))))
        # 計算したオッズの列を追加
        df_tmp['odds'] = l_p
        df_tmp['odds'] = df_tmp['odds'].map(lambda x: 10 ** x)
        df_tmp['odds'] = df_tmp['odds'] / df_tmp['odds'].sum()
        df_tmp['odds'] = (1.0 / df_tmp['odds']).round(1)
        # レース番号などを追加
        for idx, col in enumerate(grp_col):
            df_tmp[col] = _[idx]
        # 他のレースとのデータフレームと結合する
        df_output = pd.concat([
            df_output,
            df_tmp
        ], ignore_index=True)

    return df_output