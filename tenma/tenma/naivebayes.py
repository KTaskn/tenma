# coding:utf-8
from itertools import permutations
import numpy as np
import pandas as pd
from tenma import dataload
from datetime import datetime

class NaiveBayesSummary():
    def __init__(self):
        self.__probability = np.nan
        self.__summary = []

    def set_probability(self, p):
        self.__probability = p
    
    def set_entity(self, name, p):
        self.__summary.append((name, p))

    def get_probability(self):
        return self.__probability

    def get_entity(self, name=None):
        if name is None:
            return self.__summary
        else:
            return dict(self.__summary)[name]

    def get_bad_entity(self, num=0):
        return sorted(self.__summary, key=lambda x: x[1])[num]

    
    def get_good_entity(self, num=0):
        return sorted(self.__summary, key=lambda x: x[1], reverse=True)[num]


class NaiveBayes():
    def __init__(self, df, P_B, l_name_A, l_jyoken_A, name_B, jyoken_B):
        self.__summary = NaiveBayesSummary()

        self.df = df
        self.p_b = P_B
        self.l_name_A = l_name_A
        self.l_jyoken_A = l_jyoken_A
        self.name_B = name_B
        self.jyoken_B = jyoken_B

    def predict(self):
        self.__summary = NaiveBayesSummary()
        _df = self.df
        _p_b = self.p_b
        _name_B = self.name_B
        _jyoken_B = self.jyoken_B
        for name_A, jyoken_A in zip(self.l_name_A, self.l_jyoken_A):
            _p_b = self.__P_BA(_df, _p_b, name_A, jyoken_A, _name_B, _jyoken_B)
        self.__summary.set_probability(_p_b)
    
    def get_summary(self):
        return self.__summary

    # P(A)
    def __P_A(self, df, col_A, A):
        if(col_A not in df.columns):
            raise KeyError("%sはデータフレームに存在しない列です" % col_A)

        if(len(df.index) == 0):
            raise RuntimeError("データフレームにデータが存在しません")

        numerator = (df[col_A] == A).astype(int).sum()

        if numerator <= 0 :
            return np.log10(1.0 / (len(df[col_A].index) + len(df.index)))
        else:
            return np.log10(numerator / len(df[col_A].index))

    # P(A|B)
    def __P_AB(self, df, col_A, A, col_B, B):
        if(len(df.index) == 0):
            raise RuntimeError("データフレームにデータが存在しません")

        if(col_A not in df.columns):
            raise KeyError("%sはデータフレームに存在しない列です" % col_A)

        if(col_B not in df.columns):
            raise KeyError("%sはデータフレームに存在しない列です" % col_B)

        mask = (df[col_B] == B)
        numerator = (df[mask][col_A] == A).astype(int).sum()
        if numerator <= 0.0:
            result_p = np.log10(1.0 / (mask.astype(int).sum() + len(df.index)))
        else:
            result_p = np.log10(numerator / mask.astype(int).sum())

        return result_p

    # P_BA
    def __P_BA(self, df, P_B, col_A, A, col_B, B):
        P_AB = self.__P_AB(df, col_A, A, col_B, B)
        P_A = self.__P_A(df, col_A, A)
        self.__summary.set_entity(col_A, P_AB - P_A)
        return P_AB + P_B - P_A

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
            nb = NaiveBayes(
                df_jyoken,
                p,
                l_col,
                row[l_col].values,
                "kakuteijyuni",
                jyuni
            )
            nb.predict()
            dic_tmp[row['kettonum']] = nb.get_summary()
        dic_p[jyuni] = dic_tmp
    return dic_p

def get_p_rentan(tpl, dic):
    p = 0.0
    for idx, id in enumerate(tpl):
        p += dic["%02d" % (idx + 1)][id].get_probability()
    return p

def get_NaiveBayesProbability(df, df_target, l_col, NUM):
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

def get_NaiveBayesEntities(df, df_target, l_col, NUM):
    df_output = pd.DataFrame([])
    grp_col = ['year', 'monthday', 'jyocd', 'racenum']
    for _, df_grp in df_target.groupby(grp_col):

        # 確率のリストを作成する
        dic_p = make_dic_p(df, df_grp, l_col, NUM)

        for num in range(len(l_col)):
            df_tmp = pd.DataFrame(df_grp['kettonum'], columns=["kettonum"])
            df_tmp['factor'] = list(map(lambda row: dic_p["01"][row].get_good_entity(num)[0], df_grp['kettonum']))
            df_tmp['score'] = list(map(lambda row: dic_p["01"][row].get_good_entity(num)[1], df_grp['kettonum']))
            
            # レース番号などを追加
            for idx, col in enumerate(grp_col):
                df_tmp[col] = _[idx]
                
            # 他のレースとのデータフレームと結合する
            df_output = pd.concat([
                df_output,
                df_tmp
            ], ignore_index=True)


    return df_output