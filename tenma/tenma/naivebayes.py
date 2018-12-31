# coding:utf-8
import numpy as np
from tenma import dataload

# P(A)
def P_A(df, col_A, A):
    if(col_A not in df.columns):
        raise KeyError("%sはデータフレームに存在しない列です" % col_A)

    if(len(df.index) == 0):
        raise RuntimeError("データフレームにデータが存在しません")

    numerator = (df[col_A] == A).astype(int).sum()

    if numerator <= 0 :
        return -50.0
    else:
        return np.log(numerator / len(df[col_A].index))

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
        return -50.0
    else:
        return np.log(numerator / mask.astype(int).sum())

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

    result = []
    for idx, row in df[l_col].iterrows():
        a = NaiveBayes(
            df,
            np.log(1 / 18),
            l_col,
            row.values,
            "kakuteijyuni",
            "01"
        )
        print(a)
        result.append(a)

    return result
