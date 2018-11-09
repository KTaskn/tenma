# coding:utf-8
from sklearn import metrics
from tenma import dataload, abilitymodel, comparemodel

if __name__ == "__main__":
    df = dataload.load()
    abilitymodel.learn(df)
    comparemodel.learn(df)

    df['predict'] = abilitymodel.predict(df)
    df = abilitymodel.get_evaluate_df(df)
    y = (df['kakuteijyuni'] == '01').astype(int)
    pred = df['predict'].values
    fpr, tpr, thresholds = metrics.roc_curve(y, pred)
    print("am model auc:%f" % metrics.auc(fpr, tpr))

    df['predict_comp'] = comparemodel.predict(df)
    df = abilitymodel.get_evaluate_df(df)
    y = (df['kakuteijyuni'] == '01').astype(int)
    pred = df['predict_comp'].values
    fpr, tpr, thresholds = metrics.roc_curve(y, pred)
    print("cm model auc:%f" % metrics.auc(fpr, tpr))