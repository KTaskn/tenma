# coding:utf-8
import numpy as np
import pandas as pd
from tenma import dataload, abilitymodel, comparemodel
from tenma.rulemodel import Rule

if __name__ == "__main__":
    df = dataload.load()
    # abilitymodel.learn(df)
    # comparemodel.learn(df)

    df['odds'] = df['odds'].astype(np.int32) / 10.0
    df['reward'] = ((df['kakuteijyuni'] == "01") * df['odds']) + ((df['kakuteijyuni'] != "01") * -1.0)
    df['predict'] = abilitymodel.predict(df)

    rule = Rule()
    for i in range(50):
        for x, y, reward in df.sample(2500)[['odds', 'predict', 'reward']].values:
            action = rule.get_nearly([x, y])
            if action:
                rule.set_data(reward, action, [x, y])
            else:
                rule.set_data(0, action, [x, y])
                
        rule.set_next()
        rule.init_data()

    pd.DataFrame({
        "odds": np.array(rule.l_point)[:, 0],
        "predict": np.array(rule.l_point)[:, 1],
        "reward": rule.l_reward,
        "action": rule.l_action,
    }).to_csv('rule.csv')