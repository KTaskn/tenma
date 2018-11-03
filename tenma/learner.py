# coding:utf-8
from tenma import dataload, abilitymodel, comparemodel

if __name__ == "__main__":
    df = dataload.load()
    abilitymodel.learn(df)
    comparemodel.learn(df)