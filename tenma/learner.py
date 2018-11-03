# coding:utf-8
import numpy as np
import pandas as pd
from tenma import dataload, abilitymodel, comparemodel
from tenma.rulemodel import Rule

if __name__ == "__main__":
    df = dataload.load()
    abilitymodel.learn(df)
    comparemodel.learn(df)