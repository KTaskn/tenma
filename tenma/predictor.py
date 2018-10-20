# coding:utf-8
import json
import pandas as pd
from tenma import dataload, abilitymodel

if __name__ == "__main__":
    df = dataload.load()
    FILE_PATH = "model/ability_params.json"
    params = json.loads(FILE_PATH)
    D = 4
    X = abilitymodel.x_for_ability(df, D)
    abilitymodel.ability(X, D, params)