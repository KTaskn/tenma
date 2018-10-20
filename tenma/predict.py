# coding:utf-8
import json
import pandas as pd
from tenma import dataload, model

if __name__ == "__main__":
    df = dataload.load()
    FILE_PATH = "model/ability_params.json"
    params = json.loads(FILE_PATH)
    D = 4
    X = model.x_for_ability(df, D)
    model.ability(X, D, params)