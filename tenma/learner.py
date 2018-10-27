from tenma import dataload, abilitymodel

if __name__ == "__main__":
    df = dataload.load()
    abilitymodel.learn(df)