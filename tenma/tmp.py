i = 0
a = 0
for idx, grp in df.groupby(['year', 'monthday', 'jyocd', 'racenum', 'kettonum']):
    if np.percentile(grp['score'], 75) < grp.iloc[0]['odds'] and grp.iloc[0]['odds'] < 32.0 :
        print(np.percentile(grp['score'], 75), grp.iloc[0]['odds'])
        i -= 1
        a += 1
        if grp.iloc[0]['kakuteijyuni'] == "01":
            i += grp.iloc[0]['odds']