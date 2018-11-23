# coding:utf-8
# networkx の import
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx

def make_data():
    df = pd.read_csv('2015_2017_race.csv').pipe(
        lambda df: df[
            (df['kakuteijyuni'] > 0)
            & (df['year'] <= 2018)
            ])
    grouped = df.groupby(['year', 'monthday', 'jyocd', 'racenum'])
    with open("list.txt", "w") as f:
        for name, group in grouped:
            print(name)
            for idx_0, row_0 in group[['bamei', 'kakuteijyuni']].iterrows():
                for idx_1, row_1 in group[['bamei', 'kakuteijyuni']].iterrows():
                    if row_0['kakuteijyuni'] <= 3 and row_0['kakuteijyuni'] < row_1['kakuteijyuni']:
                        f.write("%s,%s\n" % (row_1['bamei'], row_0['bamei']))

# make_data()
# グラフを構築
df = pd.read_csv('2015_2017_race.csv', dtype=str)
G = nx.read_edgelist(
    'list.txt',
    delimiter=',',
    nodetype=str,
    create_using=nx.DiGraph())

# ノード数とエッジ数を出力
print(nx.number_of_nodes(G))
print(nx.number_of_edges(G))

# pagerank の計算
pr = nx.pagerank(G)
print(pd.Series(pr).reset_index())
tmp = pd.Series(pr).reset_index()
tmp.columns = ['bamei', 'score']
print(pd.merge(
    df.pipe(lambda df: df[
        (df['year'] == '2018')
        & (df['monthday'] == '1118')
        & (df['jyocd'] == '08')
        & (df['racenum'] == '11')
        ]),
    tmp,
    on = ['bamei']
).sort_values('score'))
# print(pd.Series(pr).sort_values())
# # 可視化
# pos = nx.spring_layout(G)
# plt.figure(figsize=(6, 6))
# nx.draw_networkx_edges(G, pos)
# nx.draw_networkx_nodes(G, pos, node_color=list(pr.values()), cmap=plt.cm.Reds)
# plt.axis('off')
# plt.savefig("network.png")