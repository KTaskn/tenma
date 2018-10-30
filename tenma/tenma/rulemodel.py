# coding:utf-8
import numpy as np
import pandas as pd
from scipy.spatial import distance

eps = 0.1

class Rule:
    def __init__(self, file=""):
        if file:
            df = pd.read_csv(file)
            # 報酬
            self.l_reward = df.reward.tolist()
            # アクション
            self.l_action = df.action.tolist()
            # 座標
            self.l_point = df[['odds', 'predict']].values.tolist()
        else:
            # 報酬
            self.l_reward = []
            # アクション
            self.l_action = []
            # 座標
            self.l_point = []
        
        
        # 報酬
        self.l_reward_n = []
        # アクション
        self.l_action_n = []
        # 座標
        self.l_point_n = []
        
    def get_nearly(self, p):
        N = 5
        if self.l_point:
            dist_M = distance.cdist([p], self.l_point, metric='euclidean')
            state = pd.DataFrame({
                "d": dist_M[0],
                "action": self.l_action,
                "reward": self.l_reward
            }).sort_values("d").head(N)
            result = state.groupby('action')['reward'].sum().sort_values(ascending=False).index[0]
            if eps > np.random.random():
                if result == 1:
                    return 0
                else:
                    return 1
            else:
                return result
        else:
            if 0.5 > np.random.random():
                return 0
            else:
                return 1

    def get_nearly_predict(self, p):
        N = 5
        if self.l_point:
            dist_M = distance.cdist([p], self.l_point, metric='euclidean')
            state = pd.DataFrame({
                "d": dist_M[0],
                "action": self.l_action,
                "reward": self.l_reward
            }).sort_values("d").head(N)
            result = state.groupby('action')['reward'].sum().sort_values(ascending=False).index[0]
            return result
        else:
            if 0.5 > np.random.random():
                return 0
            else:
                return 1
        
    def set_data(self, r, a, p):
        self.l_reward_n.append(r)
        self.l_action_n.append(a)
        self.l_point_n.append(p)
        
        return True
    
    def set_next(self):
        self.l_reward = self.l_reward_n
        self.l_action = self.l_action_n
        self.l_point = self.l_point_n
    
    def init_data(self):
        # 報酬
        self.l_reward_n = []
        # アクション
        self.l_action_n = []
        # 座標
        self.l_point_n = []