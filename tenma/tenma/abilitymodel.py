# coding:utf-8
import os
from pprint import pprint
import numpy as np

def normal(mu, sigma):
    if 'env' in os.environ.keys() and os.environ['env'] == "test":
        return mu
    return np.random.normal(mu, sigma)

def inv_logit(x):
    return 1.0 / (1.0 + np.exp(-x))

P_MU = "p_mu"
P_SIGMA = "p_sigma"
def calc_order(order, params):
    return normal(params[P_MU], params[P_SIGMA]) * (1.0 / order)

G_A_MU = "g_a_mu"
G_A_SIGMA = "g_a_sigma"
G_B_MU = "g_b_mu"
G_B_SIGMA = "g_b_sigma"
def calc_grade(bf_grade, now_grade, params):
    return inv_logit(
        normal(params[G_A_MU], params[G_A_SIGMA])
        * (now_grade - bf_grade)
        + normal(params[G_B_MU], params[G_B_SIGMA])
    )

def calc_bfrace(order, bf_grade, now_grade, params):
    return calc_order(order, params) * calc_grade(bf_grade, now_grade, params)


BIAS_MU = "bias_mu"
BIAS_SIGMA = "bias_sigma"
def predict(l_order, l_bf_grade, now_grade, all_params):
    ret = 0
    for num in range(4):
        params = {
            P_MU : all_params[P_MU][num],
            P_SIGMA : all_params[P_SIGMA][num],
            G_A_MU : all_params[G_A_MU],
            G_A_SIGMA : all_params[G_A_SIGMA],
            G_B_MU : all_params[G_B_MU],
            G_B_SIGMA : all_params[G_B_SIGMA]
        }
        params[P_MU] = all_params[P_MU][num]
        params[P_SIGMA] = all_params[P_SIGMA][num]
        ret += calc_bfrace(
            l_order[num],
            l_bf_grade[num],
            now_grade,
            params
        )
    ret += normal(
        all_params[BIAS_MU],
        all_params[BIAS_SIGMA]
    )

    return inv_logit(ret)