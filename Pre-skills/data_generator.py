import gym
import highway_env
import time
import pprint
import matplotlib.pyplot as plt
import numpy as np
from data_collect import *
"""
(x, x+, x-)
"""


def generator(num_data):

    anchors = []
    positives = []
    negatives = []

    a_h = []
    p_h = []
    n_h = []

    for i in range(num_data):
        print("第 {} 个样本".format(i+1))
        lane_change, lon_operation, a, a_pp = anchor_selector()
        p, p_pp = positive_selector(lane_change, lon_operation)
        tt_8, n_pp_8 = negative_selector(lane_change, lon_operation)
        anchors.append(a)
        positives.append(p)
        negatives.append(tt_8)

        a_h.append(a_pp)
        p_h.append(p_pp)
        n_h.append(n_pp_8)

    return anchors, positives, negatives, a_h, p_h, n_h


anchors, positives, negatives, a_h, p_h, n_h = generator(10000)
print("到这了")
a = np.array(anchors)
p = np.array(positives)
n = np.array(negatives)

a_his = np.array(a_h)
p_his = np.array(p_h)
n_his = np.array(n_h)

np.save("anchors.npy", a)
np.save("positives.npy", p)
np.save("negatives.npy", n)

np.save("a_his.npy", a_his)
np.save("p_his.npy", p_his)
np.save("n_his.npy", n_his)