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

    for i in range(num_data):
        print("第 {} 个样本".format(i+1))
        lane_change, lon_operation, a = anchor_selector()
        p = positive_selector(lane_change, lon_operation)
        tt_8 = negative_selector(lane_change, lon_operation)
        anchors.append(a)
        positives.append(p)
        negatives.append(tt_8)

    return anchors, positives, negatives


anchors, positives, negatives = generator(10000)
print("到这了")
a = np.array(anchors)
p = np.array(positives)
n = np.array(negatives)

np.save("anchors.npy", a)
np.save("positives.npy", p)
np.save("negatives.npy", n)