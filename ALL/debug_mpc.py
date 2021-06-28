from nets import ReplayBuffer
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import gym
import highway_env

load_pid = True
bf = ReplayBuffer(load_pid)
z_his, u_his, s0_his, X_his, ss_his, cl_his = bf.sample(500)

# z, u, s0, X, ss, cl = z_his[0], u_his[0], s0_his[0], X_his[0], ss_his[0], cl_his[0]
# X = X.reshape(4, 50)
# u = u.reshape(2, 50)
# plt.figure(1)
# plt.plot(X[0, :], X[1, :])
# plt.show()
#
# clpp = torch.load("pp_model.pkl")
# clpp.eval = True
# clde = torch.load("de_model.pkl")
# em = np.load("embedding.npy")
# em = torch.Tensor(em)
#
# z_his, u_his, s0_his, X_his, ss_his, cl_his = torch.Tensor(z_his), torch.Tensor(u_his), torch.Tensor(s0_his), torch.Tensor(X_his), torch.Tensor(ss_his), torch.Tensor(cl_his)
# s_hat, _, _, _, _ = clpp(s0_his, ss_his, z_his, em)

env = gym.make("myenv-r1-v0")
labels_index = np.arange(9).reshape(3, 3)
N = 50
lanes_count = env.config["lanes_count"]
lane_id = np.random.choice(np.arange(lanes_count))
v_lane_id = ("a", "b", 1)
# positon_x = np.random.choice(np.arange(0, env.road.network.get_lane(v_lane_id).length, 5))
positon_x = 0
# positon_y = np.random.choice(np.arange(-2, 2.1, 0.5))
positon_y = 0
heading = env.road.network.get_lane(v_lane_id).heading_at(positon_x)
speed = 10
env.config["v_x"] = positon_x
env.config["v_y"] = positon_y
env.config["v_h"] = heading
env.config["v_s"] = speed
env.config["v_lane_id"] = v_lane_id
env.config["action"]["type"] = "ContinuousAction"
env.reset()
