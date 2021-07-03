import torch
import torch.nn as nn
import numpy as np
from highway_env.utils import lmap
import gym
from MPC.casadi_opti import get_first_action
import time


def v_lmap(s, road_r, max_v):
    s = s.reshape(4, -1)
    x = s[0, :]
    y = s[1, :]
    theta = s[2, :]
    v = s[3, :]
    for i in range(50):
        x[i] = lmap(x[i], [-1, 1], [-(road_r + 4), (road_r + 4)])
        y[i] = lmap(y[i], [-1, 1], [0, 2 * (road_r + 4)])
        theta[i] = lmap(theta[i], [-1, 1], [-2 * np.pi, 2 * np.pi])
        v[i] = lmap(v[i], [-1, 1], [-max_v, max_v])
    return x, y, theta, v


def a_lmap(actions):
    max_omega = np.pi / 3
    max_a = 5
    a1 = torch.eye(50) * max_omega
    a2 = torch.eye(50) * max_a
    a3 = torch.zeros((50, 50))
    a4 = torch.zeros((50, 50))
    a13 = torch.hstack((a1, a3))
    a42 = torch.hstack((a4, a2))
    ww = torch.vstack((a13, a42))
    linear_transfor = nn.Linear(100, 100)
    linear_transfor.weight = torch.nn.Parameter(torch.Tensor(ww))
    actions = linear_transfor(actions)
    return actions

# 加载
clpp = torch.load("clpp_model.pkl")
clpp.eval = True
clde = torch.load("clde_model.pkl")
em = np.load("embedding.npy")
em = torch.Tensor(em)
# 初始化环境
max_v = 30
env = gym.make("myenv-r1-v0")
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
env.reset()
p = env.vehicle.position
i_h = env.vehicle.heading
i_s = env.vehicle.speed
temp = np.array([p[0], p[1], i_h, i_s])
road_r = env.config["radius"]
temp_n = [lmap(p[0], [-(road_r + 4), (road_r + 4)], [-1, 1]), lmap(p[1], [0, 2 * (road_r + 4)], [-1, 1]),
          lmap(i_h, [-2 * np.pi, 2 * np.pi], [-1, 1]), lmap(i_s, [-max_v, max_v], [-1, 1])]
x_road, y_road, x_road_n, y_road_n = env.vehicle.target_lane_position(p, road_r)
s0 = temp_n
s0 = np.array(s0)
s0 = torch.Tensor(s0)
s0 = s0.unsqueeze(0)
ss = x_road_n + y_road_n
ss = np.array(ss)
ss = torch.Tensor(ss)
ss = ss.unsqueeze(0)
z = torch.LongTensor([4])
hidden = em[z.item()]
hidden = hidden.unsqueeze(0)

s_hat, _, _, _, _ = clpp(s0, ss, z, em)
u_hat = clde(s0, ss, hidden)

s_hat = s_hat.squeeze(0)
s0 = s0.squeeze(0)

s_hat = s_hat.detach().cpu().numpy()


x, y, theta, v = v_lmap(s_hat, road_r, 30)
x_f = np.array([x[-1], y[-1], theta[-1], v[-1]])


actions = a_lmap(u_hat)
actions = actions.squeeze(0)
actions = actions.detach().cpu().numpy()

env.reset()
N = 50

s_hat = np.concatenate((x, y, theta, v))

# first_a, u_e, x_e = get_first_action(s0, actions, s_hat, z.item(), x_f)
u0 = np.zeros((N, 2))
next_states = np.zeros((N + 1, 4))
u_e, x_e = get_first_action(temp, x_f, u0, next_states)

for i in range(N):

    action = u_e[0, :]
    # action = actions.reshape(-1, 50)[:, i]

    obs, reward, terminal, info = env.step(action)

    p = env.vehicle.position
    i_h = env.vehicle.heading
    i_s = env.vehicle.speed
    temp = np.array([p[0], p[1], i_h, i_s])
    u_e = np.concatenate((u_e[1:], u_e[-1:]))
    x_e = np.concatenate((x_e[1:], x_e[-1:]))
    u_e, x_e = get_first_action(temp, x_f, u_e, x_e)

    env.render()
    time.sleep(0.1)
print("我要到的地方 {}".format(x_f))
print("实际到的地方 [{} {} {} {}]".format(env.vehicle.position[0], env.vehicle.position[1],
                                    env.vehicle.heading, env.vehicle.speed))
env.close()
