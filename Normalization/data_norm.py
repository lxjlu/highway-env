import casadi as ca
import casadi.tools as ca_tools
import gym
import highway_env
import numpy as np
import torch
from nets import CLPP
from MPC.casadi_mul_shooting import get_first_action
import time
import matplotlib.pyplot as plt
from highway_env.utils import lmap

T = 0.1  # sampling time [s]
N = 50  # prediction horizon
max_v = 30
max_omega = np.pi/6
max_a = 5  # [m/s]
accel_max = 1
omega_max = np.pi/6
clpp = torch.load("clpp_model.pkl")
clpp.eval = True
clde = torch.load("clde_model.pkl")
em = np.load("embedding.npy")
em = torch.Tensor(em)

z = torch.LongTensor([4])
hidden = em[z.item()]
print(hidden)
hidden = hidden.unsqueeze(0)

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
# env.config["action"]['type'] = 'ContinuousAction',
env.reset()

def get_hat(env, z):
    p = env.vehicle.position
    i_h = env.vehicle.heading
    i_s = env.vehicle.speed
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

    label = z
    s_hat, _, _, _, _ = clpp(s0, ss, label, em)
    print("ok")
    u_hat = clde(s0, ss, hidden)
    s_hat = s_hat.squeeze(0)
    u_hat = u_hat.squeeze(0)
    s0 = s0.squeeze(0)
    s_hat = s_hat.detach().cpu().numpy()
    u_hat = u_hat.detach().cpu().numpy()
    s0 = s0.detach().cpu().numpy()

    x_f = s_hat.reshape(4, 50)[:, -1]
    print(x_f)
    # print("原本的终端是 ", x_f)
    # x_f_local_x, x_f_local_y = env.road.network.get_lane(v_lane_id).local_coordinates(np.array([x_f[0], x_f[1]]))
    # x_f_local_y = np.clip(x_f_local_y, -4, 4)
    # x_f_xy = env.road.network.get_lane(v_lane_id).position(x_f_local_x, x_f_local_y)
    # x_f[0] = x_f_xy[0]
    # x_f[1] = x_f_xy[1]
    # x_f[2] = env.road.network.get_lane(v_lane_id).heading_at(x_f_xy[0])
    # print("现在的终端是 ", x_f)
    return s0, ss, s_hat, u_hat, x_f

s0, ss, s_hat, u_hat, x_f = get_hat(env, z)

action, u_e, x_e = get_first_action(s0, u_hat, s_hat, z.item(), x_f)

# plt.figure(1)
# plt.plot(x_e[:, 0], x_e[:, 1])
# plt.figure(2)
# dd = s_hat.reshape(4, 50)
# plt.plot(dd[0,:], dd[1,:])
# plt.show()

for i in range(N):
    action = u_e[i, :]

    obs, reward, terminal, info = env.step(action)

    env.render()
    time.sleep(0.1)
env.close()







