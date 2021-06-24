import gym
import numpy as np
import time
import highway_env
import torch
from torch.utils.data import Dataset
# from scripts.utils import record_videos, show_videos

envs = {
    0: "myenv-c1-v0",  # 直线
    1: "myenv-c2-v0",  # 中度弯曲
    2: "myenv-c3-v0",  # 大量弯曲
}
N = 50

# 1保持 0减速 2加速 lon
# -1左 0保持 1右 latral
"""
    | 0  1  2
-1  |
 0  |
 1  |
"""

labels_index = np.arange(9).reshape(3, 3)


def anchor_selector():
    """
    选不同的卯
    :return:
    """
    # 选择不同的道路
    env_lucky = envs[np.random.choice(np.arange(3))]
    # print("env is {}".format(env_lucky))

    env = gym.make(env_lucky)
    # env = record_videos(env)

    # 选择不同的初始状态
    lanes_count = env.config["lanes_count"]
    lane_id = np.random.choice(np.arange(lanes_count))
    # print("v lane id is {}".format(lane_id))

    if lane_id == 0:
        target_lane_id = np.random.choice([0, 1])
    elif lane_id == lanes_count - 1:
        target_lane_id = np.random.choice([lanes_count - 1, lanes_count - 2])
    else:
        target_lane_id = np.random.choice([lane_id - 1, lane_id, lane_id + 1])

    # print("target lane id is {}".format(target_lane_id))

    lon_operation = np.random.choice([0, 1, 2])  # 1保持 0减速 2加速
    # print("1保持 0减速 2加速 - is {}".format(lon_operation))

    v_lane_id = ("a", "b", lane_id)
    target_lane_id2 = ("a", "b", target_lane_id)
    v_target_s = (lon_operation - 1) * 5 + env.vehicle.speed
    v_target_s = np.clip(0, 30, v_target_s)

    positon_x = np.random.choice(np.arange(0, env.road.network.get_lane(v_lane_id).length, 5))
    positon_y = np.random.choice(np.arange(-2, 2.1, 0.5))
    heading = np.random.choice(
        env.road.network.get_lane(v_lane_id).heading_at(positon_x) + np.arange(-np.pi / 12, np.pi / 12, 10))
    speed = np.random.choice(np.arange(0, 25, 2))

    position = env.road.network.get_lane(v_lane_id).position(positon_x, positon_y)
    inital_state = [position, heading, speed]

    env.config["v_lane_id"] = v_lane_id
    env.config["v_target_id"] = target_lane_id2
    env.config["v_x"] = positon_x
    env.config["v_y"] = positon_y
    env.config["v_h"] = heading
    env.config["v_s"] = speed
    env.config["v_target_s"] = v_target_s

    env.reset()
    p = env.vehicle.position
    i_h = env.vehicle.heading
    i_s = env.vehicle.speed
    temp = [p[0], p[1], i_h, i_s]
    x_road, y_road = env.vehicle.target_lane_position(p)
    # temp = temp.extend(x_road)
    # temp = temp.extend(y_road)
    action_his_omega = []
    action_his_accel = []
    action = 1
    x_his, y_his, h_his, s_his = [], [], [], []
    for _ in range(N):
        if env.vehicle.on_road is False:
            print("出去了")
            break

        env.step(action)
        action_his_omega.append(env.vehicle.action["steering"])
        action_his_accel.append(env.vehicle.action["acceleration"])
        x_his.append(env.vehicle.position[0])
        y_his.append(env.vehicle.position[1])
        h_his.append(env.vehicle.heading)
        s_his.append(env.vehicle.speed)
        # env.render()
        # time.sleep(0.1)
    env.close()
    # temp = temp.extend(action_his_omega)
    # temp = temp.extend(action_his_accel)
    tt = temp + x_road + y_road + action_his_omega + action_his_accel
    # pp = np.vstack((
    #     np.array(x_his),
    #     np.array(y_his),
    #     np.array(h_his),
    #     np.array(s_his),
    # ))
    pp = x_his + y_his + h_his + s_his
    lane_change = target_lane_id - lane_id

    label = labels_index[lane_change + 1, lon_operation]
    # s0 = np.array(temp)
    s0 = temp
    # ss = np.vstack((
    #     np.array(x_road),
    #     np.array(y_road),
    # ))
    ss = x_road + y_road
    # aa = np.vstack((
    #     np.array(action_his_omega),
    #     np.array(action_his_accel)
    # ))
    aa = action_his_omega + action_his_accel
    return s0, ss, aa, tt, pp, label


def generator(num_data):
    datas = []
    labels = []

    s0_b = []
    ss_b = []
    aa_b = []
    pp_b = []



    for i in range(num_data):
        print("第 {} 个样本".format(i + 1))

        s0, ss, aa, tt, pp, label = anchor_selector()

        datas.append(tt)
        labels.append(label)

        s0_b.append(s0)
        ss_b.append(ss)
        aa_b.append(aa)
        pp_b.append(pp)

    d = np.array(datas)
    l = np.array(labels)

    s0_b = np.array(s0_b)
    ss_b = np.array(ss_b)
    aa_b = np.array(aa_b)
    pp_b = np.array(pp_b)

    np.save("data.npy", d)
    np.save("label.npy", l)

    np.save("s0_b.npy", s0_b)
    np.save("ss_b.npy", ss_b)
    np.save("aa_b.npy", aa_b)
    np.save("pp_b.npy", pp_b)
    return d, l, s0_b, ss_b, aa_b, pp_b


class CLData(Dataset):
    d = np.load("data.npy")
    l = np.load("label.npy")

    s0_b = np.load("s0_b.npy")
    ss_b = np.load("ss_b.npy")
    aa_b = np.load("aa_b.npy")
    pp_b = np.load("pp_b.npy")

    def __init__(self):
        self.data = torch.Tensor(self.d)
        # self.label = self.l.astype(np.int64)
        self.label = torch.Tensor(self.l)

        self.s0_b = torch.Tensor(self.s0_b)
        self.ss_b = torch.Tensor(self.ss_b)
        self.aa_b = torch.Tensor(self.aa_b)
        self.pp_b = torch.Tensor(self.pp_b)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx], self.s0_b[idx], self.ss_b[idx], \
    self.aa_b[idx], self.pp_b[idx]


# d, l, s0_b, ss_b, aa_b, pp_b = generator(100)
dataset = CLData()
d, l, _, _, _, _ = dataset[:100]
# s0, ss, aa, tt, pp, label = anchor_selector()
