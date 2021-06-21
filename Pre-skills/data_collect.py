import gym
import highway_env
import time
import pprint
import matplotlib.pyplot as plt
import numpy as np

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

    # 选择不同的初始状态
    lanes_count = env.config["lanes_count"]
    lane_id = np.random.choice(np.arange(lanes_count))
    print("v lane id is {}".format(lane_id))

    if lane_id == 0:
        target_lane_id = np.random.choice([0, 1])
    elif lane_id == lanes_count - 1:
        target_lane_id = np.random.choice([lanes_count - 1, lanes_count - 2])
    else:
        target_lane_id = np.random.choice([lane_id - 1, lane_id, lane_id + 1])

    print("target lane id is {}".format(target_lane_id))

    lon_operation = np.random.choice([0, 1, 2])  # 1保持 0减速 2加速
    print("1保持 0减速 2加速 - is {}".format(lon_operation))

    v_lane_id = ("a", "b", lane_id)
    target_lane_id2 = ("a", "b", target_lane_id)
    v_target_s = (lon_operation - 1) * 5 + env.vehicle.speed
    v_target_s = np.clip(0, 30, v_target_s)

    positon_x = np.random.choice(np.arange(0, env.road.network.get_lane(v_lane_id).length, 10))
    positon_y = np.random.choice(np.arange(-2, 2, 3))
    heading = np.random.choice(
        env.road.network.get_lane(v_lane_id).heading_at(positon_x) + np.arange(-np.pi / 12, np.pi / 12, 10))
    speed = np.random.choice(np.arange(0, 25, 5))

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
        # time.sleep(0.5)
    env.close()
    # temp = temp.extend(action_his_omega)
    # temp = temp.extend(action_his_accel)
    tt = temp + x_road + y_road + action_his_omega + action_his_accel
    pp = x_his + y_his + h_his + s_his
    lane_change = target_lane_id - lane_id

    label = labels_index[lane_change + 1, lon_operation]

    return lane_change, lon_operation, tt, pp, label


def positive_selector(lateral_operation, lon_operation):
    """
    选正的样本
    :return:
    """
    """ 正样本 """
    # 选择不同的初始状态
    positive_env = envs[np.random.choice(np.arange(3))]
    # print("env is {}".format(positive_env))
    env = gym.make(positive_env)
    lanes_count = env.config["lanes_count"]

    if lateral_operation == 1:
        lane_id = np.random.choice([0, 1])
        positive_lane_id = lane_id + 1
    elif lateral_operation == -1:
        lane_id = np.random.choice([2, 1])
        positive_lane_id = lane_id - 1
    else:
        lane_id = np.random.choice([2, 1, 0])
        positive_lane_id = lane_id
    # print("v lane id is {}".format(lane_id))
    v_lane_id = ("a", "b", lane_id)
    # print("target lane id is {}".format(positive_lane_id))
    target_lane_id = ("a", "b", positive_lane_id)
    v_target_s = (lon_operation - 1) * 5 + env.vehicle.speed
    # print("inital speed is {}, target speed is {}".format(env.vehicle.speed, v_target_s))

    positive_positon_x = np.random.choice(np.arange(0, env.road.network.get_lane(v_lane_id).length, 10))
    positive_positon_y = np.random.choice(np.arange(-2, 2, 3))
    positive_heading = np.random.choice(
        env.road.network.get_lane(v_lane_id).heading_at(positive_positon_x) + np.arange(-np.pi / 6, np.pi / 6, 10))
    positive_speed = np.random.choice(np.arange(0, 25, 5))

    env.config["v_lane_id"] = v_lane_id
    env.config["v_target_id"] = target_lane_id
    env.config["v_x"] = positive_positon_x
    env.config["v_y"] = positive_positon_y
    env.config["v_h"] = positive_heading
    env.config["v_s"] = positive_speed
    env.config["v_target_s"] = v_target_s

    env.reset()
    p = env.vehicle.position
    i_h = env.vehicle.heading
    i_s = env.vehicle.speed
    temp = [p[0], p[1], i_h, i_s]
    x_road, y_road = env.vehicle.target_lane_position(p)
    action = 1
    action_his_omega = []
    action_his_accel = []
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
        # time.sleep(0.5)
    tt = temp + x_road + y_road + action_his_omega + action_his_accel
    pp = x_his + y_his + h_his + s_his
    env.close()
    return tt, pp


def negative_selector(lateral_operation, lon_operation):
    """
    选8个负的样本
    :return:
    """

    """ 负样本 """
    i = 1
    tt_8 = []
    pp_8 = []
    for lateral_negative in [-1, 0, 1]:
        for lon_neagetive in [0, 1, 2]:
            # print("第 {} 次开始： ".format(i))
            # print(("later is {}, lon is {}".format(lateral_negative, lon_neagetive)))
            i += 1
            if lateral_negative == lateral_operation and lon_operation == lon_neagetive:
                print("冲突样本")
                continue

            negative_env = envs[np.random.choice(np.arange(3))]
            # print("env is {}".format(negative_env))
            env = gym.make(negative_env)
            lanes_count = env.config["lanes_count"]
            # lane_id = np.random.choice(np.arange(lanes_count))

            if lateral_negative == 1:
                lane_id = np.random.choice([0, 1])
                negative_lane_id = lane_id + 1
            elif lateral_negative == -1:
                lane_id = np.random.choice([2, 1])
                negative_lane_id = lane_id - 1
            else:
                negative_lane_id = np.random.choice([2, 1, 0])
                lane_id = negative_lane_id

            # print("v lane id is {}".format(lane_id))
            v_lane_id = ("a", "b", lane_id)
            # print("target lane id is {}".format(negative_lane_id))
            target_lane_id = ("a", "b", negative_lane_id)
            v_target_s = (lon_neagetive - 1) * 5 + env.vehicle.speed
            # print("inital speed is {}, target speed is {}".format(env.vehicle.speed, v_target_s))

            negative_positon_x = np.random.choice(np.arange(0, env.road.network.get_lane(v_lane_id).length, 10))
            negative_positon_y = np.random.choice(np.arange(-2, 1.9, 3))
            negative_heading = np.random.choice(
                env.road.network.get_lane(v_lane_id).heading_at(negative_positon_x) + np.arange(-np.pi / 6, np.pi / 6,
                                                                                                10))
            negative_speed = np.random.choice(np.arange(0, 25, 5))

            env.config["v_lane_id"] = v_lane_id
            env.config["v_target_id"] = target_lane_id
            env.config["v_x"] = negative_positon_x
            env.config["v_y"] = negative_positon_y
            env.config["v_h"] = negative_heading
            env.config["v_s"] = negative_speed
            env.config["v_target_s"] = v_target_s

            env.reset()
            p = env.vehicle.position
            i_h = env.vehicle.heading
            i_s = env.vehicle.speed
            temp = [p[0], p[1], i_h, i_s]
            x_road, y_road = env.vehicle.target_lane_position(p)
            action = 1
            action_his_omega = []
            action_his_accel = []
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
                # time.sleep(0.5)
            tt = temp + x_road + y_road + action_his_omega + action_his_accel
            pp = x_his + y_his + h_his + s_his
            tt_8.append(tt)
            pp_8.append(pp)
            env.close()
    return tt_8, pp_8


# print("------锚--------")
# inital_state, initial_road, lane_change, lon_operation = anchor_selector()
# print("------正--------")
# positive_selector(lane_change, lon_operation)
# print("------负--------")
# negative_selector(lane_change, lon_operation)


lane_change, lon_operation, tt, pp, label = anchor_selector()
# tt_8, pp_8 = negative_selector(lane_change, lon_operation)
