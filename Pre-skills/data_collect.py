import gym
import highway_env
import time
import pprint
import matplotlib.pyplot as plt
import numpy as np

envs = {
    0: "myenv-c1-v0",  # 直线
    1: "myenv-c2-v0",  # 中度弯曲
    3: "myenv-c3-v0",  # 大量弯曲
}


def test():
    """
    1. 选各种形状的路
    2. 选车辆的初始位置
    3. 选卯
    4. 选正样本
    5. 选8个负样本
    6. 设置损失函数
    7. 优化参数
    8. 梯度算法
    :return:
    """


def initial_state():
    """
    选不同的初始状态
    :return:
    """
    """ 卯 """






    # 选择不同的技巧
    """
    9 skills
         左换道   保持  右换道
    减速   0      1      2
    保持   3      4      5
    加速   6      7      8
    """
    skill = np.random.choice(np.arange(9))
    # return positon_x, positon_y, heading, speed, lateral_operation, lon_operation
    pass


def anchor_selector():
    """
    选不同的卯
    :return:
    """
    # 选择不同的道路
    env_lucky = envs[np.random.choice(np.arange(3))]
    print("env is {}".format(env_lucky))

    env = gym.make(env_lucky)

    # 选择不同的初始状态
    lanes_count = env.config["lanes_count"]
    lane_id = np.random.choice(np.arange(lanes_count))
    print("v lane id is {}".format(lane_id))

    if lane_id == 0:
        target_lane_id = np.random.choice([0, 1])
    elif lane_id == lanes_count-1:
        target_lane_id = np.random.choice([lanes_count-1, lanes_count-2])
    else:
        target_lane_id = np.random.choice([lane_id-1, lane_id, lane_id+1])

    print("target lane id is {}".format(target_lane_id))

    lon_operation = np.random.choice([0, 1, 2]) # 1保持 0减速 2加速
    print("1保持 0减速 2加速 - is {}".format(lon_operation))

    v_lane_id = ("a", "b", lane_id)
    target_lane_id = ("a", "b", target_lane_id)
    v_target_s = (lon_operation - 1) * 5 + env.vehicle.speed
    v_target_s = np.clip(0, 30, v_target_s)


    positon_x = np.random.choice(np.arange(0, env.road.network.get_lane(v_lane_id).length, 10))
    positon_y = np.random.choice(np.arange(-2, 2, 3))
    heading = np.random.choice(np.arange(-np.pi / 12, np.pi / 12, 10))
    speed = np.random.choice(np.arange(0, 25, 5))

    env.config["v_lane_id"] = v_lane_id
    env.config["v_target_id"] = target_lane_id
    env.config["v_x"] = positon_x
    env.config["v_y"] = positon_y
    env.config["v_h"] = heading
    env.config["v_s"] = speed
    env.config["v_target_s"] = v_target_s

    env.reset()

    return env








if __name__ == "__main__":
    a = {
        0: "fd",
        1: "fda",
        2: "fdafd"
    }

    print(a)
