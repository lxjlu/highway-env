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

positon_x = np.random.choice(np.arange(0, env.road.network.get_lane(v_lane_id).length, 10))
# print("position x is {}".format(positon_x))
positon_y = np.random.choice(np.arange(-2, 2, 3))
# print("position y is {}".format(positon_y))
heading = np.random.choice(
    env.road.network.get_lane(v_lane_id).heading_at(positon_x) + np.arange(-np.pi / 12, np.pi / 12, 10))
speed = np.random.choice(np.arange(0, 25, 5))

# global_x, global_y = env.road.network.get_lane(v_lane_id).position(positon_x, positon_y)


# inital_state = [position, heading, speed]

env.config["v_lane_id"] = v_lane_id
env.config["v_target_id"] = target_lane_id2
env.config["v_x"] = positon_x
env.config["v_y"] = positon_y
env.config["v_h"] = heading
env.config["v_s"] = speed
env.config["v_target_s"] = v_target_s
env.reset()

p = env.vehicle.position
x_road, y_road = env.vehicle.target_lane_position(p)


x_his = []
y_his = []
# x_road = []
# y_road = []
# x_1, y_1 = env.vehicle.target_lane2_position()
# x_road.append(x_1)
# y_road.append(y_1)
# x_road, y_road = env.vehicle.target_lane_position([global_x, global_y])
x_his.append(env.vehicle.position[0])
y_his.append(env.vehicle.position[1])
action = 1
action_his_omega = []
action_his_accel = []

for _ in range(N):
    # print("x is {}, y is {}".format(env.vehicle.position[0], env.vehicle.position[1]))
    action_his_omega.append(env.vehicle.action["steering"])
    action_his_accel.append(env.vehicle.action["acceleration"])

    if env.vehicle.on_road is False:
        print("出去了")
        break
    env.step(action)
    x_his.append(env.vehicle.position[0])
    y_his.append(env.vehicle.position[1])
    x_1, y_1 = env.vehicle.target_lane2_position()
    # x_road.append(x_1)
    # y_road.append(y_1)
    env.render()
    # time.sleep(0.5)
env.close()

lane_change = target_lane_id - lane_id

plt.figure(1)
plt.plot(x_his, y_his)
plt.plot(x_road, y_road)
# plt.figure(2)
# plt.plot(np.arange(len(action_his_accel)), action_his_accel)
plt.show()

