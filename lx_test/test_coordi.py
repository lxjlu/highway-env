import gym
import highway_env
import time
import pprint
import matplotlib.pyplot as plt
import numpy as np

env = gym.make("myenv-c1-v0")
v_lane_id = ("a", "b", 0)
target_lane_id = ("a", "b", 2)
x = 100
y = -2
h = 0
s = 10
v_target_s = 15

env.config["v_lane_id"] = v_lane_id
env.config["v_target_id"] = target_lane_id
env.config["v_x"] = x
env.config["v_y"] = y
env.config["v_h"] = h
env.config["v_s"] = s
env.config["v_target_s"] = v_target_s

env.reset()
x_his = []
y_his = []
x_his.append(env.vehicle.position[0])
y_his.append(env.vehicle.position[1])
action = 1
for _ in range(50):
    print("x is {}, y is {}".format(env.vehicle.position[0], env.vehicle.position[1]))
    if env.vehicle.on_road is False:
        print("出去了")
        break
    env.step(action)
    x_his.append(env.vehicle.position[0])
    y_his.append(env.vehicle.position[1])
    env.render()
    # time.sleep(0.5)
road_x, road_y = env.vehicle.target_lane_position([100, -2])

plt.plot(x_his, y_his)
plt.plot(road_x, road_y)
plt.show()
env.close()