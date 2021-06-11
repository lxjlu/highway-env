import gym
import highway_env
import time
import pprint
import matplotlib.pyplot as plt

"""
# 直线
v_lane_id = ("a", "b", 1)
v_target_id = ("a", "b", 0)
x = 100.0
y = 0.0
h = 0.0
s = 10.0
target_s = 20.0
env = gym.make("myenv-c1-v0")
env.config["v_lane_id"] = v_lane_id
env.config["v_target_id"] = v_target_id
env.config["v_x"] = x
env.config["v_y"] = y
env.config["v_h"] = h
env.config["v_s"] = s
env.config["v_target_s"] = target_s
env.config["KP_HEADING"] = 1
env.config["KP_LATERAL"] = 0.8
env.reset()
env.vehicle.TAU_PURSUIT = env.config["TAU_PURSUIT"]
env.vehicle.KP_LATERAL = env.config["KP_LATERAL"]
env.vehicle.KP_A = env.config["KP_A"]
env.vehicle.KP_HEADING = env.config["KP_HEADING"]

for _ in range(50):
    action = 1
    env.step(action)
    env.render()
    time.sleep(0.5)
env.close()
"""
v_lane_id = ("a", "b", 1)
v_target_id = ("a", "b", 0)
x = 100.0
y = 0.0
h = 0.0
s = 10.0
target_s = 20.0
env = gym.make("myenv-c3-v0")
env.config["real_time_rendering"] = True
env.config["scaling"] = 6
env.config["v_lane_id"] = v_lane_id
env.config["v_target_id"] = v_target_id
env.config["v_x"] = x
env.config["v_y"] = y
env.config["v_h"] = h
env.config["v_s"] = s
env.config["v_target_s"] = target_s
env.config["KP_HEADING"] = 5
env.config["KP_LATERAL"] = 5
env.config["TAU_PURSUIT"] = 0.1
env.reset()
env.vehicle.TAU_PURSUIT = env.config["TAU_PURSUIT"]
env.vehicle.KP_LATERAL = env.config["KP_LATERAL"]
env.vehicle.KP_A = env.config["KP_A"]
env.vehicle.KP_HEADING = env.config["KP_HEADING"]

for _ in range(50):
    action = 1
    env.step(action)
    env.render()
    # time.sleep(0.5)
env.close()

