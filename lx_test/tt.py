import gym
import highway_env
import time
import pprint
import matplotlib.pyplot as plt

env = gym.make("myenv-r1-v0")
# env = gym.make("u-turn-v0")
env.reset()
target_lane_index=("a", "b", 0)
target_speed = 10
env.vehicle.target_lane_index = target_lane_index
env.vehicle.targe_speed = target_speed
target_lane_x = []
target_lane_y = []
v_his_x = []
v_his_y = []

for _ in range(49):
    # action = env.action_type.actions_indexes["SLOWER"]
    action = 2
    # print(env.vehicle.target_lane_position())
    target_lane_x.append(env.vehicle.target_lane_position()[0])
    target_lane_y.append(env.vehicle.target_lane_position()[1])
    v_his_x.append(env.vehicle.position[0])
    v_his_y.append(env.vehicle.position[1])
    env.step(action)
    env.render()
    time.sleep(1)

# plt.imshow(env.render(mode='rgb_array'))
# plt.show()

# env.render()
# time.sleep(10)
env.close()
plt.plot(target_lane_x, target_lane_y)
plt.plot(v_his_x, v_his_y)
plt.show()