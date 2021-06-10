import gym
import highway_env
import time
import pprint


env = gym.make("myenv-r1-v0")
env.reset()
# for _ in range(1000):
#     action = env.action_type.actions_indexes["SLOWER"]
#     env.step(action)
#     env.render()

# plt.imshow(env.render(mode='rgb_array'))
# plt.show()

env.render()
time.sleep(10)
env.close()
