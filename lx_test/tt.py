import gym
import highway_env
import time
import pprint


env = gym.make("myenv-r1-v0")
# env = gym.make("u-turn-v0")
env.reset()

for _ in range(50):
    # action = env.action_type.actions_indexes["SLOWER"]
    action = 2
    env.step(action)
    env.render()
    time.sleep(1)

# plt.imshow(env.render(mode='rgb_array'))
# plt.show()

# env.render()
# time.sleep(10)
env.close()
