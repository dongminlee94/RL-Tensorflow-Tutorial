"""
Dummy Q-learning (table)
exploit&exploration and discounted future reward 에 대한 코드입니다.

# 지금 보여드리는 코드는 E-greedy이고, RL_Lab_04_1.py에 있는 것은 add random noise입니다.

기본적인 코드들은 RL_Lab_03_1.py를 참고 바랍니다.
"""
import gym
import numpy as np
import matplotlib.pyplot as plt
from gym.envs.registration import register
import random as pr

register(
    id='FrozenLake-v3',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name': '4x4',
            'is_slippery': False}
)

env = gym.make('FrozenLake-v3')

# Initialize table with all zeros
Q = np.zeros([env.observation_space.n, env.action_space.n])
# Set learning parameters
dis = .99
num_episodes = 2000

# create lists to contain total rewards and steps per episode
rList = []
for i in range(num_episodes):
    # Reset environment and get first new observation
    state = env.reset()
    rAll = 0
    done = False

    e = 1. / ((i // 100) + 1)  # Python2 & 3, python2와 3이 다르기 때문에 //로 통일하겠습니다.

    # The Q-Table learning algorithm
    while not done:
        """ Choose an action by e greedy, 여기 부분부터 E-greedy 코드입니다. """
        if np.random.rand(1) < e:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state, :])

        # Get new state and reward from environment
        new_state, reward, done, _ = env.step(action)

        # Update Q-Table with new knowledge using learning rate
        Q[state, action] = reward + dis * np.max(Q[new_state, :])

        rAll += reward
        state = new_state

    rList.append(rAll)

print("Success rate: " + str(sum(rList) / num_episodes))
print("Final Q-Table Values")
print(Q)
plt.bar(range(len(rList)), rList, color="blue")
plt.show()
