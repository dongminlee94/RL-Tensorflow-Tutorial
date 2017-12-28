# Dummy Q-learning (table)
# exploit&exploration and discounted future reward 에 대한 코드입니다.

# exploit&exploration and discounted future reward을 해주는 이유는 Lab3에서는 랜덤하지 못하게(유연하지 못하게) state를 돌기 때문입니다.
# 안가본 곳도 가봐야 더 좋은 길이 있을 수도 있고, 더 효율성이 증가하기 때문에 안가본 곳도 한 번 가볼 필요가 있습니다.

# 두 가지만 정해주면 됩니다.
# Select an action a -> exploit(내가 현재 있는 값을 이용한다.) & exploration(한 번 모험한다, 도전한다.)
# Receive immediate reward r - > discounted reward

## Select an action a -> exploit & exploration에는 세 가지 방법이 있습니다.
## 'E-greedy'와 'dacaying E-greedy'와 'add random noise'
## 지금 보여드리는 코드는 add random noise이고, RL_Lab_04_2.py에 있는 것은 E-greedy입니다.

# 기본적인 코드들은 RL_Lab_03_1.py를 참고 바랍니다.

import gym
import numpy as np
import matplotlib.pyplot as plt
from gym.envs.registration import register

register(
    id='FrozenLake-v3',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name': '4x4',
            'is_slippery': False}
)

env = gym.make('FrozenLake-v3')

# Initialize table with all zeros
Q = np.zeros([env.observation_space.n, env.action_space.n])

# Discount factor
dis = 0.99 # discounted reward를 설정해줍니다. 보통은 0.99 or 0.9로 합니다.
num_episodes = 2000

# create lists to contain total rewards and steps per episode
rList = []

for i in range(num_episodes):
    # Reset environment and get first new observation
    state = env.reset()
    rAll = 0
    done = False

    # The Q-Table learning algorithm
    while not done:
        # Choose an action by greedily (with noise) picking from Q table
        action = np.argmax(Q[state, :] + np.random.randn(1, env.action_space.n) / (i + 1))

        # Get new state and reward from environment
        new_state, reward, done, _ = env.step(action)

        # Update Q-Table with new knowledge using decay rate
        Q[state, action] = reward + dis * np.max(Q[new_state, :]) # 여기서 discounted reward를 정했던 dis가 들어갑니다.

        rAll += reward
        state = new_state

    rList.append(rAll)

print("Success rate: " + str(sum(rList) / num_episodes))
print("Final Q-Table Values")
print(Q)
plt.bar(range(len(rList)), rList, color="blue")
plt.show()
