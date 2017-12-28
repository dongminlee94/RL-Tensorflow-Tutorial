# 실제 FrozenLake 게임을 stochastic(non-deterministic) Q-learning으로 돌려봅시다~!

# 기존의 Deterministic한 FrozenLake는 오른쪽으로 가면 1칸만 가기 때문에 더 쉬웠습니다.
# 하지만 원래의 FrozenLake 게임은 Stochastic(non-deterministic)하기 때문에,
# 즉 미끄럽기 때문에 오른쪽으로 한 칸을 가도 두 칸을 갈 수도 있고, 세 칸을 갈 수도 있습니다.
# 또한 미끄러지기 때문에 밑으로 갈 수도 있습니다. 방향에 대해서 무결성해집니다.
# Stochastic models possess some inherent randomness.
# The same set of parameter values and initial conditions will lead to an ensemble of different outputs.

# 미끄러지는 것을 계산하지 못하는 Q-learning을 따르면 게임이 원활하게 학습하지 못합니다.
# 그렇다면 Solution은 무엇일까요?

# 그것은 바로 Q-learning을 조금만 적용하는 것입니다.
# 하나의 예를 들자면, 한 명의 멘토가 있다고 합니다. 그 멘토의 말을 고집있게 들을 필요는 없습니다.
# 왜냐하면 그 멘토는 자기가 와본 길(살았던 길)에 대해서만 설명해주기 때문입니다.
# 우리는 때로는 우리의 고집대로 살아갈 필요가 있습니다. 우리의 인생은 우리가 선택하는 것이기 때문입니다.
# 그러므로 멘토가 하는 말 10%듣고 내가 가고 싶은대로 90%를 하면됩니다.
# Listen to Q(s') (just a little bit)
# Update Q(s) little bit (learning rate)

# 따라서 Q-learning을 통해 업데이트를 할 때 100% 다 믿는 것이 아니라 10%만 듣고 나머지 90%는 내가 하고 싶은 대로 갑니다.
# 밑에 코드에서 stochastic(non-deterministic) Q-learning Algorithm 코드를 적용시켜보겠습니다.

import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v0')

action_space_n = 4
state_space_n = 16

# Initialize table with all zeros
Q = np.zeros([state_space_n, action_space_n])

# Set learning parameters
learning_rate = .85
# 여기서는 빨리 학습을 시키고 싶어서 Q-learning에 85%를 주고, 내 고집을 15% 주겠습니다.
# learning rate가 커지면 빨리 학습을 하고, 작아지면 천천히 학습을 합니다.

# 왜 커지면 빨리 학습하고, 작아지면 천천히 학습하죠??

dis = .99
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
        action = np.argmax(Q[state, :] + np.random.randn(1,
                                                         action_space_n) / (i + 1))

        # Get new state and reward from environment
        new_state, reward, done, _ = env.step(action)

        # Update Q-Table with new knowledge using learning rate
        # 이 부분이 바로 stochastic(non-deterministic) Q-learning Algorithm 공식을 적용한 코드입니다.
        Q[state, action] = (1 - learning_rate) * Q[state, action] \
            + learning_rate * (reward + dis * np.max(Q[new_state, :]))

        rAll += reward
        state = new_state

    rList.append(rAll)

env.close()

print("Score over time: " + str(sum(rList) / num_episodes))
print("Final Q-Table Values")
print(Q)
plt.bar(range(len(rList)), rList, color="blue")
plt.show()
