""" Dummy Q-learning (table)에 대한 코드입니다.

알고리즘 중에서 DFS(깊이 우선 탐색), BFS(너비 우선 탐색)를 안다면 이해하기 더 수월합니다.
[1], [2], [3], [4], [5], [6]을 잘 따라오세요~!
"""
import gym
import numpy as np
import matplotlib.pyplot as plt
# 여기서 갑자기 에러가 생겼지만 'sudo apt-get install python3 -tk'으로 해결하였습니다.
from gym.envs.registration import register
import random as pr

# random argmax에 대한 함수입니다.
# 만약에 Q값이 같다면 random하게 설정하기 위해서 만든 것입니다.
# ex) [0. 0. 0. 0] 일 경우 모든 값을 random하게 보내고 싶다!
def rargmax(vector):
    m = np.amax(vector)
    indices = np.nonzero(vector == m)[0]
    return pr.choice(indices)

register(
    id='FrozenLake-v3',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name': '4x4',
            'is_slippery': False}
)
env = gym.make('FrozenLake-v3')

""" [1] Initialize table with all zeros """
Q = np.zeros([env.observation_space.n, env.action_space.n])
# env.observation_space.n은 맵의 칸의 개수를 말합니다. 여기서는 총 16칸입니다.
# env.action_space.n은 action의 개수를 말합니다. 여기서는 동, 서, 남, 북으로서 4개입니다.

# Set learning parameters
num_episodes = 2000

# create lists to contain total rewards and steps per episode
# 결과를 저장하기 위해서 하나의 담을 그릇(list)을 만듭니다.
rList = []

for i in range(num_episodes):
    # Reset environment and get first new observation
    """ [2] 1 epoch마다 초기화를 시켜줍니다. """
    state = env.reset()
    rAll = 0
    done = False

    # The Q-Table learning algorithm
    while not done: # 게임이 끝나지 않을 때까지 계속 반복시킵니다.
        """ [3] action은 rargmax중에 가장 큰 값의 '방향' 으로 취합니다. """
        action = rargmax(Q[state, :])

        # Get new state and reward from environment
        """ [4] reward와 다음 state(s`)를 만듭니다. """
        new_state, reward, done, _ = env.step(action)

        # Update Q-Table with new knowledge using learning rate
        """ [5] Q-learning algorithm의 하이라이트~!! """
        Q[state, action] = reward + np.max(Q[new_state, :])
        # reward와 다음 state로 갔을 때에 동, 서, 남, 북 중에서 가장 큰 값을 더해서 Q에 넣습니다.

        rAll += reward # 결과를 저장할 곳에 넣기전에 reward를 계속 더해줍니다.
        state = new_state # [6] 다음 state를 지금 state라고 바꿔줍니다.

    rList.append(rAll) # 그릇(list)에 넣어줍니다.

print("Success rate: " + str(sum(rList) / num_episodes))
print("Final Q-Table Values")
print("LEFT DOWN RIGHT UP")
print(Q)
plt.bar(range(len(rList)), rList, color="blue")
plt.show()
