# FrozenLake 게임을 Q-learning없이 해보겠습니다~!

import gym
import readchar # while True: 부분에서 key를 받아들이는 부분이 윈도우와 맥os가 다르기 때문에 설치하여 사용하면 원활하게 사용이 가능하다.
# 설치 : pip3 install readchar

# MACROS
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

# Key mapping
arrow_keys = {
    '\x1b[A': UP,
    '\x1b[B': DOWN,
    '\x1b[C': RIGHT,
    '\x1b[D': LEFT}

# is_slippery True
env = gym.make('FrozenLake-v0')

env.render()  # Show the initial board

# 'FrozenLake-v0'은 'FrozenLake-v3'과는 다르게 env.step()함수 안에 reset해주는 기능이 없다.
# 따라서 이와 같이 적어주어야 한다.
env.reset()

while True:
    # Choose an action from keyboard
    key = readchar.readkey()
    if key not in arrow_keys.keys():
        print("Game aborted!")
        break

    action = arrow_keys[key]
    state, reward, done, info = env.step(action)
    env.render()  # Show the board after action
    print("State: ", state, "Action: ", action,
          "Reward: ", reward, "Info: ", info)

    if done:
        print("Finished with reward", reward)
        break
