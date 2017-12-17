# FrozenLake게임을 맛보기로 돌려보겠습니다.
import gym
from gym.envs.registration import register
import sys
import tty # 터미널 = tty = 텍스트 입출력 환경
# 유닉스 전문 용어로 tty는 단순히 읽기와 쓰기를 넘어 몇 가지 추가적인 명령어를 지원하는 디바이스 파일을 뜻합니다.
# 그리고 터미널은 사실상 이 tty와 동일한 의미로 사용됩니다.
# python docs
# The tty module defines functions for putting the tty into cbreak and raw modes.
# Because it requires the termios module, it will work only on Unix.

import termios # Low-level terminal control interface.
# This module provides an interface to the POSIX calls for tty I/O control.

class _Getch:
    def __call__(self):
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(3)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch

inkey = _Getch() # key(키보드 화살표 위, 아래, 왼쪽, 오른쪽)를 통해서 입력 하나 받아올 수 있습니다.

# MACROS
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3
# 왼쪽이 0, 아래가 1, 오른쪽이 2, 위가 3으로서 반시계반향으로 설정하였습니다.

# Key mapping
arrow_keys = {
    '\x1b[A': UP,
    '\x1b[B': DOWN,
    '\x1b[C': RIGHT,
    '\x1b[D': LEFT}

# Register FrozenLake with is_slippery False
# 게임을 하나 만듭니다.
register(
    id='FrozenLake-v3',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name': '4x4', 'is_slippery': False}
) # 일부러 4x4로 지정하였습니다. 더 크게 할 수도 있습니다.

env = gym.make('FrozenLake-v3')
env.render()  # Show the initial board

while True:
    # Choose an action from keyboard
    key = inkey()
    if key not in arrow_keys.keys(): # 화살표가 아닌 다른 key를 누르면 끝
        print("Game aborted!")
        break

    action = arrow_keys[key] # 화살표를 누르면 거기에 대응되는 액션!
    state, reward, done, info = env.step(action)
    env.render()  # Show the board after action
    print("State: ", state, "Action: ", action,
          "Reward: ", reward, "Info: ", info)

    if done: # 게임이 끝났을 때, Hole이면 0점, Goal이면 1점
        print("Finished with reward", reward)
        break
