# 강화학습이 어떤 식으로 돌아가는지 먼저 코드로 설명하겠습니다.

# 먼저 OpenAI gym이라는 라이브러리를 설치합니다.
import gym

# 1. 환경생성
env = gym.make('FrozenLake-v3') # 여기서 FrozenLake는 게임이름

# 2. 초기화
observation = env.reset() # map 정보를 전부 2차원 배열로 하여 0으로 초기화를 시킨다.

# 1epoch에 1000번을 실행할 수 있도록 할 것이다. while문을 돌려도 되고, for문으로 해도 상관없다.
for _ in range(1000): # 꼭 range를 1000으로 줄 필요는 없다.
    # 3. 화면출력(그려주는 함수)
    env.render()  # Show the initial board

    # 4. 액션 취하기
    action = env.action_space.sample()
    # 여기에 액션을 취하도록 알고리즘을 짜서 넣어줄 것이다.
    # FrozenLake에서는 위, 아래, 왼쪽, 오른쪽이 알고리즘이 될 것이다.

    # 5. setp마다 observation, reward, done, info 결과 나타내기
    observation, reward, done, info = env.step(action)
    # observation - 상태가 어느 위치로 변했는지 체크
    # reward - Goal에 도착했으면 1, Hole에 도착했으면 0
    # done - 게임이 끝났는지 안끝났는지 체크(FrozenLake에서는 끝났으면 True, 안끝났으면 False)
    # info - 추가정보가 있을 경우 알려준다.
