# CartPole를 Q-learning을 이용하여 학습시켜보겠습니다.

import numpy as np
import tensorflow as tf

import gym
env = gym.make('CartPole-v0') # 게임을 불러옵니다.

# Constants defining our neural network
learning_rate = 1e-1
input_size = env.observation_space.shape[0] # input_size는 6_2에서 봤던 4가지가 input_size이기 때문에 '4'가 됩니다.
output_size = env.action_space.n # output_size는 action이 총 2가지이기 때문에 '2'가 됩니다.

X = tf.placeholder(tf.float32, [None, input_size], name="input_x")
# FrozenLake에서는 [1, input_size] 였지만, 보통은 [None, input_size]로 둡니다.
# 여기서는 그냥 [1, input_size]로 보셔도 무방합니다.

# First layer of weights
W1 = tf.get_variable("W1", shape=[input_size, output_size],
                     initializer=tf.contrib.layers.xavier_initializer())
# Variable로 가중치 매개변수를 줄 수도 있지만, get_variable로 줄 수도 있습니다.
# 'initializer=tf.contrib.layers.xavier_initializer()'은 Xavier 초깃값을 사용하려고 적은 코드입니다.
# Xavier 초깃값을 사용하면 앞 층에 노드가 많을수록 대상 노드의 초깃값으로 설정하는 가중치가 좁게 퍼집니다.(밑바닥부터 시작하는 딥러닝 205p.)

# 보통은 tf.Variable(tf.random_uniform(...))이나 tf.Variable(tf.random_normal(...))을 많이 쓰지만
# 시그모이드 함수같은 경우 0이나 1에 가까워지면 그 미분값이 0에 다가가서 역전파의 기울기 값이 점점 작아지다가 사라지는 gradient vanishing이
# 생기기 때문에 relu 함수를 사용하거나 층이 많을 경우 Xavier을 씁니다.
# 간단히 말하자면, 그냥 치우치지 않게 가중치를 좁게 모아주는 함수입니다.

Qpred = tf.matmul(X, W1)

# We need to define the parts of the network needed for learning a policy
Y = tf.placeholder(shape=[None, output_size], dtype=tf.float32)
# 여기서도 [None, output_size]를 [1, output_size]로 봐도 무방합니다.

# Loss function
loss = tf.reduce_sum(tf.square(Y - Qpred))
# Learning
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# Values for q learning
max_episodes = 5000
dis = 0.9
step_history = [] # step_count(몇 번 돌고 죽는지 개수)를 담을 그릇을 만듭니다.


# Setting up our environment
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for episode in range(max_episodes):
    e = 1. / ((episode / 10) + 1)
    step_count = 0
    state = env.reset()
    # print(state)
    # 여기서 프린트를 찍어본 이유는 밑에 'x = np.reshape(state, [1, input_size])'에서 reshape을 하는 이유가 궁금해서 했습니다.
    # [0.02016351 -0.03426143  0.00718547  0.02116652] 이런 식으로 state가 찍힙니다. 2차원 배열로 바꿔주어야 합니다.
    # 우리는 [[0.02016351 -0.03426143  0.00718547  0.02116652]]식으로 바꿔주어야 원핫인코딩처럼 되어 feed_dict에 넣을 수 있습니다.
    done = False

    # The Q-Network training
    while not done:
        step_count += 1 # 몇 번을 돌고 죽는지 개수를 셉니다.
        x = np.reshape(state, [1, input_size])
        # print(x)
        # [[-0.03321235  0.01036651 -0.15757138 -0.72332038]]

        # Choose an action by greedily (with e chance of random action) from
        # the Q-network
        Q = sess.run(Qpred, feed_dict={X: x})
        # x로 받아온 것을 Qpred에 넣고 돌린 것을 Q라고 지정합니다.
        # Qpred = tf.matmul(X, W1)

        # E-greedy를 하는 코드입니다.
        if np.random.rand(1) < e:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q)

        # Get new state and reward from environment
        next_state, reward, done, _ = env.step(action)
        if done:
            Q[0, action] = -100
            # 임의로 '너 크게 잘못했어! 다음부터는 그 행동 하지마!'라는 의미로 -100이라는 큰 숫자를 줍니다.
        else:
            x_next = np.reshape(next_state, [1, input_size])
            # 원핫인코딩이 필요가 없기 때문에 input_size인 4가지의 2차원 배열의 값을 바로 집어넣었습니다.

            # Obtain the Q' values by feeding the new state through our network
            Q_next = sess.run(Qpred, feed_dict={X: x_next})
            Q[0, action] = reward + dis * np.max(Q_next)
            # 여기서 [0, action]인 이유는 (1x2), 2는 left, right이기 때문에 앞의 행을 단순히 0으로 두는 것입니다.

        # Q[0, action] = -100 와 Q[0, action] = reward + dis * np.max(Q_next)가 label 즉, Q-learning 값이 될 것입니다.

        # Train our network using target and predicted Q values on each episode
        # 위에서 돌리는 것이 아니라 이 부분을 통해서 강화학습이 일어난다.
        # Q-learning 학습만 위의 코드를 통해서 action에 대한 Q-learning 값이 매겨지고
        # Neural Network(= linear regression, 여기서는 활성화함수를 안썼지만, 선형함수를 쓰고 있으므로 결국 NN = linear regression이다. 여기서만!!)을 통해서
        # Supervised Learning으로 위의 코드에서 train으로 정한 것으로 인해 학습이 이루어진다. 캐중요!!!!!
        sess.run(train, feed_dict={X: x, Y: Q})
        state = next_state # 다음 state를 지금 state로 바꿔줍니다.

    step_history.append(step_count) # 그릇에다가 개수를 적습니다.
    print("Episode: {}  steps: {}".format(episode, step_count))
    # If last 10's avg steps are 500, it's good enough
    if len(step_history) > 10 and np.mean(step_history[-10:]) > 500:
        break
    # 그릇에 10개 이상이고 그릇에 담긴 숫자가 500보다 높을 경우 그만!!

# See our trained network in action
# 학습이 잘 되었는지 체크해보겠습니다.
observation = env.reset()
reward_sum = 0
while True:
    env.render()

    x = np.reshape(observation, [1, input_size]) # 똑같이 reshape으로 변환해줍니다.
    Q = sess.run(Qpred, feed_dict={X: x})
    action = np.argmax(Q)

    observation, reward, done, _ = env.step(action)
    reward_sum += reward
    if done:
        print("Total score: {}".format(reward_sum))
        break
