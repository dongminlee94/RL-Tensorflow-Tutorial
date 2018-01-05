# FrozenLake를 Q-Network를 이용하여 학습시켜보겠습니다~!

# 기존에 table로 했던 Q-learning은 100x100 미로나 80x80 픽셀에 RGB값까지 있는 게임에서 돌린다면
# 엄청나게 많은 array의 값들을 감당할 수 없습니다. 그렇기 때문에 Q-table을 사용할 수 없습니다.

# 따라서 table이 아닌 neural network(신경망)로 돌릴 것입니다..

# neural network(신경망)로 돌리는 방법은 두 가지가 있습니다.
# 첫 번째로, input으로 State(observation), Action으로 주고 network로 돌린 후 Q-value를 구하는 것
# 두 번째로, input으로 State만 주고 neural network(여기서는 linear regression, 활성화 함수 X)를 돌려 모든 Action값(Q-value)를 구하는 것
# 여기서는 두 번째 방법을 사용할 것입니다.

# 결국 network로 한다는 것은 Supervised Learning으로써
# output값으로 Q-learning value(non-deterministic Q-learning Algorithm이 아닌 처음에 배운 deterministic Q-learning)
# input값으로는 State만 주어서 neural network를 거친 값이 'y = R + r maxQ(s`)'와 같아지도록
# 즉, neural network(여기서는 linear regression, 활성화 함수 X)를 거친 cost를 converge하도록 만드는 Algorithm입니다.

# 여기서 'non-deterministic Q-learning Algorithm이 아닌 처음에 배운 deterministic Q-learning' 라고 했는데
# 이 이유는 cost를 converge되도록 만드는 것과 non-deterministic Q-learning Algorithm과 사실상 같은 작업을 하기 때문입니다.

# 하지만 Q-network에서는 큰 문제점이 있다. converge가 되어야 하는데 학습을 하면서 diverge가 된다.(발산한다.)
# 이유는 두 가지이다. Correlations between samples & Non-stationary targets이다.

# 하지만 DeepMind팀에서 DQN이라는 것이 나왔기 때문에 다음 07 Lab에서 공부해볼 것입니다.

import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v0')

# Input and output size based on the Env
input_size = env.observation_space.n # 16
output_size = env.action_space.n # 4
learning_rate = 0.1

# These lines establish the feed-forward part of the network used to
# choose actions
X = tf.placeholder(shape=[1, input_size], dtype=tf.float32)  # state input
# [1, 16]의 array판을 만들자. State(observation)를 돌기위한 판임!

W = tf.Variable(tf.random_uniform(
    [input_size, output_size], 0, 0.01))  # weight
# 여기서 0은 난수값 생성구간의 하한이고, 0.01은 난수값 생성구간의 상한입니다.

Qpred = tf.matmul(X, W)  # Out Q prediction
# 위의 코드가 바로 'input값으로는 State만 주어서 network를 거친 값'입니다.

Y = tf.placeholder(shape=[1, output_size], dtype=tf.float32)  # Y label
# 위의 코드는 위에서 언급했던 'output값으로 Q-learning value(non-deterministic Q-learning Algorithm이 아닌
# 처음에 배운 deterministic Q-learning)'을 하기 위해 공간을 만들어 놓은 코드입니다.
# 밑에서 'Qs[0, a] = reward + dis * np.max(Qs1)' 부분과 'sess.run(train, feed_dict={X: one_hot(s), Y: Qs})' 부분에서 담아질 것입니다.
# 무엇이? deterministic Q-learning value !!!

loss = tf.reduce_sum(tf.square(Y - Qpred))
train = tf.train.GradientDescentOptimizer(
    learning_rate=learning_rate).minimize(loss)

# Set Q-learning related parameters
dis = .99
num_episodes = 2000

# Create lists to contain total rewards and steps per episode
rList = []

# State를 돌을 때, 0,1로는 못하니까 one_hot encoding을 통해서 하나의 array만 hot하게 불을 켠다.
def one_hot(x):
    return np.identity(16)[x:x + 1]

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(num_episodes):
        # Reset environment and get first new observation
        s = env.reset()
        e = 1. / ((i / 50) + 10)
        rAll = 0
        done = False
        local_loss = []

        # The Q-Network training
        while not done:
            # Choose an action by greedily (with e chance of random action)
            # from the Q-network

            # print(s)
            # print(one_hot(s))
            # [[ 1.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]]
            # 이런식으로 데이터 전처리를 해주어야 합니다.

            Qs = sess.run(Qpred, feed_dict={X: one_hot(s)})
            # 여기서는 add random noise가 아닌 E-greedy 방법으로 E&E action을 하였습니다.
            if np.random.rand(1) < e:
                a = env.action_space.sample()
            else:
                a = np.argmax(Qs)

            # Get new state and reward from environment
            s1, reward, done, _ = env.step(a)

            # 여기서부터 학습하는 부분입니다.
            if done:
                # Update Q, and no Qs+1, since it's a terminal state
                Qs[0, a] = reward
            else:
                # Obtain the Q_s1 values by feeding the new state through our
                # network
                Qs1 = sess.run(Qpred, feed_dict={X: one_hot(s1)})
                # Update Q
                Qs[0, a] = reward + dis * np.max(Qs1)

            # Train our network using target (Y) and predicted Q (Qpred) values
            sess.run(train, feed_dict={X: one_hot(s), Y: Qs})
            # 여기서 a(action)만 학습을 시킵니다. 그 이유는 내가 한 action에 대한 Q값만 update를 시키면 되기 때문입니다.

            rAll += reward
            s = s1
        rList.append(rAll)

print("Percent of successful episodes: " +
      str(sum(rList) / num_episodes) + "%")
plt.bar(range(len(rList)), rList, color="blue")
plt.show()
