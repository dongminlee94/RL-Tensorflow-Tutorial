""" CartPole를 DQN(NIPS 2015)을 이용하여 학습시켜보겠습니다. """

"""
이전에 보셨던 RL_Lab_07_1.py는
- Solution
1. Go deep
2. Capture and replay (Correlations between samples)
3. Separate networks (Non-stationary targets)

1, 2만 적용한 DQN이였습니다. NIPS 2015는 3번까지 적용한 DQN의 최종본이라고 보시면 되겠습니다.
DQN에 대한 이론은 RL_Lab_07_1.py을 참고바랍니다.
"""
import numpy as np
import tensorflow as tf
import random
import dqn
from collections import deque

import gym
env = gym.make('CartPole-v1')

# Constants defining our neural network
input_size = env.observation_space.shape[0] # 4
output_size = env.action_space.n # 2

dis = 0.9
REPLAY_MEMORY = 50000

def replay_train(mainDQN, targetDQN, tarin_batch):
	x_stack = np.empty(0).reshape(0, input_size)
	y_stack = np.empty(0).reshape(0, output_size)

	# Get stored information from the buffer
	for state, action, reward, next_state, done in tarin_batch:
		Q = mainDQN.predict(state)

		# terminal?
		if done:
			Q[0, action] = reward
		else :
			# Obtain the Q' values by feeding the new state through our network
			Q[0, action] = reward + dis * np.max(targetDQN.predict(next_state))
			"""이제는 mainDQN.predict로 Q-value를 불러오지 않고, targetDQN.predict로 불러옵니다."""

		y_stack = np.vstack([y_stack, Q])
		x_stack = np.vstack([x_stack, state])

	# Train our network using target and predicted Q values on each episode
	return mainDQN.update(x_stack, y_stack)
	"""
	업데이트는 mainDQN으로 합니다.
	여기서 y_stack는 target network로 빌드업된 y(Q-learning value)입니다.
	"""


def bot_play(mainDQN) :
	# See our trained network in action
	s = env.reset()
	reward_sum = 0

	while True:
		env.render()
		a = np.argmax(mainDQN.predict(s))
		s, reward, done, _ = env.step(a)
		reward_sum += reward
		if done:
			print("Total score: {}".format(reward_sum))
			break

def get_copy_var_ops( dest_scope_name="target", src_scope_name="main"):
	"""
	이 부분이 바로 복사를 하는 함수입니다. main -> target으로 w(세타)를 맞출 때 사용할 것입니다.
	scope_name은 주로 이름을 명할 때 주로 사용합니다."""
	# Copy variables src_scope to dest_scope
	op_holder = []

	src_vars = tf.get_collection( tf.GraphKeys.TRAINABLE_VARIABLES, scope= src_scope_name)
	dest_vars = tf.get_collection( tf.GraphKeys.TRAINABLE_VARIABLES, scope= dest_scope_name)
	"""
	여기서 TRAINABLE_VARIABLES는 바뀌는 w의 값들만
	src_vars, dest_vars에 담을 수 있도록 하는 함수입니다.
	"""

	for src_var, dest_var in zip(src_vars, dest_vars):
		# dest_var's mean : tesnor
		op_holder.append(dest_var.assign(src_var.value()))
	"""
	for문을 돌면서 src_vars, dest_vars에 있는 값들을 하나씩 꺼냅니다.
	dest_var가 target이었고, src_var가 main이었으니까 assign이라는 함수를 통해서
	dest_var = src_var로 복사를 합니다. 즉 src_var -> dest_var으로 복사합니다.
	복사한 값들을 append를 통해서 op_holder에 담습니다.
	"""

	return op_holder

def main():
	max_episodes = 5000

	# store the previous observations in replay memory
	replay_buffer = deque()

	with tf.Session() as sess :
		"""
		여기서 network를 두 개를 생성합니다. 생성은 sess를 통해서 하겠죠?
		"""
		mainDQN = dqn.DQN(sess, input_size, output_size, name ="main")
		targetDQN = dqn.DQN(sess, input_size, output_size, name="target")
		tf.global_variables_initializer().run()

		# initial copy q_net -> target_net
		"""
		초기의 w(세타)를 같게 만들고 시작합니다. 처음에는 당연히 network가 동일해야 합니다.
		다른 network로 해봤자 나중에 복사할 때 의미가 없기 때문입니다.
		"""
		copy_ops = get_copy_var_ops(dest_scope_name="target", src_scope_name="main")

		"""복사한 것을 실행시킵니다."""
		sess.run(copy_ops)

		for episode in range(max_episodes):
			e = 1. / ((episode / 10) + 1)
			done = False
			step_count = 0

			state = env.reset()

			while not done:
				if np.random.rand(1) < e :
					action = env.action_space.sample()
				else :
					# Choose an action by greedilty from the Q-network
					action = np.argmax(mainDQN.predict(state))


				# Get new state and reward from environment
				next_state, reward, done, _ = env.step(action)
				if done: #big penalty
					reward = -100

				# Save the experience to our buffer
				replay_buffer.append((state, action, reward, next_state, done))
				if len(replay_buffer) > REPLAY_MEMORY:
					replay_buffer.popleft()

				state = next_state
				step_count += 1

			print("Episode: {}  steps: {} ".format(episode, step_count))

			if episode % 10 == 1 : # train every 10 episodes
				# Get a random batch of experiences.
				for _ in range(50):
					# Minibatch works better
					minibatch = random.sample(replay_buffer, 10)
					loss, _ = replay_train(mainDQN, targetDQN, minibatch)
					"""
					replay_train에 mainDQN, minibatch가 아니라
					mainDQN, targetDQN, minibatch 값이 들어갑니다.
					"""

				print("Loss: ", loss)

				# copy q_net -> target_net
				sess.run(copy_ops)
				"""학습된 main(q) network의 w값을 target network의 w값으로 복사를 합니다."""

		bot_play(mainDQN)

if __name__ == "__main__":
	main()
