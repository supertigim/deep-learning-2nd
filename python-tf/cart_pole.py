import numpy as np
import tensorflow as tf

import gym 
env = gym.make('CartPole-v0')

# Constants defining the neural network 
learning_rate = 1e-1
input_size = env.observation_space.shape[0] 	# definitely 4
output_size = env.action_space.n 			# probably 2

X = tf.placeholder(tf.float32, [None, input_size], name="input_x")

# First layer of weights
W1 = tf.get_variable("W1", shape=[input_size, output_size],
					initializer= tf.contrib.layers.xavier_initializer())

Qpred = tf.matmul(X, W1)

Y = tf.placeholder(shape=[None, output_size], dtype=tf.float32)

# Loss function 
loss = tf.reduce_sum(tf.square(Y - Qpred))

# Learning 
train = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# Values for q learning
random_episodes = 2000
dis = 0.9
rList = []

reward_sum = 0

#init = tf.global_variables_initializer()
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for i in range(random_episodes):

	e = 1. / ((i/10) + 1)
	rAll = 0
	step_count = 0
	done = False

	s = env.reset()

	while not done:
		step_count += 1
		x = np.reshape(s, [1, input_size])

		Qs = sess.run(Qpred, feed_dict={X: x})

		if np.random.rand(1) < e:
			a = env.action_space.sample()
		else:
			a = np.argmax(Qs)

		# Get new state and reward from environment 
		s1, reward, done, _ = env.step(a)

		if done:
			Qs[0, a] = -100
		else:
			x1 = np.reshape(s1, [1, input_size])

			#Obtain the Q' value by feeding the new state through the network
			Qs1 = sess.run(Qpred, feed_dict={X: x})
			Qs[0, a] = reward + dis * np.max(Qs1)

		# Train the network using target and predicted Q value on each episode
		sess.run(train, feed_dict={X: x, Y: Qs})

		s = s1

	rList.append(step_count)
	print("Episode: {} steps: {}".format(1, step_count))

	# If last 10's avg steps are 500, it's good enough
	if len(rList) > 10 and np.mean(rList[-10:]) > 500 :
		break


observation = env.reset()
reward_sum = 0
while True:
	env.render()

	x = np.reshape(observation, [1, input_size])
	Qs = sess.run(Qpred, feed_dict={X: x})
	a = np.argmax(Qs)

	observation, reward, done, _ = env.step(a)
	reward_sum += reward
	if done:
		print("Total score: {}".format(reward_sum))
		break


def test():
	env.render()

	action = env.action_space.sample()

	observation, reward, done, _ = env.step(action)

	print(observation, reward, done)

	reward_sum += reward

	if done:
		random_episodes += 1
		print ("Reward for this episode was ", reward_sum)
		reward_sum = 0
		env.reset()

