import gym
import numpy as np
import time
import matplotlib.pyplot as plt
import csv
import os
from visualize import visualize_Q

MIN_POSITION = -1.2
MAX_POSITION = 0.6
MIN_VELOCITY = -0.07
MAX_VELOCITY = 0.07
MAX_ITERATION = 200

STEPS_COUNT = 200
ACTIONS_COUNT = 3
ACTIONS_NAMES = ('Left', 'Stop', 'Right')
NEEDED_REWARD = -108


def observation_to_grid(observation, position_grid_size, velocity_grid_size):
	grid_observation = np.zeros(shape=(2), dtype=np.int32)

	grid_observation[0] = int((observation[0] - MIN_POSITION) * (position_grid_size - 1) / (MAX_POSITION - MIN_POSITION))
	grid_observation[1] = int((observation[1] - MIN_VELOCITY) * (velocity_grid_size - 1) / (MAX_VELOCITY - MIN_VELOCITY))

	return grid_observation


def get_action(Q, grid_observation, random_action_probability):
	best_action = 0
	for another_action in range(ACTIONS_COUNT):
		if Q[grid_observation[0], grid_observation[1], another_action] > Q[grid_observation[0], grid_observation[1], best_action]:
			best_action = another_action

	action = best_action
	if np.random.uniform(low=0, high=1) < random_action_probability:
		action = np.random.randint(low=0, high=ACTIONS_COUNT)

	return action, best_action


def get_action_ensemble(Qs, grid_observation, random_action_probability):
	actions = [0] * Qs.shape[0]
	best_actions = [0] * Qs.shape[0]

	for q_number in range(Qs.shape[0]):
		Q = Qs[q_number]

		for another_action in range(ACTIONS_COUNT):
			if Q[grid_observation[0], grid_observation[1], another_action] > Q[grid_observation[0], grid_observation[1], best_actions[q_number]]:
				best_actions[q_number] = another_action

		actions[q_number] = best_actions[q_number]
		if np.random.uniform(low=0, high=1) < random_action_probability:
			actions[q_number] = np.random.randint(low=0, high=ACTIONS_COUNT)

	# action = np.argmax(actions, axis=1)
	action = np.bincount(actions).argmax()

	# best_actions = np.argmax(best_actions, axis=1)
	best_action = np.bincount(best_actions).argmax()

	return action, best_action


def train_Q(Q, lr, next_step_part, random_action_probability, position_grid_size, velocity_grid_size, episodes_count, speed_reward=0):
	env = gym.make('MountainCar-v0')
	random_seed = int(time.time())
	env.seed(random_seed)

	for episode in range(episodes_count):
		observation = env.reset()
		grid_observation = observation_to_grid(observation, position_grid_size, velocity_grid_size)
		action, _ = get_action(Q, grid_observation, random_action_probability)

		for step in range(STEPS_COUNT):
			
			new_observation, reward, done, _ = env.step(action)
			new_grid_observation = observation_to_grid(new_observation, position_grid_size, velocity_grid_size)
			new_action, best_new_action = get_action(Q, new_grid_observation, random_action_probability)

			# Speed reward.
			reward = reward + speed_reward * abs(new_observation[1])

			Q[grid_observation[0], grid_observation[1], action] += lr * (reward + next_step_part * Q[new_grid_observation[0], new_grid_observation[1], best_new_action] - Q[grid_observation[0], grid_observation[1], action])
			grid_observation = new_grid_observation
			action = new_action

			if done:
				break

	env.close()

	return Q


def test_Q(Q, position_grid_size, velocity_grid_size, episodes_count, is_ensemble=False, render_last=False):
	env = gym.make('MountainCar-v0')
	random_seed = int(time.time())
	env.seed(random_seed)

	total_reward = 0
	finished_episodes = 0
	for episode in range(episodes_count):
		observation = env.reset()
		
		for step in range(STEPS_COUNT):
			if render_last and episode == episodes_count - 1:
				env.render()

			grid_observation = observation_to_grid(observation, position_grid_size, velocity_grid_size)

			if is_ensemble:
				_, best_action = get_action_ensemble(Q, grid_observation, 0)
			else:
				_, best_action = get_action(Q, grid_observation, 0)

			observation, reward, done, _ = env.step(best_action)
			total_reward += reward

			if done and step < STEPS_COUNT - 1:
				finished_episodes += 1
				break

	env.close()
	return total_reward / episodes_count, finished_episodes


def compare_Q():
	random_seed = int(time.time())
	np.random.seed(random_seed)

	test_name = 'speed_reward'
	speed_rewards = (0, 0.5, 5, 10)

	lr = 0.1
	lr_decay = 1
	next_step_part = 0.99
	random_action_probability = 0.7
	random_action_probability_decay = 0.9
	position_grid_size = 50
	velocity_grid_size = 50

	train_episodes_count = 1000
	test_episodes_count = 100
	epochs_count = 30

	with open(test_name + '.csv', 'w') as file:
		columns = [None] * (epochs_count + 1)
		columns[0] = 'Name'
		for i in range(1, epochs_count + 1):
			columns[i] = str(i * train_episodes_count)

		writer = csv.DictWriter(file, columns)
		writer.writeheader()

		for speed_reward in speed_rewards:
			lr = 0.1
			random_action_probability = 0.7

			row_name = str(lr) + '_' + str(next_step_part) + '_' +  str(random_action_probability) + '_' + str(position_grid_size) + '_' + str(velocity_grid_size) + '_' + str(random_action_probability_decay) + '_with_lr_decay_' + str(lr_decay) + '_speed_reward_' + str(speed_reward) 

			row = {'Name': row_name}

			Q = np.zeros(shape=(position_grid_size, velocity_grid_size, ACTIONS_COUNT), dtype=np.float32)
			# Q = np.random.rand(position_grid_size, velocity_grid_size, ACTIONS_COUNT)

			for epoch in range(epochs_count):
				Q = train_Q(Q=Q, lr=lr, next_step_part=next_step_part, random_action_probability=random_action_probability,
					position_grid_size=position_grid_size, velocity_grid_size=velocity_grid_size, episodes_count=train_episodes_count,
					speed_reward=speed_reward)

				lr *= lr_decay
				random_action_probability *= random_action_probability_decay

				visualize_Q(Q=Q, name=str((epoch + 1) * train_episodes_count), save_path=row_name)

				average_reward, finished_episodes = test_Q(Q=Q, position_grid_size=position_grid_size, velocity_grid_size=velocity_grid_size, episodes_count=test_episodes_count)

				row[str((epoch + 1) * train_episodes_count)] = average_reward

				print('Epoch: ', epoch)
				print('Average reward: ', average_reward)
				print('Finished episodes: ', finished_episodes)

			writer.writerow(row)


def train_best_ensemble(ensemble_size):
	lr = 0.1
	lr_decay = 0.97
	next_step_part = 0.99
	random_action_probability = 0.7
	random_action_probability_decay = 0.97
	position_grid_size = 50
	velocity_grid_size = 50
	speed_reward = 10

	train_episodes_count = 1000
	test_episodes_count = 100
	epochs_count = 100

	Qs = np.zeros(shape=(ensemble_size, position_grid_size, velocity_grid_size, ACTIONS_COUNT), dtype=np.float32)

	for q_number in range(ensemble_size):
		lr = 0.1
		random_action_probability = 0.7
		current_Q = np.random.rand(position_grid_size, velocity_grid_size, ACTIONS_COUNT)
		best_reward = -200

		for epoch in range(epochs_count):
			current_Q = train_Q(Q=current_Q, lr=lr, next_step_part=next_step_part, random_action_probability=random_action_probability,
				position_grid_size=position_grid_size, velocity_grid_size=velocity_grid_size, episodes_count=train_episodes_count,
				speed_reward=speed_reward)

			lr *= lr_decay
			random_action_probability *= random_action_probability_decay

			current_reward, _ = test_Q(Q=current_Q, position_grid_size=position_grid_size,
				velocity_grid_size=velocity_grid_size, episodes_count=test_episodes_count)

			if current_reward > best_reward:
				best_reward = current_reward
				print('Q' + str(q_number) + ' epoch ' + str(epoch) + ' best reward: ' + str(best_reward))
				Qs[q_number] = current_Q.copy()

			if best_reward > NEEDED_REWARD:
				break

	return Qs


def test_best_ensemble(path, position_grid_size, velocity_grid_size, episodes_count, test_count):
	Qs = np.load(path)

	for test_number in range(test_count):
		if Qs.shape[0] > 1:
			average_reward, finished_episodes = test_Q(Q=Qs, position_grid_size=position_grid_size, velocity_grid_size=velocity_grid_size,
				episodes_count=episodes_count, is_ensemble=True, render_last=True)

			print('Ensemble average reward: ', average_reward)
			print('Ensemble finished episodes: ', finished_episodes)
			print('\n/////////\n')
		else:
			average_reward, finished_episodes = test_Q(Q=Qs[0], position_grid_size=position_grid_size, velocity_grid_size=velocity_grid_size,
			episodes_count=episodes_count, is_ensemble=False, render_last=True)

			print('Test number: ', test_number)
			print('Solo Q average reward: ', average_reward)
			print('Solo Q finished episodes: ', finished_episodes)
			print('\n/////////\n')


best_Q_path = 'best_Q.npy'

if __name__ == "__main__":
	SHOULD_TRAIN = False

	if SHOULD_TRAIN:
		Qs = train_best_ensemble(ensemble_size=1)
		np.save(best_Q_path, Qs)

test_best_ensemble(best_Q_path, position_grid_size=50, velocity_grid_size=50, episodes_count=100, test_count=10)
visualize_Q(np.load(best_Q_path)[0], name='best_Q', save_path='visualize')
