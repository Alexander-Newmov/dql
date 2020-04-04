import gym
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import time
import numpy as np
import os
import matplotlib.pyplot as plt
import random
import math
from visualize_model import visualize_model

MIN_POSITION = -1.2
MAX_POSITION = 0.6
MIN_VELOCITY = -0.07
MAX_VELOCITY = 0.07

MEAN_POS = (MIN_POSITION + MAX_POSITION) / 2
STD_POS = abs(MIN_POSITION - MEAN_POS)
MEAN_VEL = (MIN_VELOCITY + MAX_VELOCITY) / 2
STD_VEL = abs(MIN_VELOCITY - MEAN_VEL)

STEPS_COUNT = 200
OBSERVATION_SIZE = 2
ACTIONS_COUNT = 3
ACTIONS_NAMES = ('Left', 'Stop', 'Right')

DEVICE = torch.device("cuda")


class Car_Net(torch.nn.Module):
	def __init__(self):
		super(Car_Net, self).__init__()

		self.fc1 = nn.Linear(2, 16)
		self.fc2 = nn.Linear(16, 32)
		self.fc3 = nn.Linear(32, 64)
		self.fc4 = nn.Linear(64, 3)

	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))
		x = self.fc4(x)

		return x


def normalize_observation(observation):
	normalized_observation = np.copy(observation)

	normalized_observation[..., 0] = (normalized_observation[..., 0] - MEAN_POS) / STD_POS
	normalized_observation[..., 1] = (normalized_observation[..., 1] - MEAN_VEL) / STD_VEL

	return normalized_observation


def get_action(model, observation, random_action_probability):
	if random.random() < random_action_probability:
		return random.randint(0, ACTIONS_COUNT - 1)

	normalized_observation = normalize_observation(observation)

	with torch.no_grad():
		best_action = np.argmax(model(torch.tensor(normalized_observation).to(DEVICE).float()).cpu().numpy())

	return best_action


def get_best_reward(model, observation):
	normalized_observation = normalize_observation(observation)

	with torch.no_grad():
		best_reward = np.amax(model(torch.tensor(normalized_observation).to(DEVICE).float()).cpu().numpy())

	return best_reward


def create_train_data(model, next_step_part, speed_part, random_action_probability, data_size):
	env = gym.make('MountainCar-v0')
	random_seed = int(time.time())
	env.seed(random_seed)

	observations = np.zeros(shape=(data_size, OBSERVATION_SIZE), dtype=np.float32)
	actions = np.zeros(shape=(data_size), dtype=np.float32)
	rewards = np.zeros(shape=(data_size), dtype=np.float32)
	counter = 0

	done = True
	while True:
		if done:
			observation = env.reset()

		for step in range(ACTIONS_COUNT):
			action = get_action(model, observation, random_action_probability)
			new_observation, reward, done, _ = env.step(action)
			reward += speed_part * abs(new_observation[1])

			if not done:
				reward += next_step_part * get_best_reward(model, new_observation)

			observations[counter] = observation
			actions[counter] = action
			rewards[counter] = reward
			counter += 1
			
			observation = new_observation

			if counter == data_size:
				randomize = np.arange(len(observations))
				np.random.shuffle(randomize)

				observations = observations[randomize]
				actions = actions[randomize]
				rewards = rewards[randomize]

				return observations, actions, rewards


def test_model(model, episodes_count, render_last=False):
	env = gym.make('MountainCar-v0')
	random_seed = int(time.time())
	env.seed(random_seed)
	np.random.seed(random_seed)

	with torch.no_grad():
		total_reward = 0
		finished_episodes = 0

		for episode in range(episodes_count):
			observation = env.reset()
			
			for step in range(STEPS_COUNT):
				if render_last and episode == episodes_count - 1:
					env.render()

				best_action = get_action(model, observation, 0)
				observation, reward, done, _ = env.step(best_action)
				total_reward += reward

				if done and step < STEPS_COUNT - 1:
					finished_episodes += 1
					break

				if done:
					break

	env.close()

	return total_reward / episodes_count, finished_episodes


def teach_model(lr, epochs_count, save_path, load_path=None):
	# Best parameters found
	next_step_part = 0.99
	speed_part = 10
	random_action_probability_max = 0.5
	random_action_probability_min = 0.1
	epoch_size = 128000
	batch_size = 128
	test_episodes_count = 100
	test_every = 1
	best_reward = -200
	average_rewards = []

	model = Car_Net()
	model.to(DEVICE)
	optimizer = optim.Adam(model.parameters(), lr=lr)

	if load_path:
		model.load_state_dict(torch.load(load_path))

	for epoch in range(epochs_count):
		print("Epoch: ", epoch)

		random_action_probability = random_action_probability_max - epoch * (random_action_probability_max - random_action_probability_min) / epochs_count

		if epoch % test_every == 0:
			average_reward, finished_episodes = test_model(model, test_episodes_count)
			average_rewards.append(average_reward)

			print('Average reward: ', average_reward)
			print('Finished episodes: ', finished_episodes)

			if average_reward > best_reward:
				best_reward = average_reward
				torch.save(model.state_dict(), save_path)

		observations, actions, rewards = create_train_data(model=model, next_step_part=next_step_part, speed_part=speed_part,
			random_action_probability=random_action_probability, data_size=epoch_size)
		normalized_observations = normalize_observation(observations)

		for step in range(epoch_size // batch_size):
			first = step * batch_size
			last = (step + 1) * batch_size

			batch_observations = torch.tensor(normalized_observations[first: last]).to(DEVICE).float()
			batch_actions = torch.tensor(actions[first: last]).to(DEVICE).long()
			batch_rewards = torch.tensor(rewards[first: last]).to(DEVICE).float()

			outputs = model(batch_observations)
			outputs = outputs.gather(-1, batch_actions.view(-1, 1))

			loss = F.smooth_l1_loss(outputs, batch_rewards.view(-1, 1))
			optimizer.zero_grad()
			loss.backward()

			optimizer.step()

	return average_rewards


def visualize_rewards(average_rewards, save_path):
	plt.figure()
	plt.ylim(-200, 0)
	plt.plot(average_rewards)
	plt.savefig(save_path)


if __name__ == "__main__":
	SHOULD_TRAIN = False

	FIRST_STEP_PATH = 'first_step.pth'
	SECOND_STEP_PATH = 'second_step.pth'

	if SHOULD_TRAIN:
		average_rewards = teach_model(lr=0.001, epochs_count=100, save_path=FIRST_STEP_PATH)
		visualize_rewards(average_rewards, 'first_step.png')

		average_rewards = teach_model(lr=0.000001, epochs_count=100, save_path=SECOND_STEP_PATH, load_path=FIRST_STEP_PATH)
		visualize_rewards(average_rewards, 'second_step.png')

	model = Car_Net()
	model.to(DEVICE)
	model.load_state_dict(torch.load(SECOND_STEP_PATH))
	visualize_model(model, 'model', '.')

	for i in range(10):
		average_reward, finished_episodes = test_model(model=model, episodes_count=100, render_last=True)

		print('Average reward: ', average_reward)
		print('Finished episodes: ', finished_episodes)
