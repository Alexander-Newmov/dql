import os
import numpy as np
import torch
import matplotlib.pyplot as plt


MIN_POSITION = -1.2
MAX_POSITION = 0.6
MIN_VELOCITY = -0.07
MAX_VELOCITY = 0.07

OBSERVATION_SIZE = 2
ACTIONS_COUNT = 3
ACTIONS_NAMES = ('Left', 'Stop', 'Right')

DEVICE = torch.device("cuda")


def visualize_model(model, name, save_path):
	if not os.path.exists(save_path):
		os.makedirs(save_path)

	postition_grid_step = 0.01
	velocity_grid_step = 0.01
	position_grid = int((MAX_POSITION - MIN_POSITION) // postition_grid_step + 1)
	velocity_grid = int((MAX_VELOCITY - MIN_VELOCITY) // velocity_grid_step + 1)
	plot_Q = np.zeros(shape=(200, 200, ACTIONS_COUNT))

	with torch.no_grad():
		i = 0
		for position in np.arange(-1, 1, postition_grid_step):
			j = 0
			for velocity in np.arange(-1, 1, velocity_grid_step):
				plot_Q[i, j] = model(torch.tensor((position, velocity)).to(DEVICE).float()).cpu().numpy()
				j += 1
			i += 1

	plot_Q = np.swapaxes(plot_Q, 0, 1)

	# Plot rewards.
	figure, axes = plt.subplots(1, ACTIONS_COUNT + 1, figsize=(16, 5))
	min_Q = np.min(plot_Q)
	max_Q = np.max(plot_Q)
	for action in range(ACTIONS_COUNT):
		image = axes[action].imshow(plot_Q[:, :, action], cmap='jet', vmin=min_Q, vmax=max_Q)
		axes[action].set_title(ACTIONS_NAMES[action])
		axes[action].set_xlabel('Position')
		axes[action].set_ylabel('Velocity')

	figure.colorbar(image, axes[ACTIONS_COUNT])
	plt.savefig(os.path.join(save_path, name + '.png'))
	plt.close(figure)

	# Difference between right reward and left reward.
	figure, axes = plt.subplots(1, 2, figsize=(8, 5))
	image = axes[0].imshow(plot_Q[:, :, 2] - plot_Q[:, :, 0], cmap='jet', vmin=-2, vmax=2)
	axes[0].set_title('RIGHT - LEFT')
	axes[0].set_xlabel('Position')
	axes[0].set_ylabel('Velocity')

	figure.colorbar(image, axes[1])
	plt.savefig(os.path.join(save_path, 'difference_' + name + '.png'))
	plt.close(figure)

	# Selected action.
	figure, axes = plt.subplots(1, 2, figsize=(8, 5))
	image = axes[0].imshow(np.argmax(plot_Q, axis=2), cmap='jet')
	axes[0].set_title('ACTIONS')
	axes[0].set_xlabel('Position')
	axes[0].set_ylabel('Velocity')

	figure.colorbar(image, axes[1])
	plt.savefig(os.path.join(save_path, 'action_' + name + '.png'))
	plt.close(figure)
