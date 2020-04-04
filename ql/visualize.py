import os
import matplotlib.pyplot as plt
import numpy as np

ACTIONS_COUNT = 3
ACTIONS_NAMES = ('Left', 'Stop', 'Right')

def visualize_Q(Q, name, save_path='.'):
	if not os.path.exists(save_path):
		os.makedirs(save_path)

	plot_Q = Q.copy()
	plot_Q = np.swapaxes(plot_Q, 0, 1)

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


	figure, axes = plt.subplots(1, 2, figsize=(8, 5))
	image = axes[0].imshow(plot_Q[:, :, 2] - plot_Q[:, :, 0], cmap='jet', vmin=-2, vmax=2)
	axes[0].set_title('RIGHT - LEFT')
	axes[0].set_xlabel('Position')
	axes[0].set_ylabel('Velocity')

	figure.colorbar(image, axes[1])
	plt.savefig(os.path.join(save_path, 'difference_' + name + '.png'))
	plt.close(figure)


	figure, axes = plt.subplots(1, 2, figsize=(8, 5))
	image = axes[0].imshow(np.argmax(plot_Q, axis=2), cmap='jet')
	axes[0].set_title('ACTION')
	axes[0].set_xlabel('Position')
	axes[0].set_ylabel('Velocity')

	figure.colorbar(image, axes[1])
	plt.savefig(os.path.join(save_path, 'action_' + name + '.png'))
	plt.close(figure)