from env_common import get_screen
from common import select_action_dqn
import gym
from itertools import count
import math
import numpy as np
import random
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
import visdom

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Hyperparameters
printout_freq = 10
lr = 0.001
gamma = 0.999  # discount rate
num_episodes = 1000
replay_buffer_size = 10000
batch_size = 128
soft_update = True
soft_update_coeff = 0.3
target_update_freq = 1


# Initialize visualization
viz = visdom.Visdom()
loss_window = viz.line(
    Y=torch.zeros((1)),
    X=torch.zeros((1)),
    opts=dict(xlabel='step', ylabel='Loss', title='training loss'))

episode_length_window = viz.line(
    Y=torch.zeros((1)),
    X=torch.zeros((1)),
    opts=dict(xlabel='step', ylabel='Episode length', title='Episode length'))


# Initialize environment and replay buffer
env = gym.make("CartPole-v0")
env.reset()

init_screen = get_screen(env)
_, _, screen_height, screen_width = init_screen.shape
num_actions = env.action_space.n


class ReplayBuffer:
    """Stores collected samples of experience up to a maximum amount.
    """
    def __init__(self, max_size):
        self.buffer = []
        self.max_size = max_size
        self.position = 0

    def __len__(self):
        return len(self.buffer)

    def add(self, *sample):
        """Add experience to replay buffer.
        """
        if len(self.buffer) < self.max_size:
            self.buffer.append(sample)
        else:
            self.buffer[self.position % self.max_size] = sample
            self.position = (self.position + 1) % self.max_size

    def sample(self, size):
        """Sample a batch of size 'size' from replay buffer.
        """
        return random.sample(self.buffer, min(len(self.buffer), size))

replay_buffer = ReplayBuffer(replay_buffer_size)


class DQN(nn.Module):

    def __init__(self, h, w, outputs):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


# Initialize network and optimizer
online_dqn = DQN(screen_height, screen_width, num_actions).to(device)
target_dqn = DQN(screen_height, screen_width, num_actions).to(device)
target_dqn.load_state_dict(online_dqn.state_dict())
target_dqn.eval()

optimizer = Adam(online_dqn.parameters(), lr=lr)


# Store duration of episodes to test performance
episode_durations = []


# Training loop
training_step = 0
for episode in range(num_episodes):
    exp_threshold = 0.05 + (0.9 - 0.05) * math.exp(-1. * training_step / 400.0) # adjust exploration threshold

    env.reset()

    last_screen = get_screen(env)
    current_screen = get_screen(env)
    state = current_screen - last_screen

    for t in count():
        # Select action, move environment and store in buffer
        action = select_action_dqn(state.to(device), online_dqn, env, exp_threshold)
        _, reward, done, _ = env.step(action)
        last_screen = current_screen
        current_screen = get_screen(env)
        next_state = current_screen - last_screen
        replay_buffer.add(state, action, next_state, torch.tensor(reward), torch.tensor(done))

        state = next_state

        # Update network
        if len(replay_buffer) >= batch_size:
            optimizer.zero_grad()

            batch = replay_buffer.sample(batch_size)
            states, actions, next_states, rewards, done_mask  = zip(*batch)
            states = torch.cat(states).to(device)
            actions = torch.tensor(actions).to(device)
            next_states = torch.cat(next_states).to(device)
            rewards = torch.stack(rewards).to(device)
            done_mask = torch.stack(done_mask).to(device)

            output_current = online_dqn.forward(states)
            output_target = target_dqn.forward(next_states)
            current_Q_values = output_current.gather(1, actions.unsqueeze(1)).view(-1)

            target_Q_values = torch.max(output_target, 1)[0].detach()
            target_Q_values[done_mask] = 0
            target_Q_values = (rewards + gamma * target_Q_values)

            bellman_error = F.smooth_l1_loss(current_Q_values, target_Q_values)
            bellman_error.backward()

            # Clip error
            for param in online_dqn.parameters():
                param.grad.data.clamp_(-1, 1)
            optimizer.step()

            if soft_update:
                target_state_dict = target_dqn.state_dict()
                for name, param in online_dqn.state_dict().items():
                    param = param * soft_update_coeff + target_state_dict[name] * (1-soft_update_coeff)
                    target_state_dict[name].copy_(param)
            else:
                if training_step % target_update_freq == 0:
                    target_dqn.load_state_dict(online_dqn.state_dict())

            if training_step % printout_freq == 0:
                viz.line(X=torch.ones((1, 1)) * training_step, Y=torch.ones((1, 1)) * bellman_error.sum().item(),
                        win=loss_window, update='append', name='Bellman loss')

            training_step = training_step + 1

        if done:
            episode_durations.append(t + 1)
            viz.line(X=torch.ones((1, 1))*episode, Y=torch.ones((1, 1)) * episode_durations[-1],
                win=episode_length_window, update='append', name='Episode durations')

            # Plot 50 episode averages
            if len(episode_durations) >= 50:
                mean = np.mean(episode_durations[-50:])
                viz.line(X=torch.ones((1, 1))*episode, Y=torch.ones((1, 1)) * mean,
                    win=episode_length_window, update='append', name='Mean episode durations')

            break

print('Complete')
env.close()
