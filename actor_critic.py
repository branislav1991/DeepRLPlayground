from env_common import get_screen
from common import select_action_policy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import RMSprop
from itertools import count
import numpy as np
import gym
import visdom

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Hyperparameters
printout_freq = 1
num_episodes = 300
batch_size = 5
learning_rate_policy = 0.001
learning_rate_value = 0.001
gamma = 0.99
lam = 0.95 # lambda for GAE-lambda
train_v_iters = 10


# Initialize visualization
viz = visdom.Visdom()
loss_window = viz.line(
    Y=torch.zeros((1)),
    X=torch.zeros((1)),
    opts=dict(xlabel='step', ylabel='Loss', title='Training loss'))

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


class Buffer:
    
    def __init__(self, gamma, lam):
        self.buffer = []
        self.advantages = []
        self.discounted_rewards = []
        self.gamma = gamma
        self.lam = lam

    def add(self, state, action, value, reward):
        self.buffer.append((state, action, value, reward))

    def get(self, i):
        """Return state, action, discounted advantage and discounted reward at i.

        Requires that finalize() has been called previously to calculate
        discounted rewards.
        """
        if i >= len(self.buffer) or i >= len(self.advantages) or i >= len(self.discounted_rewards):
            return None
        else:
            state, action, _, _ = self.buffer[i]
            reward = self.discounted_rewards[i]
            advantage = self.advantages[i]
            return state, torch.FloatTensor([action]).to(device), advantage, reward

    def finalize(self):
        """Call at end of sample collection to calculate advantages and discounted rewards.
        """
        _, _, values, rewards = zip(*self.buffer)

        # Calculate advantages
        self.advantages = [0] * len(self.buffer)
        for i in range(len(self.advantages)-1):
            if rewards[i] != 0: # if reward is zero, we ended the episode
                delta = rewards[i] + self.gamma * values[i+1] - values[i]
                self.advantages[i] = delta.item()

        # Discount advantages
        running_add = 0
        for i in reversed(range(len(self.advantages))):
            if self.advantages[i] == 0:
                running_add = 0
            else:
                running_add = running_add * self.gamma * self.lam + self.advantages[i]
                self.advantages[i] = running_add

        # Normalize advantages
        adv_mean = np.mean(self.advantages)
        adv_std = np.std(self.advantages)
        for i in range(steps):
            self.advantages[i] = (self.advantages[i] - adv_mean) / adv_std

        # Calculate discounted rewards
        self.discounted_rewards = [0] * len(self.buffer)

        running_add = 0
        for i in reversed(range(len(self.discounted_rewards))):
            if rewards[i] == 0:
                running_add = 0
            else:
                running_add = running_add * self.gamma + rewards[i]
                self.discounted_rewards[i] = running_add


    def empty(self):
        self.buffer = []
        self.advantages = []
        self.discounted_rewards = []


buffer = Buffer(gamma, lam)


class PolicyNet(nn.Module):

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


policy_network = PolicyNet(screen_height, screen_width, num_actions).to(device)
value_network = PolicyNet(screen_height, screen_width, 1).to(device)
optimizer_policy = RMSprop(policy_network.parameters(), lr=learning_rate_policy)
optimizer_value = RMSprop(value_network.parameters(), lr=learning_rate_value)


# Store duration of episodes to test performance
episode_durations = []


# Training loop
steps = 0
training_step = 0
for episode in range(num_episodes):
    env.reset()
    last_screen = get_screen(env)
    current_screen = get_screen(env)
    state = current_screen - last_screen
    state = state.to(device)

    for t in count():
        action, _, val = select_action_policy(state, policy_network, value_network)
        _, reward, done, _ = env.step(action)

        # Move to next state
        last_screen = current_screen
        current_screen = get_screen(env)
        next_state = current_screen - last_screen
        next_state = next_state.to(device)

        # To mark boundarys between episodes
        if done:
            reward = 0

        buffer.add(state, float(action), val, reward)

        state = next_state

        steps += 1

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

    # Update policy
    if episode > 0 and episode % batch_size == 0:

        # Compute discounted rewards
        buffer.finalize()

        # Policy function learning
        optimizer_policy.zero_grad()
        for i in range(steps):
            state, action, advantage, _ = buffer.get(i)

            probs = policy_network(state).squeeze(0)
            m = torch.distributions.Categorical(logits=probs)
            policy_loss = -m.log_prob(action) * advantage  # Negtive score function x reward
            policy_loss.backward()
        optimizer_policy.step()

        # Value function learning
        for i in range(train_v_iters):
            optimizer_value.zero_grad()
            for i in range(steps):
                state, action, _, reward = buffer.get(i)

                value_loss = ((value_network(state).squeeze(0) - reward)**2).mean()
                value_loss.backward()
            optimizer_value.step()

        if training_step % printout_freq == 0:
            viz.line(X=torch.ones((1, 1)) * training_step, Y=torch.ones((1, 1)) * policy_loss.item(),
                    win=loss_window, update='append', name='Policy loss')

        training_step = training_step + 1

        buffer.empty()
        steps = 0