#from env_common import get_screen
from common import select_action_policy
import gym
from itertools import count
import math
import numpy as np
import random
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions.normal import Normal
from torch.optim import Adam
import visdom

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Hyperparameters
printout_freq = 10
lr_actor = 1e-4
lr_critic = 1e-3
gamma = 0.99 # discount rate
num_episodes = 50
replay_buffer_size = 20000
batch_size = 128
soft_update_coeff = 1e-1
alpha=0.2 # entropy regularization factor


# Initialize visualization
viz = visdom.Visdom()
loss_window = viz.line(
    Y=torch.zeros((1)),
    X=torch.zeros((1)),
    opts=dict(xlabel='step', ylabel='Loss', title='Training loss'))

episode_reward_window = viz.line(
    Y=torch.zeros((1)),
    X=torch.zeros((1)),
    opts=dict(xlabel='step', ylabel='Episode reward', title='Episode rewards'))


# Initialize environment and replay buffer
class NormalizedEnv(gym.ActionWrapper):
    """ Wrap action """
    def __init__(self, env):
        super().__init__(env)

    def action(self, action):
        act_k = (self.action_space.high - self.action_space.low)/ 2.
        act_b = (self.action_space.high + self.action_space.low)/ 2.
        return act_k * action + act_b

    def reverse_action(self, action):
        act_k_inv = 2./(self.action_space.high - self.action_space.low)
        act_b = (self.action_space.high + self.action_space.low)/ 2.
        return act_k_inv * (action - act_b)

env = NormalizedEnv(gym.make("Pendulum-v0"))
env.reset()

num_actions = env.action_space.shape[0]
num_states = env.observation_space.shape[0]


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


class CriticNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, state, action):
        """
        Params state and actions are torch tensors
        """
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)

        return x

class ActorNet(nn.Module):
    LOG_STD_MAX = 2
    LOG_STD_MIN = -20

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)

        self.mu_layer = nn.Linear(hidden_size, output_size)
        self.log_std_layer = nn.Linear(hidden_size, output_size)
        
    def forward(self, state, deterministic=False):
        """
        Param state is a torch tensor
        """
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)

        mu = self.mu_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
        logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
        logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)

        pi_action = torch.tanh(pi_action)
        return pi_action, logp_pi


# Initialize network and optimizer
actor = ActorNet(num_states, 256, num_actions).to(device)

critic1 = CriticNet(num_states+num_actions, 256, num_actions).to(device)
critic1_target = CriticNet(num_states+num_actions, 256, num_actions).to(device)
critic1_target.load_state_dict(critic1.state_dict())
critic1_target.eval()

critic2 = CriticNet(num_states+num_actions, 256, num_actions).to(device)
critic2_target = CriticNet(num_states+num_actions, 256, num_actions).to(device)
critic2_target.load_state_dict(critic2.state_dict())
critic2_target.eval()

optimizer_actor = Adam(actor.parameters(), lr=lr_actor)
optimizer_critic1 = Adam(critic1.parameters(), lr=lr_critic)
optimizer_critic2 = Adam(critic2.parameters(), lr=lr_critic)


# Store rewards of episodes to test performance
episode_rewards = []


# Training loop
training_step = 0
for episode in range(num_episodes):
    episode_reward = 0

    state = env.reset()
    state = torch.tensor(state, dtype=torch.float32)

    env.render(mode='rgb_array')

    for t in count():
        # Select action, move environment and store in buffer
        action, logp = actor.forward(state.to(device).unsqueeze(0))
        action = action.item()
        next_state, reward, done, _ = env.step(action)
        action = torch.tensor(action, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        reward = torch.tensor(reward, dtype=torch.float32)

        env.render(mode='rgb_array')

        replay_buffer.add(state, action, next_state, reward, torch.tensor(done))

        state = next_state
        episode_reward += reward

        # Update network
        if len(replay_buffer) >= batch_size:
            critic1.train()
            critic2.train()
            actor.train()

            batch = replay_buffer.sample(batch_size)
            states, actions, next_states, rewards, done_mask  = zip(*batch)
            states = torch.stack(states).to(device)
            actions = torch.stack(actions).to(device)
            next_states = torch.stack(next_states).to(device)
            rewards = torch.stack(rewards).to(device)
            done_mask = torch.stack(done_mask).to(device)

            # Update critics
            with torch.no_grad():
                next_actions, logp = actor.forward(next_states)
                target1_Q_values = critic1_target.forward(next_states, next_actions.detach())
                target2_Q_values = critic2_target.forward(next_states, next_actions.detach())
                target_Q_values = torch.min(target1_Q_values, target2_Q_values)
                target_Q_values = target_Q_values - alpha * logp.unsqueeze(1)

                target_Q_values = rewards.unsqueeze(1) + gamma * target_Q_values
                target_Q_values[done_mask] = rewards.unsqueeze(1)[done_mask]

            # Critic 1
            current_Q_values = critic1.forward(states, actions.unsqueeze(1))
            loss_critic1 = F.mse_loss(current_Q_values, target_Q_values)
            optimizer_critic1.zero_grad()
            loss_critic1.backward()
            optimizer_critic1.step()

            # Critic 2
            current_Q_values = critic2.forward(states, actions.unsqueeze(1))
            loss_critic2 = F.mse_loss(current_Q_values, target_Q_values)
            optimizer_critic2.zero_grad()
            loss_critic2.backward()
            optimizer_critic2.step()

            # Update actor
            actions, logp = actor.forward(states)
            Q1_values = critic1(states, actions)
            Q2_values = critic2(states, actions)
            Q_values = torch.min(Q1_values, Q2_values)

            # Entropy-regularized policy loss
            policy_loss = (alpha * logp - Q_values).mean()

            optimizer_actor.zero_grad()
            policy_loss.backward()
            optimizer_actor.step()

            # Soft update critics
            target_state_dict = critic1_target.state_dict()
            for name, param in critic1.state_dict().items():
                p = param * soft_update_coeff + target_state_dict[name] * (1-soft_update_coeff)
                target_state_dict[name].copy_(p)

            target_state_dict = critic2_target.state_dict()
            for name, param in critic2.state_dict().items():
                p = param * soft_update_coeff + target_state_dict[name] * (1-soft_update_coeff)
                target_state_dict[name].copy_(p)

            if training_step % printout_freq == 0:
                viz.line(X=torch.ones((1, 1)) * training_step, Y=torch.ones((1, 1)) * loss_critic1.item(),
                        win=loss_window, update='append', name='Critic 1 loss')
                viz.line(X=torch.ones((1, 1)) * training_step, Y=torch.ones((1, 1)) * loss_critic2.item(),
                        win=loss_window, update='append', name='Critic 2 loss')
                viz.line(X=torch.ones((1, 1)) * training_step, Y=torch.ones((1, 1)) * policy_loss.item(),
                        win=loss_window, update='append', name='Policy loss')

            training_step = training_step + 1

        if done:
            episode_rewards.append(episode_reward)
            viz.line(X=torch.ones((1, 1))*episode, Y=torch.ones((1, 1)) * episode_rewards[-1],
                win=episode_reward_window, update='append', name='Episode rewards')

            # Plot 50 episode averages
            if len(episode_rewards) >= 10:
                mean = np.mean(episode_rewards[-10:])
                viz.line(X=torch.ones((1, 1))*episode, Y=torch.ones((1, 1)) * mean,
                    win=episode_reward_window, update='append', name='Mean episode rewards')

            break

print('Complete')
env.close()
