#from env_common import get_screen
from common import select_action_ddpg
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
lr_actor = 1e-4
lr_critic = 1e-3
gamma = 0.99  # discount rate
num_episodes = 100
replay_buffer_size = 20000
batch_size = 128
soft_update_coeff = 1e-1


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


"""
Taken from https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py
"""
class OUNoise:
    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000):
        self.mu           = mu
        self.theta        = theta
        self.sigma        = max_sigma
        self.max_sigma    = max_sigma
        self.min_sigma    = min_sigma
        self.decay_period = decay_period
        self.action_dim   = action_space.shape[0]
        self.low          = action_space.low
        self.high         = action_space.high
        self.reset()
        
    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu
        
    def evolve_state(self):
        x  = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state
    
    def get_action(self, action, t=0): 
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)

noise = OUNoise(env.action_space)


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
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, state):
        """
        Param state is a torch tensor
        """
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = torch.tanh(self.linear3(x))

        return x


# Initialize network and optimizer
actor = ActorNet(num_states, 256, num_actions).to(device)
actor_target = ActorNet(num_states, 256, num_actions).to(device)
actor_target.load_state_dict(actor.state_dict())
actor_target.eval()

critic = CriticNet(num_states+num_actions, 256, num_actions).to(device)
critic_target = CriticNet(num_states+num_actions, 256, num_actions).to(device)
critic_target.load_state_dict(critic.state_dict())
critic_target.eval()

optimizer_actor = Adam(actor.parameters(), lr=lr_actor)
optimizer_critic = Adam(critic.parameters(), lr=lr_critic)


# Store rewards of episodes to test performance
episode_rewards = []


# Training loop
training_step = 0
for episode in range(num_episodes):
    episode_reward = 0

    state = env.reset()
    state = torch.tensor(state, dtype=torch.float32)
    noise.reset()

    env.render(mode='rgb_array')

    for t in count():
        # Select action, move environment and store in buffer
        action = select_action_ddpg(state.to(device).unsqueeze(0), actor)
        action = noise.get_action(action, t)
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
            critic.train()
            actor.train()

            batch = replay_buffer.sample(batch_size)
            states, actions, next_states, rewards, done_mask  = zip(*batch)
            states = torch.stack(states).to(device)
            actions = torch.stack(actions).to(device)
            next_states = torch.stack(next_states).to(device)
            rewards = torch.stack(rewards).to(device)
            done_mask = torch.stack(done_mask).to(device)

            # Update critic
            current_Q_values = critic.forward(states, actions)
            next_actions = actor_target.forward(next_states)
            target_Q_values = critic_target.forward(next_states, next_actions.detach())
            target_Q_values[done_mask] = 0
            target_Q_values = rewards.unsqueeze(1) + gamma * target_Q_values

            bellman_error = F.mse_loss(current_Q_values, target_Q_values)

            optimizer_critic.zero_grad()
            bellman_error.backward()
            optimizer_critic.step()

            # Update actor
            policy_loss = -critic.forward(states, actor.forward(states)).mean()

            optimizer_actor.zero_grad()
            policy_loss.backward()
            optimizer_actor.step()

            # Soft update target
            target_state_dict = critic_target.state_dict()
            for name, param in critic.state_dict().items():
                p = param * soft_update_coeff + target_state_dict[name] * (1-soft_update_coeff)
                target_state_dict[name].copy_(p)

            target_state_dict = actor_target.state_dict()
            for name, param in actor.state_dict().items():
                p = param * soft_update_coeff + target_state_dict[name] * (1-soft_update_coeff)
                target_state_dict[name].copy_(p)

            if training_step % printout_freq == 0:
                viz.line(X=torch.ones((1, 1)) * training_step, Y=torch.ones((1, 1)) * bellman_error.item(),
                        win=loss_window, update='append', name='Bellman loss')
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
