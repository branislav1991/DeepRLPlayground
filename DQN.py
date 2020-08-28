import gym
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
import visdom

env = gym.make("FrozenLake-v0")
printout_freq = 30
lr = 0.0001
gamma = 0.99  # discount rate
num_epochs = 20
replay_buffer_size = 10000
batch_size = 32
max_env_steps = 100

# We want to over-sample frames where things happened. So we'll sort the buffer on the absolute reward
# (either positive or negative) and apply a geometric probability in order to bias our sampling to the
# earlier (more extreme) replays
unusual_sample_factor = 0.992


class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense1 = nn.Linear(env.observation_space.n, 32)
        self.dense2 = nn.Linear(32, env.action_space.n)

    def forward(self, x):
        x_out = F.relu(self.dense1(x))
        x_out = self.dense2(x_out)
        return x_out


# initialize network
dqn = DQN()

mse_loss = nn.MSELoss()
optimizer = Adam(dqn.parameters(), lr=lr)

viz = visdom.Visdom()
loss_window = viz.line(
    Y=torch.zeros((1)),
    X=torch.zeros((1)),
    opts=dict(xlabel='step', ylabel='Loss', title='training loss', legend=['Loss']))

cum_reward_window = viz.line(
    Y=torch.zeros((1)),
    X=torch.zeros((1)),
    opts=dict(xlabel='step', ylabel='Average Cum Reward', title='average cummulative reward', legend=['Average Cum Reward']))

training_step = 0
episode_step = 0

for epoch in range(num_epochs):
    exp_threshold = epoch / num_epochs
    replay_buffer = []

    dqn.eval()
    with torch.no_grad():
        # collect replay buffer samples
        observation = env.reset()
        for _ in range(replay_buffer_size):
            # run episode for max_num_steps
            for _ in range(max_env_steps):
                if np.random.rand() > exp_threshold:
                    action = np.random.randint(env.action_space.n)
                else:
                    input = torch.tensor(
                        observation, dtype=torch.int64).view((1))
                    input = F.one_hot(input, env.observation_space.n).float()
                    action = torch.argmax(dqn.forward(input)).item()

                new_observation, reward, done, _ = env.step(action)
                replay_buffer.append(
                    (observation, action, new_observation, reward, done))
                observation = new_observation

                if done:
                    observation = env.reset()
                    break

    # randomize replay buffer
    replay_buffer = np.random.permutation(replay_buffer)
    replay_buffer = np.stack(
        sorted(replay_buffer, key=lambda replay: abs(replay[3]), reverse=True))
    p = np.array([unusual_sample_factor **
                  i for i in range(len(replay_buffer))])
    p = p / sum(p)

    # update network
    dqn.train()
    for _ in range(1000):
        optimizer.zero_grad()

        sample_idxs = np.random.choice(
            np.arange(len(replay_buffer)), size=batch_size, p=p)
        sample = [replay_buffer[idx] for idx in sample_idxs]

        # tensors from replay buffer
        observations = torch.tensor(
            [r[0] for r in sample], dtype=torch.int64).view((-1))
        observations = F.one_hot(observations, env.observation_space.n).float()
        actions = torch.tensor([r[1] for r in sample],
                               dtype=torch.int64).view(-1, 1)
        new_observations = torch.tensor(
            [r[2] for r in sample], dtype=torch.int64).view((-1))
        new_observations = F.one_hot(
            new_observations, env.observation_space.n).float()
        rewards = torch.tensor([r[3] for r in sample], dtype=torch.float32)
        not_done_mask = torch.BoolTensor([not r[4] for r in sample])

        output_old = dqn.forward(observations)
        output_new = dqn.forward(new_observations)
        current_Q_values = output_old.gather(1, actions).view(-1)

        target_Q_values = torch.zeros((batch_size), dtype=torch.float32)
        next_Q_values = torch.max(output_new, 1)[0]
        target_Q_values[not_done_mask] = next_Q_values[not_done_mask]
        target_Q_values = (rewards + gamma * target_Q_values)

        bellman_error = mse_loss(target_Q_values, current_Q_values)
        bellman_error.backward()
        # clip error
        for param in dqn.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()

        # visualize training
        if training_step % printout_freq == 0:
            viz.line(X=torch.ones((1, 1))*(training_step), Y=torch.ones((1, 1))*bellman_error.sum().item(),
                     win=loss_window, update='append')
            print("Loss: {0}".format(bellman_error.sum().item()))
        training_step += 1

    # test learned policy
    cum_reward = 0.0
    for _ in range(100): # number of episodes
        observation = env.reset()
        for _ in range(100): # max number of steps
            input = torch.tensor(observation, dtype=torch.int64).view((1))
            input = F.one_hot(input, env.observation_space.n).float()
            action = torch.argmax(dqn.forward(input)).item()
            observation, reward, done, _ = env.step(action)
            cum_reward += reward

            if done:
                break

    viz.line(X=torch.ones((1, 1))*episode_step, Y=torch.ones((1, 1))*(cum_reward/100.0),
             win=cum_reward_window, update='append')
    episode_step += 1

env.close()
