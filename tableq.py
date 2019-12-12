import gym
import numpy as np

env = gym.make("FrozenLake-v0")

# initialize Q-table
qtable = np.zeros((env.observation_space.n, env.action_space.n))
lr = 0.4
gamma = 0.9 # discount rate
num_episodes = 5000

observation = env.reset()
for episode in range(num_episodes):
  observation = env.reset()
  for t in range(100):
    #env.render()
    action = env.action_space.sample() # take random action
    new_observation, reward, done, info = env.step(action)

    # update equation
    update_val = reward + np.amax(qtable[new_observation,:]) - qtable[observation, action]
    qtable[observation, action] += lr * update_val

    observation = new_observation

    if done:
      qtable[new_observation, :] += lr * reward # final reward
      print("Episode finished after {} timesteps".format(t+1))
      break

env.close()

print('Final Q table:')
print(qtable)