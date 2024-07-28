# In this code --> seperate network is used for actor and Critic
# concept of detaching the tensor is used here
# it is based on following tutorial
# -->  https://arshren.medium.com/unlocking-the-secrets-of-actor-critic-reinforcement-learning-a-beginners-guide-3c5953b13551


# Import required libraries
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

# Create the Actor Network
'''
Defines a class "Actor" that implements a deep neural network model 
a simple feedforward neural network with 3 linear layers and ReLU activation functions
outputs the probability of taking a specific action given the current state.
'''
class Actor(nn.Module):
    def __init__(self, state_dim, action_size):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, action_size)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = F.softmax(self.fc3(out), dim=-1)
        return out

# Create the Critic Network
'''
The critic network estimates the expected return or value of a state or a state-action pair
a simple feedforward neural network with 3 linear layers and ReLU activation functions
outputs a scalar value, representing the estimated value of a state or a state-action pair
'''
class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

# Create the CartPole environment
env = gym.make("CartPole-v1")
env.reset(seed=0)
state_dim = env.observation_space.shape[0]
n_actions = env.action_space.n

# Create Actor and Critic
actor = Actor(state_dim, n_actions)
critic = Critic(state_dim)

# Setting the optimizer and learning rate
adam_actor = torch.optim.Adam(actor.parameters(), lr=1e-3)
adam_critic = torch.optim.Adam(critic.parameters(), lr=1e-3)
gamma = 0.99
episode_rewards = []
episode_num = 500
stats = {'actor_loss': [], 'critic_loss': [], 'return': []}

'''
each episode, the code resets the environment using the "env.reset()" method, 
repeatedly takes actions in the environment and 
updates the parameters of the actor and critic networks until the episode is done. 
The "done" flag is set to True when the episode is terminated.
'''
# Training loop
for i in range(episode_num):
    done = False
    total_reward = 0
    state, _ = env.reset(seed=0)
    while not done:
        '''
        The actor network outputs the probability of taking each possible action, 
        which are used to sample an action using the Categorical distribution 
        '''
        # The actor network outputs the probability of taking each possible action
        probs = actor(torch.from_numpy(state).float())
        dist = torch.distributions.Categorical(probs=probs)
        action = dist.sample()

        # Perform the action and receive the next state, reward, and "done" flag from the environment
        next_state, reward, done, _, _ = env.step(action.item())  # action needs to be converted to a Python scalar
        
        # The reward and the estimated value of the next state are used to calculate the advantage
        advantage = reward + (1 - done) * gamma * critic(torch.from_numpy(next_state).float()) - critic(torch.from_numpy(state).float())

        total_reward += reward
        state = next_state

        # Update Critic
        critic_loss = advantage.pow(2).mean()
        adam_critic.zero_grad()
        critic_loss.backward()
        adam_critic.step()

        # Update Actor
        actor_loss = -dist.log_prob(action) * advantage.detach()
        actor_loss = actor_loss.mean()  # Ensure actor_loss is a scalar
        adam_actor.zero_grad()
        actor_loss.backward()
        adam_actor.step()

    stats['actor_loss'].append(actor_loss.item())
    stats['critic_loss'].append(critic_loss.item())
    stats['return'].append(total_reward)

    episode_rewards.append(total_reward)

def plot_stats(stats):
    rows = len(stats)
    cols = 1

    fig, ax = plt.subplots(rows, cols, figsize=(12, 6))

    for i, key in enumerate(stats):
        vals = stats[key]
        if rows > 1:
            ax[i].plot(range(len(vals)), vals)
            ax[i].set_title(key, size=18)
        else:
            ax.plot(range(len(vals)), vals)
            ax.set_title(key, size=18)
    plt.tight_layout()
    plt.show()

plot_stats(stats)
