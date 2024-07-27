# This tutorial is based on the following page
# https://dilithjay.com/blog/actor-critic-methods
# it uses shared Network for traning both actor and critic
# importanting Important Libraries
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Define the actor-critic network
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc_pi = nn.Linear(64, action_dim)
        self.fc_v = nn.Linear(64, 1)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        pi = torch.softmax(self.fc_pi(x), dim=0)
        v = self.fc_v(x)
        
        return pi, v
    

#  # Define the environment and other parameters
env = gym.make('CartPole-v1')
num_episodes = 1000
discount_factor = 0.99

learning_rate = 0.001

# Initialize the ActorCritic network
agent = ActorCritic(env.observation_space.shape[0], env.action_space.n)

# Define the optimizer
optimizer = optim.Adam(agent.parameters(), lr=learning_rate)

########################
# Define the training loop
for episode in range(num_episodes):
    # Initialize the environment
    state, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        # Select an action using the agent's policy
        probs, val = agent(torch.tensor(state, dtype=torch.float32))
        action = np.random.choice(np.arange(len(probs)), p=probs.detach().numpy())

        # Take a step in the environment
        next_state, reward, done, _, _ = env.step(action)
        total_reward += reward

        # Calculate the TD error and loss
        _, next_val = agent(torch.tensor(next_state, dtype=torch.float32))
        err = reward + discount_factor * (next_val * (1 - done)) - val
        actor_loss = -torch.log(probs[action]) * err
        critic_loss = torch.square(err)
        loss = actor_loss + critic_loss

        # Update the network
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Set the state to the next state
        state = next_state

    # Print the total reward for the episode
    print(f'Episode {episode}: Total reward = {total_reward}')