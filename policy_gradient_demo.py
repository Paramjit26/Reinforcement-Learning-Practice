# importing Important libraries
import numpy as np
import gymnasium as gym
from collections import deque
import numpy as np

# for Video recording 
import imageio

## torch Modules
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
############## Setting the Device #############
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

############ 1) setting up our enviroment ##########

env_id = "CartPole-v1"
# Create the env
env = gym.make(env_id, render_mode='rgb_array')

# Create the evaluation env
eval_env = gym.make(env_id,render_mode='rgb_array')

# Get the state space and action space
s_size = env.observation_space.shape[0]
a_size = env.action_space.n

print("_____OBSERVATION SPACE_____ \n")
print("The State Space is: ", s_size)
print("Sample observation", env.observation_space.sample()) # Get a random observation

print("\n _____ACTION SPACE_____ \n")
print("The Action Space is: ", a_size)
print("Action Space Sample", env.action_space.sample()) # Take a random action

########### 2) Setting up the Hyperparameters #########
cartpole_hyperparameters = {
    "h_size": 16,
    "n_training_episodes": 1000,
    "n_evaluation_episodes": 10,
    "max_t": 1000,
    "gamma": 1.0,
    "lr": 1e-2,
    "env_id": env_id,
    "state_space": s_size,
    "action_space": a_size,
}
##### 3) Building our Neural Network ###################

class Policy(nn.Module):
    def __init__(self, s_size, a_size, h_size):
        super(Policy,self).__init__()
        ## here we will define neural network layers
        self.fc1 = nn.Linear(s_size,h_size)
        self.fc2 = nn.Linear(h_size,a_size)

    def forward(self,x):
        ### What need to be pass in our forward function
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(x,dim=1)

    def act(self, state):
        # what need to define in ac funtion?
        state = torch.from_numpy(state).float().unsqueeze(dim=0).to(device)
        probs =  self.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)
    
############ 4) Building our Reinforce function ###########

def reinforce(policy,optimizer, n_training_episodes, max_t, gamma, print_every):
    # 1) have to start with policy Model
    # here we are intialising our moving average and score 
    # -->  Moving average helps to provide a clearer 
    # -->  and more stable picture of the agent's learning progress 
    # -->  by mitigating the effects of high variance in individual episode rewards.
    scores_deque = deque(maxlen=100) 
    scores = []
    # 2) outside loop (A)
    for i_episode in range(1,n_training_episodes+1):
        # 3) --> Generate an Episode
        saved_log_probs = []
        rewards = []
        state, _ = env.reset()

        # 1st inner loop (episode generation)
        for t in range(max_t):
            action, log_prob = policy.act(state)
            saved_log_probs.append(log_prob)
            state, reward, done, truncated, _ = env.step(action)
            rewards.append(reward)
            if done or truncated:
                break
        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))

        # 4) --> Inner Loop (B)
        returns = deque(maxlen=max_t)
        n_steps = len(rewards)

        # 2nd inner loop (Discounted return calculation)
        for t in range(n_steps)[::-1]:
            disc_return_t = (returns[0] if len(returns)>0 else 0)
            returns.appendleft(gamma*disc_return_t + rewards[t])
        # 5) ----> return  
        # Standardization (Basically we are doing is feature scaling)
        eps = np.finfo(np.float32).eps.item()
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std()+eps)

        # 6) --> Loss
        policy_loss = []
        for log_prob, disc_return in zip(saved_log_probs,returns):
            policy_loss.append(-log_prob*disc_return)
        policy_loss = torch.cat(policy_loss).sum()


        # 7) --> optimise policy (Pytorch prefer Gradient Descent)
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        # Printing To view Agent learning progress
        if i_episode % print_every == 0:
            print("Episode {}\tAverage score:{:.2f}".format(i_episode,np.mean(scores_deque)))
    
    return scores


# Create policy and place it to the device
cartpole_policy = Policy(cartpole_hyperparameters["state_space"], cartpole_hyperparameters["action_space"], cartpole_hyperparameters["h_size"]).to(device)
cartpole_optimizer = optim.Adam(cartpole_policy.parameters(), lr=cartpole_hyperparameters["lr"])

scores = reinforce(cartpole_policy,
                   cartpole_optimizer,
                   cartpole_hyperparameters["n_training_episodes"], 
                   cartpole_hyperparameters["max_t"],
                   cartpole_hyperparameters["gamma"], 
                   100)

################## Evaluation ###########################################
def evaluate_agent(env, max_steps, n_eval_episodes, policy):
  """
  Evaluate the agent for ``n_eval_episodes`` episodes and returns average reward and std of reward.
  :param env: The evaluation environment
  :param n_eval_episodes: Number of episode to evaluate the agent
  :param policy: The Reinforce agent
  """
  episode_rewards = []
  for episode in range(n_eval_episodes):
    state, _ = env.reset()
    step = 0
    done = False
    total_rewards_ep = 0
    
    for step in range(max_steps):
      action, _ = policy.act(state)
      new_state, reward, done, truncated,_ = env.step(action)
      total_rewards_ep += reward
        
      if done or truncated:
        break
      state = new_state
    episode_rewards.append(total_rewards_ep)
  mean_reward = np.mean(episode_rewards)
  std_reward = np.std(episode_rewards)

  return mean_reward, std_reward

eval_mean_reward, eval_std = evaluate_agent(eval_env, 
                                           cartpole_hyperparameters["max_t"], 
                                           cartpole_hyperparameters["n_evaluation_episodes"],
                                           cartpole_policy)

print("Total reward {}\t std rewras{}".format(eval_mean_reward,eval_std))

####### recording the video of performance ############
def record_video(env, policy, out_directory, fps=30):
    """
    Generate a replay video of the agent
    :param env: The environment to record
    :param policy: The policy used by the agent
    :param out_directory: Directory to save the video
    :param fps: Frames per second for the video
    """
    images = []
    done = False
    state, _ = env.reset()
    img = env.render()
    images.append(img)
    frame_count = 0
    while not done:
        action, _ = policy.act(state)
        state, reward, done, truncated, _ = env.step(action)
        img = env.render()
        images.append(img)
        frame_count += 1
        if done or truncated:
            break
    imageio.mimsave(out_directory, [np.array(img) for i, img in enumerate(images)], fps=fps)
    
    duration = frame_count / fps
    print(f"Video saved to {out_directory}. Duration: {duration:.2f} seconds at {fps} FPS.")

# Assuming you have already defined your environment `env` and trained policy `cartpole_policy`
out_directory = "cartpole_agent.mp4"
record_video(env, cartpole_policy, out_directory, fps=30)

