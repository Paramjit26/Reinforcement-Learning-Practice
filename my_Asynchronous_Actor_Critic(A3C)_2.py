# this code based on the following tutorial 
# --> https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/PolicyGradient/A3C/pytorch/a3c.py#L105
# --> Most important thing to learn in this code is how to use multiprocessing module for A3C algorithm
# --> there are lots of concepts involved in SharedAdam class 
# ------> param.groups, share_memory_(), state dictionary of optimizer, etc. it would be good to have some idea about this concepts 
import gymnasium as gym
import torch as T
import torch.multiprocessing as mp # helps in parallel computing task
import torch.nn as nn # to handle our layer
import torch.nn.functional as F # to handle our activation function
from torch.distributions import Categorical 
# it take s probability output from deep neural network 
# and map it to the distribution, so that we can do actual sampling 

class SharedAdam(T.optim.Adam):
    """This Class is basically used to put the Internal Attribute of the optimizer into shared memory
    """
    def __init__(self,params,lr = 1e-3, betas=(0.9,0.99),eps=1e-8, weight_decay=0):
        super(SharedAdam,self).__init__(params,lr=lr, betas=betas, eps=eps, weight_decay= weight_decay)

        for group in self.param_groups:
            for p in group['params']:
                # here p signify memory address of parameter tensor
                state = self.state[p]
                state['step'] = T.tensor(0.).share_memory_()
                state['exp_avg'] = T.zeros_like(p.data).share_memory_()
                state['exp_avg_sq'] = T.zeros_like(p.data).share_memory_()

                # state['exp_avg'].share_memory_()
                # state['exp_avg_sq'].share_memory_()

class ActorCritic(nn.Module):
    """We are using shared network Architecture!"""
    def __init__(self, input_dims,n_actions,gamma = 0.99):
        super(ActorCritic,self).__init__()

        self.gamma = gamma

        self.pi1 = nn.Linear(*input_dims,128)
        self.vi1 = nn.Linear(*input_dims,128)
        self.pi = nn.Linear(128,n_actions)
        self.v = nn.Linear(128,1)

        self.rewards = []
        self.actions = []
        self.states= []

    def remember(self,state,action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.rewards = []
        
    def forward(self,state):
        pi1 = F.relu(self.pi1(state))
        vi1 = F.relu(self.vi1(state))

        pi = self.pi(pi1)
        v= self.v(vi1)

        return pi,v
    
    def calc_R(self, done):
        """Used to calculate the discounte return"""
        states = T.tensor(self.states, dtype=T.float)
        _, v = self.forward(states) 

        R = v[-1].item()*(1-int(done))

        batch_return = []

        for reward in self.rewards[::-1]:
            R = reward + self.gamma*R
            batch_return.append(R)
        batch_return.reverse()
        batch_return = T.tensor(batch_return, dtype = T.float)

        return batch_return
    
    def calc_loss(self,done):
        """We calculate both actor loss and Critic Loss"""
        states = T.tensor(self.states, dtype=T.float)
        actions = T.tensor(self.actions, dtype=T.float)

        returns = self.calc_R(done)
        # getting our prediction 
        pi,values = self.forward(states)
        # calculating Critic loss
        values = values.squeeze()
        critic_loss = (returns -  values)**2

        # calculating Actor Loss 
        probs = T.softmax(pi,dim = 1) # dim = 1 ensure that softmax is applied along actions dimension not on states
        dist = Categorical(probs)
        log_probs = dist.log_prob(actions)
        actor_loss = -log_probs*(returns - values)

        total_loss = (critic_loss + actor_loss).mean()

        return total_loss

    def choose_action(self,observation):
        """this is to choose action based on the observation"""
        state = T.tensor([observation],dtype=T.float)
        pi,v = self.forward(state)
        probs = T.softmax(pi,dim=1)
        dist = Categorical(probs)
        action = dist.sample().item()

        return action

class Agent (mp.Process):
    def __init__(self, global_actor_critic, optimizer, input_dims, n_actions,
                 gamma, lr,name, global_ep_idx, env_id):
        super(Agent,self).__init__() # initialise the parent class properly
        self.local_actor_critic = ActorCritic(input_dims, n_actions, gamma)
        self.global_actor_critic = global_actor_critic
        self.name = 'w%02i'% name # 02 represents 2 digit place, i for integer
        self.episode_idx = global_ep_idx # for global episode counter
        self.env = gym.make(env_id) # set the local enviroment instance
        self.optimizer = optimizer # cautious we are using optimizer only for the global netwrok

    def run(self): # in Torch multiproessing module, run() function would be automatically excecute we don't need to call it
        t_step = 1
        while self.episode_idx.value < N_GAMES:
            # 1) self.episode_idx would be updated by all the workers (agent)
            done = False
            observation, _ = self.env.reset()
            score = 0
            self.local_actor_critic.clear_memory()
            while not done:
                # 2) Step loop --> Agent choose an action and record rewards, state and action
                action = self.local_actor_critic.choose_action(observation)
                observation_, reward, done, truncated,info = self.env.step(action)
                done = done or truncated
                score += reward
                self.local_actor_critic.remember(observation,action,reward)
                if t_step % T_MAX ==0 or done:
                    # 3) Conditional statement to update the Global Network 
                    loss = self.local_actor_critic.calc_loss(done)
                    self.optimizer.zero_grad()
                    loss.backward() # compute the gradient of loss w.r.t to local model parameter. 
                    for local_param, global_param in zip(
                        self.local_actor_critic.parameters(),
                        self.global_actor_critic.parameters()):
                        global_param._grad = local_param.grad # 4) copy gradient to global parameter
                    self.optimizer.step() # Caution this update the global paramter!!! Not the local parameters of workers

                    self.local_actor_critic.load_state_dict(
                        self.global_actor_critic.state_dict()) #  local model’s parameters are synchronized with the updated global model’s parameters.
                    
                    self.local_actor_critic.clear_memory()
                t_step += 1
                observation = observation_
            
            with self.episode_idx.get_lock():
                # use to update gobal episode value. 
                # using context manager to ensure that only one agent update value at a time
                self.episode_idx.value += 1
            
            print(self.name, 'episode',self.episode_idx.value, 'reward %.1f' % score)

if __name__ == '__main__':
    lr = 1e-4
    env_id = 'CartPole-v1'
    n_actions = 2
    input_dims = [4]
    N_GAMES = 3000
    T_MAX = 5
    global_actor_critic = ActorCritic(input_dims, n_actions)
    global_actor_critic.share_memory() #  Makes the model parameters shared across multiple processes, allowing different agents (workers) to read and write to the same network
    optim = SharedAdam(global_actor_critic.parameters(), lr=lr, betas=(0.92,0.99)) # Notice we are sharing paramters of global network
    global_ep = mp.Value('i',0)

    workers = [Agent(global_actor_critic,
                     optim,
                     input_dims,
                     n_actions,
                     gamma=0.99,
                     lr=lr,
                     name=i,
                     global_ep_idx=global_ep,
                     env_id=env_id) for i in range(mp.cpu_count())]
    
    [w.start() for w in workers]
    [w.join() for w in workers]