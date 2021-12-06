import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import matplotlib.pyplot as plt
import numpy as np
import random
import gym
import math
from collections import deque
import time


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True



class episodic_buffer(object):
    def __init__(self, capacity, actors, device):
        self.capacity = capacity
        self.actors = actors
        self.device = device
        
        self.episodic_buffer = deque(maxlen=self.capacity)
        self.mask = torch.zeros((self.actors, self.capacity)).to(self.device)
        
    def _get_episodic_info(self):
        state_embedding, target_return = zip(*list(self.episodic_buffer))
        
        state_embedding = torch.stack(state_embedding, 0).transpose(0, 1)
        target_return = torch.stack(target_return, 0).transpose(0, 1)
        
        mask = self.mask[:, :state_embedding.shape[1]]
        
        return state_embedding, target_return, mask.unsqueeze(-1)
        
    def store(self, state_embedding, target_return):
        target_return = torch.FloatTensor(target_return).to(self.device)
        
        self.episodic_buffer.append((state_embedding, target_return))
        self.mask = torch.roll(self.mask, -1, 1)
        self.mask[:,-1] = 1
        
    def reset(self, actor):
        self.mask[actor] = 0


class sequence_buffer(object):
    def __init__(self, sequence_length, overlapping_length, burn_in_length):
        self.sequence_length = sequence_length
        self.overlapping_length = overlapping_length
        self.burn_in_length = burn_in_length
        
        self.sequence_buffer = deque(maxlen=self.sequence_length + self.burn_in_length)
        
    def _get_sequence_info(self):
        observation, action, reward, next_observation, done, hidden, prior = zip(*list(self.sequence_buffer))
        
        observation = np.array(observation)
        action = np.array(action)
        reward = np.array(reward)
        next_observation = np.array(next_observation)
        done = np.array(done)
        prior = np.array(prior)
        
        if len(self.sequence_buffer) == (self.sequence_length + self.burn_in_length):
            _ = [self.sequence_buffer.popleft() for _ in range(self.overlapping_length)]
            
        return observation, action, reward, next_observation, done, hidden[0], prior
        
    def store(self, observation, action, reward, next_observation, done, hidden, prior):
        while len(self.sequence_buffer) < self.burn_in_length:
            self.sequence_buffer.append((
                np.zeros_like(observation), 0, 0.0, np.zeros_like(next_observation), False, (np.zeros_like(hidden[0]), np.zeros_like(hidden[1])), np.zeros_like(prior)
            ))
            
        self.sequence_buffer.append((observation, action, reward, next_observation, done, hidden, prior))
        
    def reset(self):
        self.sequence_buffer.clear()


class local_buffer(object):
    def __init__(self, sequence_length, overlapping_length, burn_in_length, eta):
        self.sequence_length = sequence_length
        self.overlapping_length = overlapping_length
        self.burn_in_length = burn_in_length
        self.eta = eta
        
        self.sequence_buffer = sequence_buffer(self.sequence_length, self.overlapping_length, self.burn_in_length)
        
    def add_global_buffer(self, learner, observation, action, reward, next_observation, done, hidden, prior):
        hidden = (hidden[0].cpu().numpy(), hidden[1].cpu().numpy())
        
        self.sequence_buffer.store(observation, action, reward, next_observation, done, hidden, prior)
        if done:
            pass
        elif len(self.sequence_buffer.sequence_buffer) < (self.sequence_length + self.burn_in_length):
            return
            
        observation, action, reward, next_observation, done, hidden, prior = self.sequence_buffer._get_sequence_info()
        
        if len(learner.buffer.memory) < learner.buffer.capacity:
            learner.buffer.memory.append((observation, action, reward, next_observation, done, hidden))
        else:
            learner.buffer.memory[learner.buffer.pos] = (observation, action, reward, next_observation, done, hidden)
            
        prior = prior[:, 0]
        prior_max = prior[-self.sequence_length:].max()
        prior_mean = prior[-self.sequence_length:].mean()
        
        prior = self.eta * prior_max + (1 - self.eta) * prior_mean
        
        learner.buffer.priorities[learner.buffer.pos] = prior
        learner.buffer.pos = (learner.buffer.pos + 1) % learner.buffer.capacity


class global_buffer(object):
    def __init__(self, capacity, alpha, beta_start, beta_frames, sequence_length, burn_in_length, observation_dim):
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.sequence_length = sequence_length
        self.burn_in_length = burn_in_length
        self.observation_dim = observation_dim
        self.frame = 1
        self.pos = 0
        
        self.memory = []
        self.priorities = np.zeros((self.capacity))
        
    def beta_by_frame(self, frame_idx):
        return min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames)
        
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            
    def sample(self, size):
        if len(self.memory) < self.capacity:
            probs = self.priorities[:len(self.memory)]
        else:
            probs = self.priorities
            
        probs = probs ** self.alpha
        probs = probs / np.sum(probs)
        
        indices = np.random.choice(len(self.memory), size, p=probs)
        samples = [self.memory[idx] for idx in indices]
        
        beta = self.beta_by_frame(self.frame)
        self.frame += 1
        
        weights = (len(self.memory) * probs[indices]) ** (-beta)
        weights = weights / np.max(weights)
        weights = np.array(weights, dtype=np.float32)
        
        observation_list, action_list, reward_list, next_observation_list, done_list, hidden = zip(*samples)
        
        observation = np.zeros((size, self.sequence_length + self.burn_in_length, self.observation_dim))
        action = np.zeros((size, self.sequence_length + self.burn_in_length))
        reward = np.zeros((size, self.sequence_length + self.burn_in_length))
        next_observation = np.zeros((size, self.sequence_length + self.burn_in_length, self.observation_dim))
        done = np.zeros((size, self.sequence_length + self.burn_in_length))
        masks = np.zeros((size, self.sequence_length + self.burn_in_length))
        
        for i in range(len(observation_list)):
            observation[i, :len(observation_list[i])] = observation_list[i]
            action[i, :len(action_list[i])] = action_list[i]
            reward[i, :len(reward_list[i])] = reward_list[i]
            next_observation[i, :len(next_observation_list[i])] = next_observation_list[i]
            done[i, :len(done_list[i])] = done_list[i]
            masks[i, self.burn_in_length:len(observation_list[i])] = 1
            
        return observation, action, reward, next_observation, done, indices, weights, masks, hidden


class state_associative_net(nn.Module):
    def __init__(self, hidden_dim):
        super(state_associative_net, self).__init__()
        self.hidden_dim = hidden_dim
        
        self.gate_fc1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.gate_fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.gate_fc3 = nn.Linear(self.hidden_dim, 1)
        
        self.bias_fc1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.bias_fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.bias_fc3 = nn.Linear(self.hidden_dim, 1)
        
        self.synthetic_return_fc1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.synthetic_return_fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.synthetic_return_fc3 = nn.Linear(self.hidden_dim, 1)
        
    def gate(self, x):
        out = F.relu(self.gate_fc1(x))
        out = F.relu(self.gate_fc2(out))
        out = torch.sigmoid(self.gate_fc3(out))
        
        return out
        
    def bias(self, x):
        out = F.relu(self.bias_fc1(x))
        out = F.relu(self.bias_fc2(out))
        out = self.bias_fc3(out)
        
        return out
        
    def synthetic_return(self, x):
        out = F.relu(self.synthetic_return_fc1(x))
        out = F.relu(self.synthetic_return_fc2(out))
        out = self.synthetic_return_fc3(out)
        
        return out


class temporal_difference_net(nn.Module):
    def __init__(self, observation_dim, hidden_dim, action_dim, device):
        super(temporal_difference_net, self).__init__()
        self.observation_dim = observation_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.device = device
        
        self.lstm = nn.LSTM(self.observation_dim, self.hidden_dim, batch_first=True)
        
        self.advantage = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.action_dim)
        )
        
        self.value = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1)
        )
        
    def forward(self, observation, hidden=None):
        state_embedding, hidden = self.lstm(observation, hidden)
        x = F.relu(state_embedding)
        
        advantage = self.advantage(x)
        value = self.value(x)
        
        return advantage + value - advantage.mean(dim=2, keepdim=True), hidden, state_embedding
        
    def get_action(self, observation, hidden):
        observation = torch.FloatTensor(observation).to(self.device).unsqueeze(1)
        
        with torch.no_grad():
            q_value, hidden, state_embedding = self.forward(observation, hidden)
            action = q_value.max(2)[1].detach().cpu().numpy()
            
        return action, hidden, state_embedding


class Actor(object):
    def __init__(self, actors, device, observation_dim, hidden_dim, action_dim, sequence_length, overlapping_length, burn_in_length, eta):
        self.actors = actors
        self.device = device
        self.sequence_length = sequence_length
        self.overlapping_length = overlapping_length
        self.burn_in_length = burn_in_length
        self.eta = eta
        
        self.epsilon_init = 0.9
        self.epsilon_min = 0.001
        self.decay = np.array([(5000. * self.actors) / actor for actor in range(1, self.actors + 1)])
        
        self.observation_dim = observation_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        
        self.online_net = temporal_difference_net(self.observation_dim, self.hidden_dim, self.action_dim, self.device).to(self.device)
        self.target_net = temporal_difference_net(self.observation_dim, self.hidden_dim, self.action_dim, self.device).to(self.device)
        
        self.buffers = [local_buffer(self.sequence_length, self.overlapping_length, self.burn_in_length, self.eta) for actor in range(self.actors)]
        
    def epsilon_fn(self, count):
        return self.epsilon_min + (self.epsilon_init - self.epsilon_min) * np.exp(-1. * count / self.decay)
        
    def get_action(self, observation, hidden, count):
        epsilon = self.epsilon_fn(count + np.arange(self.actors))
        action, hidden, state_embedding = self.online_net.get_action(observation, hidden)
        
        randoms = np.random.uniform(low=0.0, high=1.0, size=observation.shape[0])
        action[randoms < epsilon] = np.expand_dims(np.random.randint(self.action_dim, size=(randoms < epsilon).sum()), 1)
        
        return action, hidden, state_embedding


class Learner(object):
    def __init__(self, capacity, learning_rate, device, alpha, beta_start, beta_frames, sequence_length, burn_in_length, observation_dim, hidden_dim, action_dim):
        self.capacity = capacity
        self.learning_rate = learning_rate
        self.device = device
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.sequence_length = sequence_length
        self.burn_in_length = burn_in_length
        
        self.observation_dim = observation_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        
        self.online_net = temporal_difference_net(self.observation_dim, self.hidden_dim, self.action_dim, self.device).to(self.device)
        self.target_net = temporal_difference_net(self.observation_dim, self.hidden_dim, self.action_dim, self.device).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=self.learning_rate)
        self.buffer = global_buffer(self.capacity, self.alpha, self.beta_start, self.beta_frames, self.sequence_length, self.burn_in_length, self.observation_dim)
        
    def update_model(self):
        self.target_net.load_state_dict(self.online_net.state_dict())
        
    def push_model(self, actor):
        actor.online_net.load_state_dict(self.online_net.state_dict())
        actor.target_net.load_state_dict(self.target_net.state_dict())


class r2d2(object):
    def __init__(self, capacity, actors, batch_size, learning_rate, device, alpha, beta_start, beta_frames, gamma,
                 sequence_length, overlapping_length, burn_in_length, eta, state_associative_capacity, state_associative_learning_rate,
                 return_alpha, return_beta, hidden_dim):
        self.capacity = capacity
        self.actors = actors
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.gamma = gamma
        self.sequence_length = sequence_length
        self.overlapping_length = overlapping_length
        self.burn_in_length = burn_in_length
        self.eta = eta
        self.state_associative_capacity = state_associative_capacity
        self.state_associative_learning_rate = state_associative_learning_rate
        self.return_alpha = return_alpha
        self.return_beta = return_beta
        
        self.observation_dim = envs[0].observation_space.shape[0]
        self.hidden_dim = hidden_dim
        self.action_dim = envs[0].action_space.n
        
        self.actor = Actor(self.actors, self.device, self.observation_dim, self.hidden_dim, self.action_dim,
                           self.sequence_length, self.overlapping_length, self.burn_in_length, self.eta)
        self.learner = Learner(self.capacity, self.learning_rate, self.device, self.alpha, self.beta_start, self.beta_frames,
                               self.sequence_length, self.burn_in_length, self.observation_dim, self.hidden_dim, self.action_dim)
        
        self.state_associative_net = state_associative_net(self.hidden_dim).to(self.device)
        self.state_associative_optimizer = torch.optim.Adam(self.state_associative_net.parameters(), self.state_associative_learning_rate)
        self.state_associative_buffer = episodic_buffer(self.state_associative_capacity, self.actors, self.device)
        
    def get_synthetic_return(self):
        state_embedding, target_return, mask = self.state_associative_buffer._get_episodic_info()
        
        with torch.no_grad():
            synthetic_return = self.state_associative_net.synthetic_return(state_embedding[:,-1:])
            
        bias = self.state_associative_net.bias(state_embedding[:,-1:]).squeeze()
        gate = self.state_associative_net.gate(state_embedding[:,-1:]).squeeze()
        
        past_synthetic_return = self.state_associative_net.synthetic_return(state_embedding[:,:-1]) * mask[:,:-1]
        synthetic_return_sum = past_synthetic_return.sum(1).squeeze()
        
        prediction = gate * synthetic_return_sum - bias
        
        with torch.no_grad():
            augmented_return = self.return_alpha * synthetic_return[:,0,0] + self.return_beta * target_return[:,-1]
            
        loss = 0.5 * (prediction - target_return[:,-1]).pow(2).mean()
        
        self.state_associative_optimizer.zero_grad()
        loss.backward()
        self.state_associative_optimizer.step()
        
        return augmented_return.detach().cpu().numpy()
        
    def get_td_error(self, online_net, target_net, observation, action, reward, next_observation, done, masks, hidden):
        q_values, _, _ = online_net.forward(observation, hidden)
        
        with torch.no_grad():
            next_q_values, _, _ = target_net.forward(next_observation, hidden)
            argmax_actions, _, _ = online_net.forward(next_observation, hidden)
            argmax_actions = argmax_actions.max(2)[1]
            
        q_value = q_values.gather(2, action.unsqueeze(2)).squeeze(2)
        next_q_value = next_q_values.gather(2, argmax_actions.unsqueeze(2)).squeeze(2)
        
        expected_q_value = reward + next_q_value * (1 - done) * self.gamma
        
        td_error = expected_q_value - q_value
        td_error = td_error * masks
        
        return td_error
        
    def get_prior(self, online_net, target_net, observation, action, reward, next_observation, done, hidden):
        observation = torch.FloatTensor(observation).to(self.device).unsqueeze(1)
        action = torch.LongTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device).unsqueeze(1)
        next_observation = torch.FloatTensor(next_observation).to(self.device).unsqueeze(1)
        done = torch.FloatTensor(done).to(self.device).unsqueeze(1)
        masks = torch.ones(self.actors, 1).to(self.device)
        
        with torch.no_grad():
            td_error = self.get_td_error(online_net, target_net, observation, action, reward, next_observation, done, masks, hidden)
            
        prior = abs(td_error.detach().cpu().numpy())
        return prior
        
    def train(self):
        observation, action, reward, next_observation, done, indices, weights, masks, hidden = self.learner.buffer.sample(self.batch_size)
        
        observation = torch.FloatTensor(observation).to(self.device)
        action = torch.LongTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        next_observation = torch.FloatTensor(next_observation).to(self.device)
        done = torch.FloatTensor(done).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)
        masks = torch.FloatTensor(masks).to(self.device)
        hidden = torch.chunk(torch.FloatTensor(hidden).to(self.device).transpose(0,1).squeeze().contiguous(), 2, dim=0)
        
        td_error = self.get_td_error(self.learner.online_net, self.learner.target_net, observation, action, reward, next_observation, done, masks, hidden)
        loss = (td_error.pow(2) * weights.unsqueeze(1)).mean()
        
        self.learner.optimizer.zero_grad()
        loss.backward()
        self.learner.optimizer.step()
        
        self.learner.buffer.update_priorities(indices, abs(td_error.mean(1).detach().cpu().numpy()))



seed = 4
set_seed(seed)


capacity = 100000
actors = 32
batch_size = 64
learning_rate = 1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
alpha = 0.6
beta_start = 0.4
beta_frames = 100000
gamma = 0.99
sequence_length = 80
overlapping_length = 40
burn_in_length = 40
exploration = 5000
update_freq = 800 * actors
push_freq = 400 * actors
eta = 0.9
state_associative_capacity = 384
state_associative_learning_rate = 1e-3
return_alpha = 0.5
return_beta = 0.0

hidden_dim = 128


envs = [gym.make('CartPole-v0').unwrapped for actor in range(actors)]

agent = r2d2(capacity, actors, batch_size, learning_rate, device, alpha, beta_start, beta_frames, gamma,
             sequence_length, overlapping_length, burn_in_length, eta, state_associative_capacity, state_associative_learning_rate,
             return_alpha, return_beta, hidden_dim)


episodes = 5000
count = 0

reward_total = np.zeros(actors)
weight_reward = None
reward_list = []

unixtime_start = int(time.time())
obs = np.array([envs[env].reset() for env in range(actors)])#32,4
hidden = (torch.zeros(1, actors, hidden_dim).to(device), torch.zeros(1, actors, hidden_dim).to(device))

while int(time.time()) - unixtime_start < 240*60:
    action, hidden_new, state_embedding = agent.actor.get_action(obs, hidden, count)
    next_obs, reward, done, _ = zip(*[envs[env].step(int(action[env])) for env in range(actors)])
    next_obs, reward, done = np.array(next_obs), np.array(reward), np.array(done)   #32,4
    reward_total += reward
    
    if count % (125*actors) != 0:
        reward = reward * 0
    else:
        reward = np.log(reward * 125)
        
    if done.sum() > 0:
        reward[done] = 1
        
    agent.state_associative_buffer.store(state_embedding.detach().squeeze(1), reward)
    reward = agent.get_synthetic_return()
    
    prior = agent.get_prior(agent.actor.online_net, agent.actor.target_net, obs, action, reward, next_obs, done, hidden)
    hidden = hidden_new
    
    count += actors
    
    for actor in range(actors):
        agent.actor.buffers[actor].add_global_buffer(
            agent.learner, obs[actor], action[actor][0], reward[actor], next_obs[actor], done[actor], (hidden[0][:, actor:actor+1], hidden[1][:, actor:actor+1]), prior[actor]
        )
        
    obs = next_obs
    if len(agent.learner.buffer.memory) > exploration:
        agent.train()
        
    if count % update_freq == 0:
        agent.learner.update_model()
        
    if count % push_freq == 0:
        agent.learner.push_model(agent.actor)
        
    if done.sum() > 0:
        obs[done] = np.array([envs[env].reset() for env in np.arange(actors)[done]])
        
        hidden[0][0, done] = torch.zeros(done.sum(), hidden_dim).to(device)
        hidden[1][0, done] = torch.zeros(done.sum(), hidden_dim).to(device)
        
        for i in range(done.sum()):
            if not weight_reward:
                weight_reward = reward_total[done][i]
                reward_list.append([time.time() - unixtime_start, weight_reward])
                
            else:
                weight_reward = 0.999 * weight_reward + 0.001 * reward_total[done][i]
                reward_list.append([time.time() - unixtime_start, weight_reward])
                
            agent.actor.buffers[np.arange(actors)[done][i]].sequence_buffer.reset()
            agent.state_associative_buffer.reset(np.arange(actors)[done][i])
            
            print('time: {} \t reward: {} \t weight_reward: {:.3f}'.format(int(time.time()) - unixtime_start, reward_total[done][i], weight_reward))
            reward_total[np.arange(actors)[done][i]] = 0
