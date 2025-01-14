#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 11:30:07 2024

@author: sanup
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# Actor Network using Gumbel-Softmax
class GumbelSoftmaxActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, temperature=1.0):
        super(GumbelSoftmaxActor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.logits = nn.Linear(hidden_dim, action_dim)
        self.temperature = temperature

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        logits = self.logits(x)
        return logits

    def sample_action(self, state, evaluate=False):
        logits = self.forward(state)
        if evaluate:
            action = torch.argmax(logits, dim=-1)
        else:
            # Gumbel-Softmax sampling
            action = F.gumbel_softmax(logits, tau=self.temperature, hard=True)
        return action


# Critic Network
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q_value = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = self.q_value(x)
        return q_value


class SACGumbelAgent:
    def __init__(self, state_dim, action_dim, hidden_dim, actor_lr, critic_lr, alpha_lr, gamma, tau, buffer_size, batch_size, temperature=1.0):
        # Actor Network with Gumbel-Softmax
        self.actor = GumbelSoftmaxActor(state_dim, action_dim, hidden_dim, temperature)
        self.critic1 = Critic(state_dim, action_dim, hidden_dim)
        self.critic2 = Critic(state_dim, action_dim, hidden_dim)
        self.target_critic1 = Critic(state_dim, action_dim, hidden_dim)
        self.target_critic2 = Critic(state_dim, action_dim, hidden_dim)

        # Copy target networks
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=critic_lr)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size

        # Entropy coefficient
        self.log_alpha = torch.tensor(np.log(0.1), requires_grad=True)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)
        self.alpha = self.log_alpha.exp()

        # Hyperparameters
        self.gamma = gamma
        self.tau = tau
        self.temperature = temperature

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor.sample_action(state, evaluate)
        return action.detach().numpy()[0]

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return
    
        # Sample from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
    
        # Convert to torch tensors
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)  # Make sure rewards is a column vector
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)  # Make sure dones is a column vector
    
        # Critic update
        with torch.no_grad():
            next_action_logits = self.actor(next_states)
            next_actions = F.gumbel_softmax(next_action_logits, tau=self.temperature, hard=True)
            target_q1 = self.target_critic1(next_states, next_actions)
            target_q2 = self.target_critic2(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_actions.sum(dim=-1, keepdim=True)
            target_value = rewards + (1 - dones) * self.gamma * target_q
    
        # Update critic networks
        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)
        critic1_loss = F.mse_loss(q1, target_value)  # Ensure q1 and target_value have same shape
        critic2_loss = F.mse_loss(q2, target_value)  # Ensure q2 and target_value have same shape
    
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
    
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
    
        # Actor update
        action_logits = self.actor(states)
        actions_sampled = F.gumbel_softmax(action_logits, tau=self.temperature, hard=True)
        q1 = self.critic1(states, actions_sampled)
        q2 = self.critic2(states, actions_sampled)
        actor_loss = (self.alpha * actions_sampled - torch.min(q1, q2)).mean()
    
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
    
        # Entropy coefficient update
        alpha_loss = -(self.log_alpha * (actions_sampled + self.alpha.detach()).detach()).mean()
    
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp()


    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)



class ReplayBuffer:
    def __init__(self, size):
        self.buffer = []
        self.max_size = size
        self.ptr = 0

    def add(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.max_size:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.ptr] = (state, action, reward, next_state, done)
        self.ptr = (self.ptr + 1) % self.max_size

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

    def __len__(self):
        return len(self.buffer)



import gymnasium as gym

# Initialize environment
env = gym.make('LunarLander-v2', continuous = True, render_mode = 'human')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
hidden_dim = 256

# Hyperparameters
actor_lr = 3e-4
critic_lr = 3e-4
alpha_lr = 3e-4
gamma = 0.99
tau = 0.005
buffer_size = 1000000
batch_size = 256
max_episodes = 500
temperature = 1.0

# Initialize SAC agent with Gumbel-Softmax
agent = SACGumbelAgent(state_dim, action_dim, hidden_dim, actor_lr, critic_lr, alpha_lr, gamma, tau, buffer_size, batch_size, temperature)

# Training loop
for episode in range(max_episodes):
    state, _ = env.reset()
    episode_reward = 0

    while True:
        action = agent.select_action(state)
        next_state, reward, done, _, _ = env.step(action)
        agent.replay_buffer.add(state, action, reward, next_state, done)

        agent.update()
        agent.soft_update(agent.target_critic1, agent.critic1, tau)
        agent.soft_update(agent.target_critic2, agent.critic2, tau)

        state = next_state
        episode_reward += reward

        if done:
            break

    print(f"Episode {episode}, Reward: {episode_reward}")

env.close()






