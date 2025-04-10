#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 16:09:30 2024

@author: sanup
"""


import torch
import os
import gc
import random
import gymnasium
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.distributions import Normal


lr = 10e-3
numIters = 500
MAX_ACTIONS = 300
SHOW_TRAINING = True
ONLY_TEST = False
CAPTURE_STATE = True
# PROBLEM = 'BipedalWalker-v3'
PROBLEM = 'LunarLander-v2'
# PROBLEM = 'Acrobot-v1'
# PROBLEM = 'MountainCarContinuous-v0'
# render = False
gamma = .99
betas = (0.9, 0.999)
random_seed = random.randint(1, 10000)
RIGHTACTIONS = []


# filepath = "model/SPIEModels/BWalkerModel.pkl"
# filepath_best = "model/SPIEModels/BWalkerModelBest.pkl"

filepath = "model/SPIEModels/LLModel2.pkl"
filepath_best = "model/SPIEModels/LLModelBest2.pkl"

# filepath = "model/SPIEModels/AcrobatModel.pkl"
# filepath_best = "model/SPIEModels/AcrobatModelBest.pkl"

# filepath = "model/SPIEModels/MCCModel.pkl"
# filepath_best = "model/SPIEModels/MCCModelBest.pkl"


ruleante = ['verylow', 'low', 'mediumlow', 'medium', 'mediumhigh', 'high', 'veryhigh']
rulebool = ['notlanded', 'landed']

# device = "mps" if torch.backends.mps.is_available() else "cpu"              # Is not really speeding it up!
device = "cpu"


# LOG_SIG_MAX = 2
# LOG_SIG_MIN = -20
# epsilon = 1e-6

# # Initialize Policy weights
# def weights_init_(m):
#     if isinstance(m, nn.Linear):
#         torch.nn.init.xavier_uniform_(m.weight, gain=1)
#         torch.nn.init.constant_(m.bias, 0)

# class ActorCritic(nn.Module):
#     def __init__(self, state_dim, action_dim, action_space=None):
#         super(ActorCritic, self).__init__()
#         self.layer1 = nn.Linear(state_dim, 32).to(device)
#         # self.dropout1 = nn.Dropout(p=0.2)
#         self.layer2 = nn.Linear(32, 128).to(device)
#         # self.dropout2 = nn.Dropout(p=0.2)

#         # self.layer3 = nn.Linear(state_dim + action_dim, 32).to(device)
#         # self.layer4 = nn.Linear(32, 128).to(device)

#         # self.mean_linear = nn.Linear(128, action_dim)
#         # self.log_std_linear = nn.Linear(128, action_dim)

#         self.action_layer = nn.Linear(128, action_dim).to(device)
#         # self.softmax = nn.Softmax(dim=-1)
#         self.value_layer = nn.Linear(128, 1).to(device)
#         self.std_layer = nn.Linear(128, action_dim).to(device)

#         self.logprobs = []
#         self.state_values = []
#         self.rewards = []

#     def forward(self, state):
#         state = torch.from_numpy(state).float().to(device)
#         outputA = F.relu(self.layer1(state)).to(device)
#         # outputA = self.dropout1(outputA)
#         outputA = F.relu(self.layer2(outputA)).to(device)
#         # outputA = self.dropout2(outputA)
#         # outputB = self.softmax(outputA)

#         # outputC = F.relu(self.layer1(state)).to(device)
#         # outputC = F.relu(self.layer2(outputC)).to(device)

#         action_probs = self.action_layer(outputA).to(device)

#         action_probs = torch.tanh(action_probs).to(device)

#         std = torch.clamp(self.std_layer(outputA).to(device), -1, 1)
#         std = torch.exp(std)
#         # std = torch.tensor([1]).expand_as(action_probs).to(device)      # Taking an arbitrary STD and expanding it to match the action space
#         action_distribution = Normal(action_probs, std)

#         action = action_distribution.sample()

#         # outputC = F.relu(self.layer3(torch.cat((state, action))).to(device))
#         # outputC = F.relu(self.layer4(outputC)).to(device)

#         state_value = self.value_layer(outputA).to(device)

#         self.logprobs.append(action_distribution.log_prob(action))
#         self.state_values.append(state_value)

#         return action.clamp(-1., 1.)

#     def calculateLoss(self, gamma=0.99):
#         rewards = []
#         dis_reward = 0
#         for reward in self.rewards[::-1]:
#             dis_reward = reward + gamma * dis_reward
#             rewards.insert(0, np.float32(dis_reward))

#         rewards = torch.tensor(rewards).to(device)
#         rewards = (rewards - rewards.mean()) / rewards.std()
#         # rewards = rewards / len(rewards)

#         loss = 0
#         for logprob, value, reward in zip(self.logprobs, self.state_values, rewards):
#             advantage = reward - value.item()
#             action_loss = -logprob * advantage
#             reward = torch.tensor([reward]).to(device)
#             value_loss = F.smooth_l1_loss(value, reward)
#             # print(reward)

#             loss += torch.sum(action_loss + value_loss)

#         return loss

#     def clearMemory(self):
#         self.logprobs.clear()
#         self.state_values.clear()
#         self.rewards.clear()

#     def optimize(self, optimizer, scheduler):
#         # for _ in range(5):
#         # print(_)
#         optimizer.zero_grad()

#         loss = self.calculateLoss(gamma)
#         loss.backward()

#         optimizer.step()
#         scheduler.step()

#         self.clearMemory()
#         gc.collect()


# Replay Buffer for Experience Storage
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

# Actor Network (Policy)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, log_std_min=-20, log_std_max=2):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick
        action = torch.tanh(x_t)  # Squash to [-1, 1]
        log_prob = normal.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob

# Critic Network (Q-Function)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q

# SAC Agent


class SACAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=256, actor_lr=3e-4, critic_lr=3e-4,
                 gamma=0.99, tau=0.005, alpha=0.2, buffer_capacity=1000000, batch_size=64):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.batch_size = batch_size
        self.device = torch.device("mps" if torch.cuda.is_available() else "cpu")

        # Actor network and optimizer
        self.actor = Actor(state_dim, action_dim, hidden_dim).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)

        # Critic networks and optimizer (using two critics for stability)
        self.critic1 = Critic(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic2 = Critic(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_optimizer = optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=critic_lr
        )

        # Target networks for the critics
        self.target_critic1 = Critic(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_critic2 = Critic(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        # Replay Buffer
        self.replay_buffer = ReplayBuffer(buffer_capacity)

    def select_action(self, state, evaluate=False):
        # print(state)
        state = torch.FloatTensor(state)
        state = state.unsqueeze(0).to(self.device)
        if evaluate:
            with torch.no_grad():
                mean, _ = self.actor(state)
                action = torch.tanh(mean)
                return action.cpu().numpy()[0]
        else:
            with torch.no_grad():
                action, _ = self.actor.sample(state)
                return action.cpu().numpy()[0]

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        transitions = self.replay_buffer.sample(self.batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        state = torch.FloatTensor(np.array(state)).to(self.device)
        action = torch.FloatTensor(np.array(action)).to(self.device)
        reward = torch.FloatTensor(np.array(reward)).unsqueeze(1).to(self.device)
        next_state = torch.FloatTensor(np.array(next_state)).to(self.device)
        done = torch.FloatTensor(np.array(done)).unsqueeze(1).to(self.device)

        # Compute target Q-value using target critics
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_state)
            target_q1 = self.target_critic1(next_state, next_action)
            target_q2 = self.target_critic2(next_state, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            target_value = reward + (1 - done) * self.gamma * target_q

        # Update critics
        current_q1 = self.critic1(state, action)
        current_q2 = self.critic2(state, action)
        critic_loss = F.mse_loss(current_q1, target_value) + F.mse_loss(current_q2, target_value)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor
        new_action, log_prob = self.actor.sample(state)
        q1_new = self.critic1(state, new_action)
        q2_new = self.critic2(state, new_action)
        actor_loss = (self.alpha * log_prob - torch.min(q1_new, q2_new)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update target networks
        for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
        for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)


def train():

    torch.manual_seed(random_seed)
    envS = gymnasium.make(PROBLEM, continuous=True, render_mode='human')
    envN = gymnasium.make(PROBLEM, continuous=True)

    ####
    rulemodel = torch.load('model/SPIEModels/LLCapturedState.pkl')
    fuzzyKB = generate_fuzzy_kb(envN)

    # print(envN.observation_space.shape[0], envN.action_space.shape[0])
    policy = SACAgent(envN.observation_space.shape[0], envN.action_space.shape[0])
    # policy = ActorCritic(envN.observation_space.shape[0], envN.action_space.n)
    # optimizer = optim.Adam(policy.parameters(), lr=lr, betas=betas)
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    best_reward, best_epoch = 0, 0
    running_reward = 0
    early_stop_reward = []
    # randomize = False

    for epoch in range(numIters):
        if SHOW_TRAINING and ((epoch+1) % 200 == 0):
            env = envS
        else:
            env = envN

        state, _ = env.reset()
        count = 0
        right_action = 0

        for i in range(MAX_ACTIONS):
            # if count > epoch:
            #     action = torch.tensor(env.action_space.sample())
            # else:
            # _, _, action = policy.sample(state)
            action = policy.select_action(state)
            # print(action)
            state, reward, done, trunc, _ = env.step(np.array(action))
            # policy.rewards.append(reward)
            count += 1

            if CAPTURE_STATE:
                inputrule = list(state)
                inputrule.extend(np.array(action))
                # print(inputrule)
                captureintervals = []
                for idx, item in enumerate(inputrule):
                    if idx in [6, 7]:
                        for index, interval in enumerate(fuzzyKB[idx]):
                            if interval[0] <= item <= interval[1]:
                                captureintervals.append(rulebool[index])
                                break
                            else:
                                continue
                    else:
                        for index, interval in enumerate(fuzzyKB[idx]):
                            if interval[0] <= item <= interval[1]:
                                captureintervals.append(ruleante[index])
                                break
                            else:
                                continue

                # print(captureintervals)
                # outputrule.add(tuple(captureintervals))

                if (tuple(captureintervals)) in rulemodel:
                    # print("**Right Action")
                    right_action += 1
                # else:
                    # print("Wrong Action")

            if done:
                break

        # if (epoch+1) % 10 == 0:
        RIGHTACTIONS.append(right_action/3)  # Convert to percentage.
        # running_reward = sum(policy.rewards)
        # early_stop_reward.append(running_reward)

        # if ((epoch+1) % 1000) == 0:
        # if early_stop_reward[-1] == early_stop_reward[-50]:
        #     print('Stopping early since Model is not learning.')
        #     print("Last action score and running reward respectively for iteration {}: {} and {}".format(epoch+1, count, running_reward))
        #     break

        # Stupid attempt #### Didn't work
        # print('Early stop check: ', sum(early_stop_reward[-50::1]))
        # if sum(early_stop_reward[-50::1]) < 0 and randomize == False:
        #     policy.apply(policy._init_weights)
        #     randomize = True
        #     print('Model parameters randomized.')

        # print(epoch)
        if (epoch+1) % 200 == 0:
            print("Number of actions taken in this episode and running reward respectively for iteration {}: {} and {}".format(
                epoch+1, count, running_reward))

        # if running_reward > best_reward:
        #     best_reward = running_reward
        #     best_epoch = epoch + 1
        #     best_model = policy

        policy.replay_buffer.push(state, action, reward, state, done)
        policy.update()

        # running_reward = 0

        # if (epoch+1) % 500 == 0 and SHOW_TRAINING:
        #     env.render()

    torch.save(policy, filepath)

    # print("The best running reward in this training session is {} in the epoch {}.".format(best_reward, best_epoch))
    # torch.save(best_model, filepath_best)

    env.close()

    x = np.arange(1, 501)
    y = RIGHTACTIONS
    # Create the dot graph
    plt.figure(figsize=(10, 6))
    # Scatter plot for dot graph
    plt.scatter(x, y, s=10, alpha=0.7, c='blue', label='Percentage of Right Decisions Made per Episode')
    plt.title("Dot Graph of Number of Correct Decisions Taken Per Episode of Training")
    plt.xlabel("Number of Episodes")
    plt.ylabel("Percentage")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # Show the plot
    plt.show()


def generate_fuzzy_kb(env):
    """Generates a fuzzy knowledge base (KB) for rule antecedents and consequents."""
    fuzzy_kb = []
    rule_length = env.observation_space.shape[0] + env.action_space.shape[0]
    for idx in range(rule_length):
        fuzzy_kb.append(create_fuzzy_intervals(idx, env))
    return fuzzy_kb


def create_fuzzy_intervals(idx, env):
    """Creates fuzzy intervals based on the index."""

    kb_list = []

    if idx < (env.observation_space.shape[0]):
        featureRange = abs(env.observation_space.low[idx]) + abs(env.observation_space.high[idx])
        intervalGap = featureRange / len(ruleante)

        lower = env.observation_space.low[idx]
    else:
        idx = idx - env.observation_space.shape[0]
        featureRange = abs(env.action_space.low[idx]) + abs(env.action_space.high[idx])
        intervalGap = featureRange / len(ruleante)

        lower = env.action_space.low[idx]

    for i in range(len(ruleante)):
        kb_list.append((lower, lower + intervalGap))
        lower = lower + intervalGap

    return kb_list

    # if idx in [0, 1]:
    #     return [(-1.5, -1.), (-1.1, -0.5), (-0.4, 0.), (-0.1, 0.5), (0.4, 0.8), (0.8, 1.2), (1.1, 1.5)]
    # elif idx in [2, 3, 5]:
    #     return [(-5., -3.5), (-4., -2.), (-3., -1.), (-1.5, 1.), (0.5, 2.5), (2., 4.), (3.5, 5.)]
    # elif idx == 4:
    #     return [(-3.14, -2.), (-2., -0.5), (-1., 0.), (-0.2, 1.), (0.5, 1.5), (1.3, 2.5), (2.3, 3.14)]
    # elif idx in [6, 7]:
    #     return [(0., 0.), (1., 1.)]
    # elif idx in [8, 9]:
    #     return [(-1, -0.8), (-0.85, -0.65), (-0.7, -0.5), (-0.5, 0.5), (0.5, 0.7), (0.65, 0.85), (0.8, 1.)]


if __name__ == '__main__':

    print("***************************")
    print(f"***** {PROBLEM} ******")

    if ONLY_TEST != True:
        # Training the trained model
        print("***************************")
        print("***** Training Model ******")
        print("***************************")
        train()

    # Test the trained model
    print("***************************")
    print("****** Testing Model ******")
    print("***************************")

    # if os.path.exists('model/LLStableHover.pkl'):
    #     actor = torch.load('model/LLStableHover.pkl')
    # if os.path.exists('model/LLCheck.pkl'):
    #     actor = torch.load('model/LLCheck.pkl')
    if os.path.exists(filepath_best):
        actor = torch.load(filepath_best)

        outputrule = set()

        env = gymnasium.make(PROBLEM, continuous=True, render_mode='human')
        fuzzyKB = generate_fuzzy_kb(env)

        for i in range(50):
            print(f"Test iteration {i}")
            state, _ = env.reset()

            for _ in range(MAX_ACTIONS):
                # print(state)
                # action, _, _ = actor.sample(state)
                action = actor(state)
                # print(action)
                state, reward, done, trunc, _ = env.step(action.cpu().detach().numpy())

                if CAPTURE_STATE:
                    inputrule = list(state)
                    inputrule.extend(action.cpu().detach().numpy())
                    # print(inputrule)
                    captureintervals = []
                    for idx, item in enumerate(inputrule):
                        if idx in [6, 7]:
                            for index, interval in enumerate(fuzzyKB[idx]):
                                if interval[0] <= item <= interval[1]:
                                    captureintervals.append(rulebool[index])
                                    break
                                else:
                                    continue
                        else:
                            for index, interval in enumerate(fuzzyKB[idx]):
                                if interval[0] <= item <= interval[1]:
                                    captureintervals.append(ruleante[index])
                                    break
                                else:
                                    continue

                    # print(captureintervals)
                    outputrule.add(tuple(captureintervals))

                if done:
                    break

            env.render()

        # for rule in (sorted(outputrule, reverse=True)):
        #     print(rule)

        print('Length of Phenotypes: ', len(outputrule))
        # outputrule = set(outputrule)
        # outputrule = np.unique(outputrule).tolist()
        torch.save(outputrule, 'model/SPIEModels/LLCapturedState3_poster.pkl')

        env.reset()
        env.close()

    else:
        print("Couldn't find the file to load.")
