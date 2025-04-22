#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 21:56:48 2025

@author: sanup
"""


import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# ================================
# Fuzzy Rule Agent Components
# ================================

# Membership functions


def triangular(x, a, b, c):
    """Triangular membership function."""
    return np.maximum(0, np.minimum((x - a) / (b - a + 1e-6), (c - x) / (c - b + 1e-6)))


def trapezoidal(x, a, b, c, d):
    """Trapezoidal membership function."""
    return np.maximum(0, np.minimum(np.minimum((x - a)/(b - a + 1e-6), 1), (d - x)/(d - c + 1e-6)))

# Compute fuzzy set distance using numerical integration


def fuzzy_set_distance(mu1, mu2, x):
    return np.trapz((mu1 - mu2)**2, x)

# Adaptive aggregation: weighted sum for antecedents or consequents


def adaptive_aggregation(distances, weights=None):
    distances = np.array(distances)
    if weights is None:
        weights = np.ones_like(distances) / len(distances)
    else:
        weights = np.array(weights) / np.sum(weights)
    return np.sum(weights * distances)


def rule_distance(rule1, rule2, x, alpha_weights=None, beta_weights=None, gamma=0.5):
    """
    Compute overall distance between two fuzzy rules.
    Each rule is represented as a dictionary with 'antecedents' and 'consequents'.
    """
    antecedent_distances = []
    consequent_distances = []
    # Antecedents
    for params1, params2 in zip(rule1['antecedents'], rule2['antecedents']):
        func1, p1 = params1
        func2, p2 = params2
        mu1 = func1(x, *p1)
        mu2 = func2(x, *p2)
        d = fuzzy_set_distance(mu1, mu2, x)
        antecedent_distances.append(d)
    # Consequents
    for params1, params2 in zip(rule1['consequents'], rule2['consequents']):
        func1, p1 = params1
        func2, p2 = params2
        mu1 = func1(x, *p1)
        mu2 = func2(x, *p2)
        d = fuzzy_set_distance(mu1, mu2, x)
        consequent_distances.append(d)
    d_ant = adaptive_aggregation(antecedent_distances, alpha_weights)
    d_cons = adaptive_aggregation(consequent_distances, beta_weights)
    return gamma * d_ant + (1 - gamma) * d_cons, d_ant, d_cons


# Fuzzy rule definitions (using state indices: altitude=state[1], vertical velocity=state[3])
x_domain = np.linspace(0, 10, 1000)
rule1 = {
    'antecedents': [
        (trapezoidal, (0, 0, 4, 6)),            # Altitude: "Low"
        (triangular, (5, 7, 9))                  # Vertical velocity: "High"
    ],
    'consequents': [
        (triangular, (0, 0.5, 1))                # Thrust: moderate (peak at 0.5)
    ]
}
rule2 = {
    'antecedents': [
        (triangular, (2, 5, 8)),                 # Altitude: "Medium"
        (triangular, (2, 4, 6))                  # Vertical velocity: "Medium"
    ],
    'consequents': [
        (triangular, (0.5, 0.75, 1))             # Thrust: high (peak at 0.75)
    ]
}
alpha_weights = [0.6, 0.4]  # More weight to altitude
beta_weights = [1.0]

total_rule_dist, d_ant, d_cons = rule_distance(
    rule1, rule2, x_domain, alpha_weights, beta_weights, gamma=0.5)
print("Fuzzy Rule Distance between Rule 1 and Rule 2: {:.4f}".format(total_rule_dist))


def fuzzy_inference(state, rule_set):
    """
    Compute control action (main engine thrust) using fuzzy rules.
    Uses state[1] (altitude) and state[3] (vertical velocity). 
    Returns a thrust value in [0,1]. 
    """
    altitude = np.clip(state[1], 0, 10)
    y_vel = np.clip(state[3] + 5, 0, 10)  # shift vertical velocity to [0,10]
    activations = []
    for rule in rule_set:
        alt_mu = rule['antecedents'][0][0](np.array([altitude]), *rule['antecedents'][0][1])[0]
        vel_mu = rule['antecedents'][1][0](np.array([y_vel]), *rule['antecedents'][1][1])[0]
        activation = min(alt_mu, vel_mu)
        activations.append(activation)
    outputs = []
    for rule in rule_set:
        # Use peak value of the triangular consequent as the output
        outputs.append(rule['consequents'][0][1][1])
    if sum(activations) > 0:
        action = np.dot(activations, outputs) / sum(activations)
    else:
        action = 0.0
    return np.clip(action, 0, 1)


def run_fuzzy_agent(episodes=10, render=False):
    env = gym.make("LunarLanderContinuous-v2")
    rewards = []
    fuzzy_rules = [rule1, rule2]
    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        while not done:
            if render:
                env.render()
            thrust = fuzzy_inference(state, fuzzy_rules)
            action = np.array([thrust, 0.0])
            state, reward, done, trun, info = env.step(action)
            total_reward += reward
        rewards.append(total_reward)
        print("Fuzzy Agent Episode {}: Reward = {:.2f}".format(ep+1, total_reward))
    env.close()
    avg_reward = np.mean(rewards)
    print("Fuzzy Agent Average Reward over {} episodes: {:.2f}".format(episodes, avg_reward))
    return avg_reward

# ================================
# PPO Agent using PyTorch
# ================================


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=64):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        # Actor head: outputs mean for each action dimension
        self.actor_mean = nn.Linear(hidden_size, action_dim)
        # Log std parameter (learned as a parameter vector)
        self.actor_logstd = nn.Parameter(torch.zeros(action_dim))
        # Critic head: outputs state value
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.actor_mean(x)
        std = self.actor_logstd.exp().expand_as(mean)
        value = self.critic(x)
        return mean, std, value


class PPOAgent:
    def __init__(self, state_dim, action_dim, hidden_size=64, lr=3e-4, clip_param=0.2, max_grad_norm=0.5):
        self.policy = ActorCritic(state_dim, action_dim, hidden_size)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.clip_param = clip_param
        self.max_grad_norm = max_grad_norm

    def select_action(self, state):
        # print(state)
        state = torch.FloatTensor(state).unsqueeze(0)
        mean, std, _ = self.policy(state)
        normal = torch.distributions.Normal(mean, std)
        action = normal.sample()
        action_log_prob = normal.log_prob(action).sum(dim=-1)
        return action.detach().numpy()[0], action_log_prob.detach(), normal

    def evaluate_actions(self, states, actions):
        mean, std, values = self.policy(states)
        normal = torch.distributions.Normal(mean, std)
        action_log_probs = normal.log_prob(actions).sum(dim=-1)
        dist_entropy = normal.entropy().sum(dim=-1)
        return action_log_probs, torch.squeeze(values), dist_entropy

    def update(self, trajectories, ppo_epochs=4, batch_size=64, gamma=0.99, lam=0.95):
        states = torch.FloatTensor(np.vstack([t['state'] for t in trajectories]))
        actions = torch.FloatTensor(np.vstack([t['action'] for t in trajectories]))
        rewards = [t['reward'] for t in trajectories]
        dones = [t['done'] for t in trajectories]
        old_action_log_probs = torch.FloatTensor(np.vstack([t['log_prob'] for t in trajectories])).squeeze()

        # Compute returns and advantages
        returns = []
        advantages = []
        R = 0
        A = 0
        values = self.policy(states)[2].detach().squeeze().numpy()
        for i in reversed(range(len(rewards))):
            mask = 0 if dones[i] else 1
            R = rewards[i] + gamma * R * mask
            delta = rewards[i] + gamma * values[i+1] * mask if i < len(rewards)-1 else rewards[i] - values[i]
            A = delta + gamma * lam * A * mask
            returns.insert(0, R)
            advantages.insert(0, A)
        returns = torch.FloatTensor(returns)
        advantages = torch.FloatTensor(advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

        # PPO update
        dataset_size = states.size(0)
        for epoch in range(ppo_epochs):
            indices = np.arange(dataset_size)
            np.random.shuffle(indices)
            for start in range(0, dataset_size, batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_action_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]

                log_probs, values_pred, dist_entropy = self.evaluate_actions(batch_states, batch_actions)
                ratio = (log_probs - batch_old_log_probs).exp()
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * batch_advantages
                action_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values_pred, batch_returns)
                loss = action_loss + 0.5 * value_loss - 0.01 * dist_entropy.mean()

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

# Training loop for the PPO Agent


def train_ppo_agent(total_episodes=300, update_interval=2048):
    env = gym.make("LunarLanderContinuous-v2")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = PPOAgent(state_dim, action_dim, hidden_size=64)

    trajectories = []
    episode_rewards = []
    state, _ = env.reset()
    total_steps = 0
    for episode in range(total_episodes):
        ep_reward = 0
        done = False
        while not done:
            # print(total_steps)
            action, log_prob, _ = agent.select_action(state)
            next_state, reward, done, truc, info = env.step(action)
            trajectories.append({
                'state': state,
                'action': action,
                'reward': reward,
                'done': done,
                'log_prob': log_prob.item()
            })
            state = next_state
            ep_reward += reward
            total_steps += 1
            # Update policy after collecting update_interval steps
            if total_steps % update_interval == 0:
                agent.update(trajectories)
                trajectories = []
        episode_rewards.append(ep_reward)
        state = env.reset()
        if (episode+1) % 20 == 0:
            avg_rw = np.mean(episode_rewards[-20:])
            print("PPO Episode {}: Average Reward = {:.2f}".format(episode+1, avg_rw))
    env.close()
    avg_reward = np.mean(episode_rewards[-20:])
    print("PPO Agent Average Reward (last 20 episodes): {:.2f}".format(avg_reward))
    return agent, avg_reward


def evaluate_ppo_agent(agent, episodes=10):
    env = gym.make("LunarLanderContinuous-v2")
    rewards = []
    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        while not done:
            action, _, _ = agent.select_action(state)
            state, reward, term, trun, info = env.step(action)
            total_reward += reward
        rewards.append(total_reward)
        print("PPO Evaluation Episode {}: Reward = {:.2f}".format(ep+1, total_reward))
    env.close()
    avg_reward = np.mean(rewards)
    print("PPO Agent Average Evaluation Reward: {:.2f}".format(avg_reward))
    return avg_reward


# ================================
# Comparison and Visualization
# ================================
if __name__ == "__main__":
    # Evaluate Fuzzy Rule Agent
    print("\n--- Evaluating Fuzzy Rule Agent ---")
    fuzzy_avg_reward = run_fuzzy_agent(episodes=25, render=False)

    # Train and evaluate the PPO agent using PyTorch
    print("\n--- Training PPO Agent (PyTorch) ---")
    ppo_agent, ppo_train_reward = train_ppo_agent(total_episodes=300, update_interval=2048)
    print("\n--- Evaluating PPO Agent ---")
    ppo_avg_reward = evaluate_ppo_agent(ppo_agent, episodes=10)

    # Plot a bar chart to compare performance
    agents = ['Fuzzy Agent', 'PPO Agent (PyTorch)']
    rewards_compare = [fuzzy_avg_reward, ppo_avg_reward]
    plt.bar(agents, rewards_compare, color=['skyblue', 'salmon'])
    plt.ylabel("Average Reward")
    plt.title("Agent Performance on LunarLanderContinuous-v2")
    plt.show()

    # Plot membership functions for fuzzy rules for interpretability
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    axs[0].plot(x_domain, trapezoidal(x_domain, *rule1['antecedents'][0][1]), label="Rule1 Altitude 'Low'")
    axs[0].plot(x_domain, triangular(x_domain, *rule2['antecedents'][0][1]), label="Rule2 Altitude 'Medium'")
    axs[0].set_title("Altitude Membership Functions")
    axs[0].legend()

    axs[1].plot(x_domain, triangular(x_domain, *rule1['antecedents'][1][1]), label="Rule1 Vel 'High'")
    axs[1].plot(x_domain, triangular(x_domain, *rule2['antecedents'][1][1]), label="Rule2 Vel 'Medium'")
    axs[1].set_title("Vertical Velocity Membership Functions")
    axs[1].legend()
    plt.tight_layout()
    plt.show()
