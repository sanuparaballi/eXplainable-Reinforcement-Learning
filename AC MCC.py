import gymnasium as gym
import torch
import random
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Normal
import os

# Hyperparameters
HYPERPARAMS = {
    'env_name': 'MountainCarContinuous-v0',
    # 'env_name': 'LunarLander-v2',
    'episodes': 1000,
    'gamma': 0.99,
    'lr': 50e-4,
    'test_only': False,
    # 'test_only': True,
    'save_path': './models',
    # 'model_name': 'LL-NewModel',
    'model_name': 'MCC-NewModel',
}


random_seed = random.randint(1, 100000)


# Simplified Actor-Critic Network
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.dropout1 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(64, 128)
        self.dropout2 = nn.Dropout(p=0.2)

        self.mean_layer = nn.Linear(128, action_dim)
        self.std_layer = nn.Linear(128, action_dim)
        self.value_layer = nn.Linear(128, 1)

    def forward(self, state):
        x = torch.tanh(self.fc1(state))
        x = self.dropout1(x)
        x = torch.tanh(self.fc2(x))
        x = self.dropout2(x)
        mean = self.mean_layer(x)
        mean = torch.tanh(mean)
        std = torch.clamp(self.std_layer(x), -20, 2)  # Clamp log std for numerical stability
        value = self.value_layer(x)
        return mean, std, value

# Function to sample action
def sample_action(mean, log_std):
    std = torch.exp(log_std)
    dist = Normal(mean, std)
    action = dist.sample()
    log_prob = dist.log_prob(action).sum(dim=-1)
    return action.clamp(-1.0, 1.0), log_prob  # Ensure actions are within bounds

# Function to compute returns
def compute_returns(rewards, dones, gamma):
    returns = []
    G = 0
    for reward, done in zip(reversed(rewards), reversed(dones)):
        if done:
            G = 0
        G = reward + gamma * G
        returns.insert(0, np.float32(G))
        
    # returns = torch.tensor(returns)
    # returns = (returns - returns.mean()) / returns.std()
    return returns

# Save model
def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

# Load model
def load_model(model, path):
    model.load_state_dict(torch.load(path))

# Training
def train_actor_critic(hyperparams):
    torch.manual_seed(random_seed)
    env = gym.make(hyperparams['env_name'])
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    model = ActorCritic(state_dim, action_dim)
    if hyperparams['test_only']:
        load_model(model, os.path.join(hyperparams['save_path'], f"{hyperparams['model_name']}_best.pth"))
        env = gym.make(hyperparams['env_name']) #, render_mode = 'human')
        test_model(env, model)
        return

    actor_optimizer = optim.Adam(model.parameters(), lr=hyperparams['lr'])
    critic_optimizer = optim.Adam(model.parameters(), lr=hyperparams['lr'])

    best_reward = -float('inf')
    
    states, actions, log_probs, rewards, dones = [], [], [], [], []
    for episode in range(hyperparams['episodes']):
        state = env.reset()[0]
        
        # states, actions, log_probs, rewards, dones = [], [], [], [], []
        sub_total = []

        # Rollout
        done = False
        step = 0
        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32)
            mean, log_std, value = model(state_tensor)
            action, log_prob = sample_action(mean, log_std)

            next_state, reward, done, _, _ = env.step(action.detach().numpy())

            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            sub_total.append(reward)
            dones.append(done)

            state = next_state
            step += 1
            
            if done:
                break
            
            if step == 1999:
                break
                
        
        
        
        # print(f"Episode: {episode + 1}, Actions: {step}, Reward: {sum(sub_total)}")
        # sub_total.clear()
        
        # if episode % 10 == 0:
        # Compute returns and advantages
        returns = compute_returns(rewards, dones, hyperparams['gamma'])
        returns = torch.tensor(returns, dtype=torch.float32)
        
        values = torch.cat([model(torch.tensor(s, dtype=torch.float32))[2] for s in states]).squeeze()
        critic_loss = nn.MSELoss()(values, returns)
        
        advantages = returns - values.detach()
        log_probs_tensor = torch.stack(log_probs)
        actor_loss = -(log_probs_tensor * advantages).mean()
        
        # loss = (torch.sum(actor_loss + critic_loss) / step)
        loss = (torch.sum(actor_loss + critic_loss))
        # print(loss)

        # Update critic and actor
        critic_optimizer.zero_grad()
        actor_optimizer.zero_grad()
        
        loss.backward()

        critic_optimizer.step()
        actor_optimizer.step()

        total_reward = sum(rewards)
        print(f"**** After learning - Episode: {episode + 1}, Actions: {step}, Reward: {total_reward:.2f}")

        total_reward /= step

        # Save best model
        if total_reward > best_reward:
            best_reward = total_reward
            best_model = model
            save_model(best_model, os.path.join(hyperparams['save_path'], f"{hyperparams['model_name']}_best.pth"))
        
    
        states, actions, log_probs, rewards, dones = [], [], [], [], []


    # Save final model
    save_model(model, os.path.join(hyperparams['save_path'], f"{hyperparams['model_name']}_final.pth"))
    env.close()

# Testing
def test_model(env, model):
    for _ in range(25):
        state = env.reset()[0]
        done = False
        total_reward = 0
        steps = 0
    
        for _ in range(1999):
            state_tensor = torch.tensor(state, dtype=torch.float32)
            mean, std, _ = model(state_tensor)
            action, _ = sample_action(mean, std)
            
            # print(action)
            next_state, reward, done, _, _ = env.step(action.cpu().detach().numpy())
            total_reward += reward
            steps += 1
            state = next_state
            if done:
                break
    
        print(f"Test Reward: {total_reward:.2f}, Total Steps: {steps}")

if __name__ == "__main__":
    train_actor_critic(HYPERPARAMS)
    
    
    # model = ActorCritic(state_dim, action_dim)
    # if hyperparams['test_only']:
    #     load_model(model, os.path.join(hyperparams['save_path'], f"{hyperparams['model_name']}_best.pth"))
    #     env = gym.make(hyperparams['env_name'], continuous=True, render_mode = 'human')
    #     test_model(env, model)
    #     return
    