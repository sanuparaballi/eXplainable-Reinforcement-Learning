
import gymnasium
import numpy as np
import torch
import os
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
# from itertools import product
import matplotlib.pyplot as plt


#########################################
# Section 1: FuzzyKB & Fuzzy Rule System with TSK Memberships and Defuzzification
#########################################

def triangular_membership(x, a, b, c):
    """Standard triangular membership function."""
    if x <= a or x >= c:
        return 0.0
    elif x == b:
        return 1.0
    elif x < b:
        return (x - a) / (b - a)
    else:  # x > b
        return (c - x) / (c - b)


def left_bound_membership(x, b, c):
    """
    Left-bound trapezoidal membership.
    For x <= b, membership is 1; between b and c, it linearly decreases to 0.
    """
    if x <= b:
        return 1.0
    elif b < x <= c:
        return (c - x) / (c - b)
    else:
        return 0.0


def right_bound_membership(x, a, b):
    """
    Right-bound trapezoidal membership.
    For x >= b, membership is 1; between a and b, it linearly increases from 0 to 1.
    """
    if x >= b:
        return 1.0
    elif a <= x < b:
        return (x - a) / (b - a)
    else:
        return 0.0


def linguistic_action_label(action):
    """Convert a numeric action (assumed in [-1, 1]) to a linguistic label."""
    if action <= -0.8:
        return "very low"
    elif action <= -0.6:
        return "low"
    elif action <= -0.3:
        return "medium low"
    elif action <= 0.1:
        return "medium"
    elif action <= 0.4:
        return "medium high"
    elif action <= 0.6:
        return "high"
    else:
        return "very high"


class FuzzyRule:
    def __init__(self, conditions, action):
        """
        conditions: list of tuples (a, b, c, label) for each state dimension.
                    a: left-bound value, b: center, c: right-bound value.
        action: optimal action value (crisp) for the fuzzy region.
        """
        self.conditions = conditions
        self.action = action
        self.action_label = linguistic_action_label(action)

    def membership(self, state):
        """
        Compute overall membership for a given state.
        - If the label is "very low", use left-bound trapezoidal membership.
        - If the label is "very high", use right-bound trapezoidal membership.
        - Otherwise, use triangular membership.
        The overall membership is the minimum membership over all dimensions.
        """
        memberships = []
        for i, cond in enumerate(self.conditions):
            a, b, c, label = cond
            x = state[i]
            if label == "very low":
                m = left_bound_membership(x, b, c)
            elif label == "very high":
                m = right_bound_membership(x, a, b)
            else:
                m = triangular_membership(x, a, b, c)
            memberships.append(m)
        return max(memberships)

    def __str__(self):
        cond_str = " AND ".join([f"x{i} is {cond[3]}" for i, cond in enumerate(self.conditions)])
        return f"IF {cond_str} THEN action is {self.action_label}"


class FuzzyKB:
    def __init__(self):
        self.rules = []

    def add_rule(self, rule):
        self.rules.append(rule)

    def get_optimal_action(self, state):
        """
        For a given state, returns the action of the fuzzy rule with the highest membership.
        (This function was used for explainability evaluation.)
        """
        best_action = None
        best_membership = -1
        for rule in self.rules:
            m = rule.membership(state)
            if m > best_membership:
                best_membership = m
                best_action = rule.action
        return best_action, best_membership

    def defuzzify(self, state):
        """
        Aggregates the output of all fuzzy rules using weighted average (centroid method).
        Returns a crisp action based on the weighted sum of rule actions.
        """
        numerator = 0.0
        denominator = 0.0
        for rule in self.rules:
            m = rule.membership(state)
            numerator += m * rule.action
            denominator += m
        if denominator > 0:
            return numerator / denominator
        else:
            # If no rule fires, default to zero action.
            return 0.0

    def __str__(self):
        return "\n".join([str(rule) for rule in self.rules])


def get_fuzzy_labels(n):
    """
    Returns linguistic labels for n fuzzy sets.
    Options:
      - n==3: ["very low", "medium", "very high"]
      - n==5: ["very low", "low", "medium", "high", "very high"]
      - n==7: ["extremely low", "very low", "low", "medium", "high", "very high", "extremely high"]
    Otherwise, returns generic labels.
    """
    if n == 3:
        return ["very low", "medium", "very high"]
    elif n == 5:
        return ["very low", "low", "medium", "high", "very high"]
    elif n == 7:
        return ["very low", "low", "medium low", "medium", "medium high", "high", "very high"]
    else:
        return [f"set{i}" for i in range(n)]


def process_experiences(experiences, env, num_bins_per_dim=3):
    """
    Processes captured experiences:
      - Filters out experiences with negative rewards.
      - Groups experiences by discretized (binned) state.
      - For each bin, retains the state-action pair with the highest reward.
      - Converts each group into an LCS-style fuzzy rule.
    """
    filtered = [exp for exp in experiences if exp[2] >= 0]

    low = env.observation_space.low
    high = env.observation_space.high
    dim = len(low)

    # Create bins for each state dimension.
    bins = [np.linspace(low[i], high[i], num_bins_per_dim+1) for i in range(dim)]

    best_experience = {}  # key: tuple of bin indices, value: (state, action, reward)
    for state, action, reward, _, _ in filtered:
        indices = []
        for i in range(dim):
            val = np.clip(state[i], low[i], high[i])
            idx = np.digitize(val, bins[i]) - 1
            idx = max(0, min(idx, num_bins_per_dim-1))
            indices.append(idx)
        indices = tuple(indices)
        if indices not in best_experience or reward > best_experience[indices][2]:
            best_experience[indices] = (state, action, reward)

    rules = []
    # Use extended fuzzy labels if desired; here we allow a choice of num_bins_per_dim.
    labels = get_fuzzy_labels(num_bins_per_dim)
    for indices, (state, action, reward) in best_experience.items():
        conditions = []
        for i, idx in enumerate(indices):
            b_vals = bins[i]
            b_center = (b_vals[idx] + b_vals[idx+1]) / 2
            if idx == 0:
                a_val = low[i]
            else:
                prev_center = (b_vals[idx-1] + b_vals[idx]) / 2
                a_val = prev_center
            if idx == num_bins_per_dim - 1:
                c_val = high[i]
            else:
                next_center = (b_vals[idx+1] + b_vals[min(idx+2, len(b_vals)-1)]) / 2
                c_val = next_center
            conditions.append((a_val, b_center, c_val, labels[idx]))
        # rule = FuzzyRule(conditions=conditions, action=action)

        if len(action) > 1:
            for act in action:
                rule = FuzzyRule(conditions=conditions, action=act)
                rules.append(rule)
        else:
            rule = FuzzyRule(conditions=conditions, action=action)
            rules.append(rule)
        # rules.append(rule)
    return rules


def crisp_distance(action_opt, action_nn):
    """Compute the absolute distance between two scalar actions."""
    return np.abs(action_opt - action_nn)


#########################################
# Section 2: SAC Agent Implementation
#########################################

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
    def __init__(self, state_dim, action_dim, hidden_dim=256, actor_lr=3e-3, critic_lr=3e-3,
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


# Training Loop for the SAC Agent
def train_sac_agent(env, agent, num_episodes=50, max_steps=200, save=False):
    print("*********************************")
    print("****** Simulated Training *******")
    print("*********************************")
    total_steps = 0
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        for step in range(max_steps):
            # print(state)
            action = agent.select_action(state)
            # Ensure action shape fits the environment
            action_env = np.array(action)
            next_state, reward, done, trunc, _ = env.step(action_env)
            agent.replay_buffer.push(state, action, reward, next_state, done)
            agent.update()
            state = next_state
            episode_reward += reward
            total_steps += 1
            if done:
                break
        print(f"Episode {episode}: Reward = {episode_reward}")

    if save:

        torch.save(agent, 'model/LLBufferBaseAgent.pkl')
        print(f"Total training steps: {total_steps}")


#########################################
# Section 3: Explainability & Plotting
#########################################
def run_episode_capture_distances(agent, env, fuzzy_kb, max_steps=200):
    """
    Runs one episode using the provided agent and captures the crisp distance at each step.
    """
    state, _ = env.reset()
    distances = []
    memberships = []
    for step in range(max_steps):
        # Use evaluation mode
        action = agent.select_action(state, evaluate=True)
        optimal_action, membership = fuzzy_kb.get_optimal_action(state)
        # If action is an array, take the first element
        agent_action = action[0] if isinstance(action, np.ndarray) else action
        dist = crisp_distance(optimal_action, agent_action)
        distances.append(dist)
        memberships.append(membership)
        next_state, reward, done, trunc, _ = env.step(action)
        state = next_state
        if done:
            break
    return distances, memberships


def compute_statistics(distances):
    arr = np.array(distances)
    return {"mean": np.mean(arr), "std": np.std(arr), "variance": np.var(arr)}


def plot_distances(distances_untrained, distances_trained1, distances_trained2, distances_trained3):
    plt.figure(figsize=(12, 6))
    plt.plot(distances_untrained, label="Untrained Agent", marker='o')
    plt.plot(distances_trained1, label="Trained Agent - 25 Episodes", marker='x')
    plt.plot(distances_trained2, label="Trained Agent - 50 Episodes", marker='*')
    plt.plot(distances_trained3, label="Trained NN Agent - LCS Rules")
    plt.xlabel("Step")
    plt.ylabel("Crisp Distance")
    plt.title("Crisp Distance vs. Steps")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_statistics(stats_agent, stats_untrained, stats_trained):
    categories = ['mean', 'std', 'variance']
    untrained_values = [stats_untrained[cat] for cat in categories]
    trained_values = [stats_trained[cat] for cat in categories]
    agent_values = [stats_agent[cat] for cat in categories]

    x = np.arange(len(categories))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, agent_values, width, label='NN1 Agent')
    plt.bar(x, untrained_values, width, label='NN2 Untrained')
    plt.bar(x + width/2, trained_values, width, label='NN2 Trained')
    plt.xticks(x, categories)
    plt.ylabel("Value")
    plt.title("Statistics of Crisp Distances")
    plt.legend()
    plt.grid(True, axis='y')
    plt.show()


def evaluate_agent(agent, env, fuzzy_kb, max_steps=200, tolerance=0.05):
    """
    Runs one evaluation episode with the agent and measures:
      - Average crisp distance between the agent's action and FuzzyKB's defuzzified action.
      - Correct action ratio: percentage of steps where the distance is below 'tolerance'.
    Returns (avg_distance, correct_ratio).
    """
    state, _ = env.reset()
    distances = []
    correct_steps = 0
    total_steps = 0
    for step in range(max_steps):
        # Get agent action in evaluation mode.
        action = agent.select_action(state, evaluate=True)
        # Get FuzzyKB's recommended action via defuzzification.
        fuzzy_action = fuzzy_kb.defuzzify(state)
        dist = crisp_distance(fuzzy_action, action[0] if isinstance(action, np.ndarray) else action)
        distances.append(dist)
        # print(dist, tolerance)
        if dist < tolerance:
            correct_steps += 1
        total_steps += 1
        next_state, reward, done, _, _ = env.step(action)
        state = next_state
        if done:
            break
    avg_distance = np.mean(distances)
    correct_ratio = correct_steps / total_steps if total_steps > 0 else 0.0
    print(f"Correct Steps: {correct_steps}, total steps: {total_steps}")
    return avg_distance, correct_ratio


def plot_progress(episodes, avg_distances, correct_ratios):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(episodes, avg_distances, marker='o', label="Average Crisp Distance")
    plt.xlabel("Episode")
    plt.ylabel("Average Crisp Distance")
    plt.title("Progress: Average Crisp Distance")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(episodes, correct_ratios, marker='x', color='green', label="Correct Action Ratio")
    plt.xlabel("Episode")
    plt.ylabel("Correct Action Ratio")
    plt.title("Progress: Correct Action Ratio")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


#########################################
# Section 4: Integration & Main Process
#########################################
def main():
    env = gymnasium.make("LunarLander-v2", continuous='True')  # , render_mode='human')
    env_show = gymnasium.make("LunarLander-v2", continuous='True', render_mode='human')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Step 1: Train an agent to build the FuzzyKB.
    if os.path.exists('model/LLBufferBaseAgent45.pkl'):
        agent = torch.load('model/LLBufferBaseAgent.pkl')
        print('****** Agent loaded from file ********')
    else:
        # Initialize and train the SAC agent.
        agent = SACAgent(state_dim, action_dim)
        train_sac_agent(env, agent, num_episodes=50, max_steps=300, save=True)

    # After training, extract experiences from the SAC agent's replay buffer.
    experiences = agent.replay_buffer.buffer
    print(f"\nCaptured {len(experiences)} experiences from SAC training.")

    # Process experiences to generate fuzzy rules (FuzzyKB).
    rules = process_experiences(experiences, env, num_bins_per_dim=7)
    fuzzy_kb = FuzzyKB()
    for rule in rules:
        fuzzy_kb.add_rule(rule)

    print("\n----- Final Generated FuzzyKB Rules -----")
    print(fuzzy_kb)
    print(f"There are {len(fuzzy_kb.rules)} rules in FuzzyKB")

    # Step 2: Create a new untrained SAC agent.
    untrained_agent = SACAgent(state_dim, action_dim)

    # Arrays to record progress metrics.
    eval_episodes = []
    avg_distances = []
    correct_ratios = []

    # Evaluate progress before training.
    avg_dist, correct_ratio = evaluate_agent(untrained_agent, env, fuzzy_kb, max_steps=300, tolerance=0.25)
    eval_episodes.append(0)
    avg_distances.append(avg_dist)
    correct_ratios.append(correct_ratio)
    print(f"Episode 0 (Untrained) - Avg Distance: {avg_dist:.4f}, Correct Ratio: {correct_ratio:.4f}")
    # Capture crisp distances for an episode with the untrained agent.
    distances_untrained, membership = run_episode_capture_distances(
        untrained_agent, env, fuzzy_kb, max_steps=300)
    print("\nDistance captured for the untrained agent.")

    # Step 3: Train the untrained agent.
    print("\nTraining the new agent for explainability evaluation.")
    total_train_episodes = 50
    for episode in range(1, total_train_episodes+1):
        train_sac_agent(env, untrained_agent, num_episodes=1, max_steps=300, save=False)
        avg_dist, correct_ratio = evaluate_agent(
            untrained_agent, env, fuzzy_kb, max_steps=300, tolerance=0.25)
        if episode % 26 == 0:
            # Capture crisp distances for an episode with the trained agent.
            distances_trained1, membership1 = run_episode_capture_distances(
                untrained_agent, env, fuzzy_kb, max_steps=300)
        eval_episodes.append(episode)
        avg_distances.append(avg_dist)
        correct_ratios.append(correct_ratio)
        print(f"Episode {episode} - Avg Distance: {avg_dist:.4f}, Correct Ratio: {correct_ratio:.4f}")

    # Capture crisp distances for an episode with the trained agent.
    distances_trained2, memberships2 = run_episode_capture_distances(
        untrained_agent, env_show, fuzzy_kb, max_steps=300)

    # Capture crisp distances for an episode with the trained agent.
    distances_agent, memberships_agent = run_episode_capture_distances(
        agent, env_show, fuzzy_kb, max_steps=300)
    # print(memberships3)

    print("\nDistance captured for the trained agent.")

    # Step 4: Plot the crisp distance vs steps.
    plot_distances(distances_untrained, distances_trained1, distances_trained2, distances_agent)

    # Step 5: Compute and plot statistics (mean, std, variance) for both episodes.
    stats_untrained = compute_statistics(distances_untrained)
    stats_trained = compute_statistics(distances_trained2)
    stats_membership = compute_statistics(membership)
    stats_membership1 = compute_statistics(membership1)
    stats_membership2 = compute_statistics(memberships2)

    stats_agent = compute_statistics(distances_agent)
    stats_membership_agent = compute_statistics(memberships_agent)

    print("\nUntrained Agent Statistics:", stats_untrained)
    print("Trained Agent Statistics:", stats_trained)

    print("\nNN1 Agent Distance Statistics:", stats_agent)
    print("NN1 Agent LCS Membership Statistics:", stats_membership_agent)

    print("\nNN2 Agent Untrained LCS Membership Statistics:", stats_membership)
    print("NN2 Agent Midtraining LCS Membership Statistics:", stats_membership1)
    print("NN2 Agent Fully trained LCS Membership Statistics:", stats_membership2)
    plot_statistics(stats_agent, stats_untrained, stats_trained)

    plot_progress(eval_episodes, avg_distances, correct_ratios)


if __name__ == "__main__":
    main()
