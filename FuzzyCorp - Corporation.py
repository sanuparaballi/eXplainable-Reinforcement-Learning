#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 12:05:45 2024

@author: sanup
"""


"""
CORPORATION STYLE LCS
"""


# ------------------------------
# Global Covering Pool: Holds covering rules when no rule fires.
# ------------------------------




import gymnasium as gym
import numpy as np
import random
import copy
class CoveringPool:
    def __init__(self):
        self.rules = []

    def add_rule(self, rule):
        self.rules.append(copy.deepcopy(rule))

    def get_rule(self):
        if self.rules:
            return copy.deepcopy(self.rules.pop(0))
        else:
            return None


covering_pool = CoveringPool()

# ------------------------------
# Global Fuzzy Knowledge Base:
# Built once from the environment's observation ranges.
# For each continuous input, creates 'num_sets' TSK membership functions:
# lowest: trapezoidal_lower, highest: trapezoidal_upper, others: triangular.
# ------------------------------


class GlobalFuzzyKnowledgeBase:
    def __init__(self, env, num_sets=7):
        self.num_inputs = env.observation_space.shape[0]
        self.fuzzy_sets = {}
        for i in range(self.num_inputs):
            low = env.observation_space.low[i]
            high = env.observation_space.high[i]
            # If bounds are infinite, use default range.
            if np.isinf(low) or np.isinf(high):
                low, high = -1, 1
            self.fuzzy_sets[i] = []
            centers = np.linspace(low, high, num_sets)
            for idx, center in enumerate(centers):
                if idx == 0:
                    mf_type = "trapezoidal_lower"
                    sigma = centers[1] - center
                elif idx == num_sets - 1:
                    mf_type = "trapezoidal_upper"
                    sigma = center - centers[-2]
                else:
                    mf_type = "triangular"
                    sigma = (centers[idx+1] - centers[idx-1]) / 2.0
                self.fuzzy_sets[i].append(FuzzySet(center, sigma, mf_type))

    def mutate(self, mutation_rate=0.05, mutation_scale=0.05):
        for i in range(self.num_inputs):
            for fs in self.fuzzy_sets[i]:
                if random.random() < mutation_rate:
                    fs.center += np.random.normal(0, mutation_scale)
                    fs.sigma += np.random.normal(0, mutation_scale)
                    fs.sigma = max(fs.sigma, 1e-3)

    def crossover(self, other):
        child = copy.deepcopy(self)
        for i in range(self.num_inputs):
            for j in range(len(self.fuzzy_sets[i])):
                if random.random() < 0.5:
                    child.fuzzy_sets[i][j] = copy.deepcopy(self.fuzzy_sets[i][j])
                else:
                    child.fuzzy_sets[i][j] = copy.deepcopy(other.fuzzy_sets[i][j])
        return child


def create_global_fuzzy_kb(env, num_sets=7):
    return GlobalFuzzyKnowledgeBase(env, num_sets)

# ------------------------------
# Fuzzy Set: TSK–style membership functions.
# ------------------------------


class FuzzySet:
    def __init__(self, center, sigma, mf_type="triangular"):
        self.center = center
        self.sigma = sigma
        self.mf_type = mf_type

    def membership(self, x):
        if self.mf_type == "triangular":
            # Gaussian-like decay for smooth transition.
            # return np.exp(-((x - self.center) ** 2) / (2 * (self.sigma ** 2)))
            return max(0.0, 1 - abs(x - self.center) / self.sigma)
        elif self.mf_type == "trapezoidal_lower":
            if x <= self.center:
                return 1.0
            elif x >= self.center + self.sigma:
                return 0.0
            else:
                return 1 - (x - self.center) / self.sigma
        elif self.mf_type == "trapezoidal_upper":
            if x >= self.center:
                return 1.0
            elif x <= self.center - self.sigma:
                return 0.0
            else:
                return 1 - (self.center - x) / self.sigma
        else:
            return 0.0

    def __str__(self):
        return f"{self.mf_type}(c={self.center:.2f}, s={self.sigma:.2f})"

# ------------------------------
# Fuzzy Rule: TSK–style rule with sequential antecedent indices.
# ------------------------------


class FuzzyRule:
    def __init__(self, num_inputs, action_dim, fuzzy_kb):
        self.num_inputs = num_inputs
        self.action_dim = action_dim
        self.fuzzy_kb = fuzzy_kb
        # Sequentially assign: for input i, use index = i mod (number of fuzzy sets for that input)
        self.indices = [i % len(fuzzy_kb.fuzzy_sets[i]) for i in range(num_inputs)]
        # TSK consequent: linear function (bias + weighted sum)
        self.weights = np.random.uniform(-1, 1, size=(num_inputs + 1, action_dim))
        self.fitness = 0.0

    def compute_membership(self, state):
        memberships = []
        for i, val in enumerate(state):
            idx = self.indices[i]
            fs = self.fuzzy_kb.fuzzy_sets[i][idx]
            memberships.append(fs.membership(val))
        return np.median(memberships)

    def compute_action(self, state):
        return self.weights[0] + np.dot(state, self.weights[1:])

    def mutate(self, mutation_rate=0.1, mutation_scale=0.1):
        for i in range(self.num_inputs):
            if random.random() < mutation_rate:
                n_sets = len(self.fuzzy_kb.fuzzy_sets[i])
                # Shift the index by 1 modulo n_sets.
                self.indices[i] = (self.indices[i] + 1) % n_sets
        if random.random() < mutation_rate:
            self.weights += np.random.normal(0, mutation_scale, self.weights.shape)

    def __str__(self):
        antecedents = []
        for i, idx in enumerate(self.indices):
            antecedents.append(f"x{i}: {self.fuzzy_kb.fuzzy_sets[i][idx]}")
        consequent = ", ".join(f"{w:.2f}" for w in self.weights.flatten())
        return "IF " + " AND ".join(antecedents) + f" THEN action=[{consequent}]"

# ------------------------------
# Corporation: A gene (fuzzy rule set) with covering rule mechanism.
# ------------------------------


class Corporation:
    def __init__(self, global_fuzzy_kb, num_rules, action_dim):
        self.num_inputs = global_fuzzy_kb.num_inputs
        self.action_dim = action_dim
        self.fitness = 0.0
        self.fuzzy_kb = global_fuzzy_kb  # Shared global KB.
        self.rules = [FuzzyRule(self.num_inputs, action_dim, self.fuzzy_kb) for _ in range(num_rules)]

    def compute_activation(self, state):
        # Aggregate activation of the corporation: sum of rule memberships.
        return sum(rule.compute_membership(state) for rule in self.rules)

    def decide_action(self, state, defuzz_method='weighted_average'):
        activated_actions = []
        memberships = []
        for rule in self.rules:
            m = rule.compute_membership(state)
            if m > 1e-3:
                activated_actions.append(rule.compute_action(state))
                memberships.append(m)
        if memberships:
            activated_actions = np.array(activated_actions)
            memberships = np.array(memberships)
            if defuzz_method == 'weighted_average':
                action = np.sum(activated_actions * memberships[:, None], axis=0) / np.sum(memberships)
                return np.clip(action, -1, 1)
            elif defuzz_method == 'mean_of_maxima':
                max_act = np.max(memberships)
                indices = np.where(memberships >= 0.99 * max_act)[0]
                action = np.mean(activated_actions[indices], axis=0)
                return np.clip(action, -1, 1)
            else:
                raise ValueError("Unknown defuzzification method")
        else:
            # Covering rule: if no rule fires, try to inject a covering rule.
            covering_rule = covering_pool.get_rule()
            if covering_rule is not None:
                self.rules.append(covering_rule)
                return covering_rule.compute_action(state)
            else:
                return np.random.uniform(-1, 1, size=(self.action_dim,))

    def mutate(self, mutation_rate=0.1, mutation_scale=0.1):
        for rule in self.rules:
            rule.mutate(mutation_rate, mutation_scale)

    def crossover(self, other):
        child = Corporation.__new__(Corporation)
        child.num_inputs = self.num_inputs
        child.action_dim = self.action_dim
        child.fitness = 0.0
        child.fuzzy_kb = self.fuzzy_kb  # Shared global KB.
        child.rules = []
        for rule_self, rule_other in zip(self.rules, other.rules):
            chosen = copy.deepcopy(rule_self) if random.random() < 0.5 else copy.deepcopy(rule_other)
            chosen.fuzzy_kb = child.fuzzy_kb
            child.rules.append(chosen)
        return child

    def __str__(self):
        s = f"Corporation(fitness={self.fitness:.2f})\n"
        for rule in self.rules:
            s += "  " + str(rule) + "\n"
        return s

# ------------------------------
# Individual: Composed of several corporations (genes).
# For each state, each corporation computes its aggregate activation;
# the corporation with the highest activation is selected to produce the final action.
# ------------------------------
# ------------------------------
# SELECTION FUNCTIONS
# ------------------------------


def tournament_selection(population, k=3):
    """Tournament selection: Pick k individuals, return the best."""
    selected = random.sample(population, k)
    return max(selected, key=lambda ind: ind.fitness)


def roulette_wheel_selection(population):
    """Roulette wheel selection: Probability proportional to fitness."""
    fitnesses = np.array([ind.fitness for ind in population])
    fitnesses -= fitnesses.min()  # Shift to ensure all positive
    if fitnesses.sum() == 0:
        return random.choice(population)  # If all fitnesses are zero, pick randomly
    probabilities = fitnesses / fitnesses.sum()
    return np.random.choice(population, p=probabilities)


def rank_selection(population):
    """Rank-based selection: Higher ranks have higher probability."""
    population.sort(key=lambda ind: ind.fitness)
    ranks = np.arange(1, len(population) + 1)
    probabilities = ranks / ranks.sum()
    return np.random.choice(population, p=probabilities)


class Individual:
    def __init__(self, global_fuzzy_kb, num_corporations, num_rules, action_dim):
        self.corporations = [Corporation(global_fuzzy_kb, num_rules, action_dim)
                             for _ in range(num_corporations)]
        self.fitness = 0.0

    def decide_action(self, state, defuzz_method='weighted_average'):
        activations = [corp.compute_activation(state) for corp in self.corporations]
        best_idx = np.argmax(activations)
        return self.corporations[best_idx].decide_action(state, defuzz_method)

    def mutate(self, mutation_rate=0.1, mutation_scale=0.1):
        for corp in self.corporations:
            corp.mutate(mutation_rate, mutation_scale)

    def crossover(self, other):
        child_corporations = []
        for corp_self, corp_other in zip(self.corporations, other.corporations):
            child_corp = corp_self.crossover(corp_other)
            child_corporations.append(child_corp)
        child = Individual.__new__(Individual)
        child.corporations = child_corporations
        child.fitness = 0.0
        return child

    def __str__(self):
        s = "Individual:\n"
        for idx, corp in enumerate(self.corporations):
            s += f"  Corporation {idx}:\n{corp}\n"
        s += f"Overall Fitness: {self.fitness:.2f}"
        return s

# ------------------------------
# Evaluation and Evolution for Individuals with Elitism.
# ------------------------------


def evaluate_individual(ind, env, episodes=1, max_steps=400, defuzz_method='weighted_average'):
    total_reward = 0.0
    for _ in range(episodes):
        state, _ = env.reset()
        ep_reward = 0.0
        steps = 0
        done = False
        while not done and steps < max_steps:
            action = ind.decide_action(state, defuzz_method)
            state, reward, done, truncated, _ = env.step(action)
            ep_reward += reward
            steps += 1
        total_reward += ep_reward
    ind.fitness = total_reward / episodes
    return ind.fitness


def evolve_individuals(population, mutation_rate=0.1, mutation_scale=0.1, elite_frac=0.2, selection_type="elitism"):
    """Evolves the population using different selection strategies."""
    population.sort(key=lambda ind: ind.fitness, reverse=True)
    elite_count = max(1, int(len(population) * elite_frac))

    # Preserve elites
    new_population = [copy.deepcopy(ind) for ind in population[:elite_count]]

    while len(new_population) < len(population):
        if selection_type == "elitism":
            parent1 = random.choice(population[:elite_count])
            parent2 = random.choice(population[:elite_count])
        elif selection_type == "tournament":
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)
        elif selection_type == "rank":
            parent1 = rank_selection(population)
            parent2 = rank_selection(population)
        elif selection_type == "roulette":
            parent1 = roulette_wheel_selection(population)
            parent2 = roulette_wheel_selection(population)
        else:
            raise ValueError(
                "Invalid selection type. Choose from 'elitism', 'tournament', 'rank', or 'roulette'.")

        # Crossover and mutation
        child = parent1.crossover(parent2)
        child.mutate(mutation_rate, mutation_scale)
        new_population.append(child)

    return new_population


def run_evolution_individuals(num_generations=50, population_size=25, num_corporations=3, num_rules=5,
                              mutation_rate=0.1, mutation_scale=0.1, num_sets=70,
                              defuzz_method='weighted_average', selection_type="elitism"):
    env = gym.make("LunarLanderContinuous-v2")
    action_dim = env.action_space.shape[0]
    global_fuzzy_kb = create_global_fuzzy_kb(env, num_sets)
    population = [Individual(global_fuzzy_kb, num_corporations, num_rules, action_dim)
                  for _ in range(population_size)]

    best_overall = None

    for gen in range(num_generations):
        for ind in population:
            evaluate_individual(ind, env, defuzz_method=defuzz_method)

        population.sort(key=lambda ind: ind.fitness, reverse=True)
        best = population[0]
        if best_overall is None or best.fitness > best_overall.fitness:
            best_overall = copy.deepcopy(best)

        print(f"Generation {gen} Best Individual Fitness: {best.fitness:.2f}")
        population = evolve_individuals(population, mutation_rate,
                                        mutation_scale, selection_type=selection_type)

    env.close()
    print("Best overall individual fitness:", best_overall.fitness)
    print("\nFinal Evolved Individual (Phenotype):")
    print(best_overall)
    return best_overall


def test_final_sequence_agent(agent, episodes=10, max_steps=1000, defuzz_method='weighted_average'):
    env = gym.make("LunarLanderContinuous-v2", render_mode='human')
    total_reward = 0.0
    for ep in range(episodes):
        state, _ = env.reset()
        ep_reward = 0.0
        steps = 0
        while steps < max_steps:
            action = agent.decide_action(state, defuzz_method)
            state, reward, done, truncated, _ = env.step(action)
            ep_reward += reward
            steps += 1
            if done:
                break
        print(f"Test Episode {ep}: Reward = {ep_reward:.2f}")
        total_reward += ep_reward
    env.close()
    avg_reward = total_reward / episodes
    print(f"Final Sequence Agent Average Reward over {episodes} episodes: {avg_reward:.2f}")
    return avg_reward


if __name__ == "__main__":
    print("=== Version 1: Non-Sequential Corporation Evolution with TSK Membership Functions ===")
    selection_type = 'tournament'
    best_corp = run_evolution_individuals(selection_type=selection_type)

    # print("\nFinal Evolved Sequence Agent (Phenotype):")
    # print(best_corp)

    test_final_sequence_agent(best_corp, episodes=10, max_steps=300, defuzz_method='weighted_average')
