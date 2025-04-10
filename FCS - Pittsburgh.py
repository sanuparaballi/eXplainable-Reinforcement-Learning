#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  8 17:29:15 2025

@author: sanup
"""

import gymnasium as gym
import numpy as np
import random
import copy

# ------------------------------
# Fuzzy Set, Knowledge Base, and Fuzzy Rule Definitions
# ------------------------------
class FuzzySet:
    def __init__(self, center, sigma, mf_type="triangular"):
        self.center = center
        self.sigma = sigma
        self.mf_type = mf_type  # For simplicity, assume continuous only here.

    def membership(self, x):
        if self.mf_type == "triangular":
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

class FuzzyKnowledgeBase:
    def __init__(self, num_inputs, num_sets=7):
        self.num_inputs = num_inputs
        self.fuzzy_sets = {}
        for i in range(num_inputs):
            # For simplicity, assume all inputs are continuous.
            self.fuzzy_sets[i] = []
            for idx in range(num_sets):
                center = np.random.uniform(-1, 1)
                sigma = np.random.uniform(0.1, 1.0)
                if idx == 0:
                    mf_type = "trapezoidal_lower"
                elif idx == num_sets - 1:
                    mf_type = "trapezoidal_upper"
                else:
                    mf_type = "triangular"
                self.fuzzy_sets[i].append(FuzzySet(center, sigma, mf_type))
    def mutate(self, mutation_rate=0.1, mutation_scale=0.1):
        for i in range(self.num_inputs):
            for fs in self.fuzzy_sets[i]:
                if random.random() < mutation_rate:
                    fs.center += np.random.normal(0, mutation_scale)
                    fs.sigma += np.random.normal(0, mutation_scale)
                    fs.sigma = max(fs.sigma, 0.05)
    def crossover(self, other):
        child = FuzzyKnowledgeBase(self.num_inputs)
        for i in range(self.num_inputs):
            child.fuzzy_sets[i] = []
            for fs1, fs2 in zip(self.fuzzy_sets[i], other.fuzzy_sets[i]):
                child.fuzzy_sets[i].append(copy.deepcopy(fs1) if random.random() < 0.5 else copy.deepcopy(fs2))
        return child

class FuzzyRule:
    def __init__(self, num_inputs, action_dim, fuzzy_kb, indices=None):
        self.num_inputs = num_inputs
        self.action_dim = action_dim
        self.fuzzy_kb = fuzzy_kb
        # Select one fuzzy set for each input
        if indices is None:
            self.indices = [random.randint(0, len(fuzzy_kb.fuzzy_sets[i])-1) for i in range(num_inputs)]
        else:
            self.indices = indices
        self.weights = np.random.uniform(-1, 1, size=(num_inputs+1, action_dim))
        self.fitness = 0.0

    def compute_membership(self, state):
        # Multiply membership values for each input.
        memberships = [self.fuzzy_kb.fuzzy_sets[i][self.indices[i]].membership(state[i])
                       for i in range(self.num_inputs)]
        return np.prod(memberships)

    def compute_action(self, state):
        return self.weights[0] + np.dot(state, self.weights[1:])

    def mutate(self, mutation_rate=0.1, mutation_scale=0.1):
        for i in range(self.num_inputs):
            if random.random() < mutation_rate:
                self.indices[i] = random.randint(0, len(self.fuzzy_kb.fuzzy_sets[i])-1)
        if random.random() < mutation_rate:
            self.weights += np.random.normal(0, mutation_scale, self.weights.shape)

    def __str__(self):
        antecedents = []
        for i, idx in enumerate(self.indices):
            antecedents.append(f"x{i} set#{idx}({self.fuzzy_kb.fuzzy_sets[i][idx]})")
        consequent = ", ".join(f"{w:.2f}" for w in self.weights.flatten())
        return "IF " + " AND ".join(antecedents) + f" THEN action=[{consequent}]"

# ------------------------------
# Pittsburgh Classifier (Complete Rule Set)
# ------------------------------
class PittsburghClassifier:
    def __init__(self, num_rules, num_inputs, action_dim, num_sets=168):
        self.num_inputs = num_inputs
        self.action_dim = action_dim
        self.fuzzy_kb = FuzzyKnowledgeBase(num_inputs, num_sets)
        self.rules = [FuzzyRule(num_inputs, action_dim, self.fuzzy_kb) for _ in range(num_rules)]
        self.fitness = 0.0

    def decide_action(self, state, defuzz_method='weighted_average'):
        activated = []
        memberships = []
        for rule in self.rules:
            m = rule.compute_membership(state)
            if m > 1e-3:
                activated.append(rule.compute_action(state))
                memberships.append(m)
        if memberships:
            activated = np.array(activated)
            memberships = np.array(memberships)
            if defuzz_method == 'weighted_average':
                return np.sum(activated * memberships[:, None], axis=0) / np.sum(memberships)
            else:
                return activated[0]  # Fallback
        else:
            # If no rule fires, return random action.
            return np.random.uniform(-1, 1, size=(self.action_dim,))

    def mutate(self, mutation_rate=0.1, mutation_scale=0.1):
        self.fuzzy_kb.mutate(mutation_rate, mutation_scale)
        for rule in self.rules:
            rule.mutate(mutation_rate, mutation_scale)

    def crossover(self, other):
        child = PittsburghClassifier(len(self.rules), self.num_inputs, self.action_dim)
        child.fuzzy_kb = self.fuzzy_kb.crossover(other.fuzzy_kb)
        child.rules = []
        for rule_self, rule_other in zip(self.rules, other.rules):
            chosen_rule = copy.deepcopy(rule_self) if random.random() < 0.5 else copy.deepcopy(rule_other)
            # Update the KB reference
            chosen_rule.fuzzy_kb = child.fuzzy_kb
            child.rules.append(chosen_rule)
        return child

    def __str__(self):
        s = f"PittsburghClassifier(fitness={self.fitness:.2f})\n"
        for rule in self.rules:
            s += "  " + str(rule) + "\n"
        return s

# ------------------------------
# Pittsburgh LCS Evolutionary System
# ------------------------------
def evaluate_pittsburgh(individual, env, episodes=1, max_steps=400):
    total_reward = 0.0
    for _ in range(episodes):
        state, _ = env.reset()
        ep_reward = 0.0
        done = False
        steps = 0
        while not done and steps < max_steps:
            action = individual.decide_action(state)
            state, reward, done, truncated, _ = env.step(action)
            ep_reward += reward
            steps += 1
        total_reward += ep_reward
    individual.fitness = total_reward / episodes
    return individual.fitness

def pittsburgh_evolution(num_generations=50, population_size=50, num_rules=10, 
                         mutation_rate=0.1, mutation_scale=0.1):
    env = gym.make("LunarLanderContinuous-v2")
    num_inputs = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    population = [PittsburghClassifier(num_rules, num_inputs, action_dim) for _ in range(population_size)]
    
    for gen in range(num_generations):
        for individual in population:
            evaluate_pittsburgh(individual, env)
        population.sort(key=lambda ind: ind.fitness, reverse=True)
        best = population[0]
        print(f"Generation {gen} Best Fitness: {best.fitness:.2f}")
        new_population = [copy.deepcopy(best)]
        while len(new_population) < population_size:
            parent1 = random.choice(population[:3])
            parent2 = random.choice(population[:3])
            child = parent1.crossover(parent2)
            child.mutate(mutation_rate, mutation_scale)
            new_population.append(child)
        population = new_population
        population.sort(key=lambda ind: ind.fitness, reverse=True)
        if population[0].fitness > best.fitness:
            best = population[0]
        else:
            population.append(best)
    env.close()
    # best = population[0]
    print("Best Pittsburgh Classifier:")
    print(best)
    return best



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
    print("=== Version 3: Pittsburgh Style LCS ===")
    best_pittsburgh = pittsburgh_evolution()

    test_final_sequence_agent(best_pittsburgh, episodes=10, max_steps=400, defuzz_method='weighted_average')
