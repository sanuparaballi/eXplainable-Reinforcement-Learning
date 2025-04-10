#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 16:12:51 2025

@author: sanup
"""

import gymnasium as gym
import numpy as np
import random
from copy import deepcopy

# =============================================================================
# Helper function to generate fuzzy set labels based on number of sets
# =============================================================================


def generate_labels(num_sets):
    """Generate human-readable labels for fuzzy sets."""
    if num_sets == 3:
        return ['low', 'medium', 'high']
    elif num_sets == 5:
        return ['very low', 'low', 'medium', 'high', 'very high']
    else:
        return [f"set_{i}" for i in range(num_sets)]

# =============================================================================
# Global Fuzzy Knowledge Base
# =============================================================================


class FuzzySet:
    def __init__(self, name, a, b, c):
        self.name = name
        self.a = a
        self.b = b
        self.c = c

    def membership(self, x):
        if x < self.a or x > self.c:
            return 0.0
        if x == self.b:
            return 1.0
        if x < self.b:
            return (x - self.a) / (self.b - self.a)
        else:
            return (self.c - x) / (self.c - self.b)


class FuzzyKnowledgeBase:
    def __init__(self, feature_bounds, num_fuzzy_sets):
        self.feature_bounds = feature_bounds
        self.num_fuzzy_sets = num_fuzzy_sets
        self.fuzzy_sets = self.initialize_fuzzy_sets()

    def initialize_fuzzy_sets(self):
        fuzzy_sets = {}
        labels = generate_labels(self.num_fuzzy_sets)
        for idx, (low, high) in enumerate(self.feature_bounds):
            sets = []
            step = (high - low) / (self.num_fuzzy_sets - 1)
            for i in range(self.num_fuzzy_sets):
                a = low + step * (i - 1) if i > 0 else low
                b = low + step * i
                c = low + step * (i + 1) if i < self.num_fuzzy_sets - 1 else high
                sets.append(FuzzySet(labels[i], a, b, c))
            fuzzy_sets[idx] = sets
        return fuzzy_sets

    def get_memberships(self, state):
        memberships = []
        for i, x in enumerate(state):
            feature_memberships = {fs.name: fs.membership(x) for fs in self.fuzzy_sets[i]}
            memberships.append(feature_memberships)
        return memberships

# =============================================================================
# Fuzzy Rule and Corporation Classes
# =============================================================================


class FuzzyRule:
    def __init__(self, antecedents, consequent_params):
        self.antecedents = antecedents
        self.consequent_params = consequent_params  # {"w": weight matrix, "b": bias vector}
        self.fitness = 0.0
        self.eligibility = 0.0

    def compute_activation(self, memberships):
        activation = 1.0
        for i, fuzzy_membership in enumerate(memberships):
            activation *= fuzzy_membership.get(self.antecedents[i], 0.0)
        return activation

    def compute_consequent(self, state):
        # state: vector of shape (state_dim,)
        # w: shape (action_dim, state_dim); b: shape (action_dim,)
        w = self.consequent_params.get("w")
        b = self.consequent_params.get("b")
        return np.dot(w, state) + b


class Corporation:
    def __init__(self, rules=None):
        self.rules = rules if rules else []
        self.fitness = 0.0

    def decide_action(self, state, fuzzy_kb, action_dim):
        memberships = fuzzy_kb.get_memberships(state)
        activations = []
        outputs = []
        for rule in self.rules:
            act = rule.compute_activation(memberships)
            out = rule.compute_consequent(state)
            activations.append(act)
            outputs.append(out)
        activations = np.array(activations)  # shape (num_rules,)
        outputs = np.array(outputs)          # shape (num_rules, action_dim)
        if activations.sum() > 0:
            weighted_output = np.sum(activations[:, None] * outputs, axis=0) / activations.sum()
        else:
            weighted_output = np.mean(outputs, axis=0) if len(outputs) > 0 else np.zeros(action_dim)
        return np.clip(weighted_output, -1, 1)

    def total_activation(self, state, fuzzy_kb):
        memberships = fuzzy_kb.get_memberships(state)
        return sum(rule.compute_activation(memberships) for rule in self.rules)

# =============================================================================
# TD Learning Update
# =============================================================================


def update_rule_fitness_td(trajectory, gamma=0.99, lambda_td=0.8):
    for state, action, rule, reward in reversed(trajectory):
        rule.eligibility = gamma * lambda_td * rule.eligibility + 1
        rule.fitness += rule.eligibility * reward

# =============================================================================
# Evolutionary Engine (GA and PSO Options) - Optional placeholder
# =============================================================================


class EvolutionaryEngine:
    def __init__(self, population, method="GA", ga_params=None, pso_params=None):
        self.population = population  # List of corporations
        self.method = method
        self.ga_params = ga_params if ga_params is not None else {
            "crossover_rate": 0.8, "mutation_rate": 0.1}
        self.pso_params = pso_params if pso_params is not None else {
            "inertia": 0.5, "cognitive": 1.5, "social": 1.5}

    def evolve(self):
        if self.method == "GA":
            self.population = self.ga_evolution(self.population)
        elif self.method == "PSO":
            self.population = self.pso_evolution(self.population)
        else:
            raise ValueError("Unknown evolutionary method.")
        return self.population

    def ga_evolution(self, population):
        new_population = []
        for _ in range(len(population) // 2):
            parent1, parent2 = self.tournament_selection(population), self.tournament_selection(population)
            child1, child2 = self.crossover(parent1, parent2)
            self.mutate(child1)
            self.mutate(child2)
            new_population.extend([child1, child2])
        return new_population

    def tournament_selection(self, population, tournament_size=3):
        participants = random.sample(population, tournament_size)
        participants.sort(key=lambda corp: corp.fitness, reverse=True)
        return deepcopy(participants[0])

    def crossover(self, parent1, parent2):
        child1 = deepcopy(parent1)
        child2 = deepcopy(parent2)
        if random.random() < self.ga_params["crossover_rate"]:
            point = random.randint(1, min(len(parent1.rules), len(parent2.rules))-1)
            child1.rules[:point], child2.rules[:point] = parent2.rules[:point], parent1.rules[:point]
        return child1, child2

    def mutate(self, corporation):
        for rule in corporation.rules:
            if random.random() < self.ga_params["mutation_rate"]:
                rule.consequent_params["w"] += np.random.normal(0,
                                                                0.1, size=rule.consequent_params["w"].shape)
                rule.consequent_params["b"] += np.random.normal(0,
                                                                0.1, size=rule.consequent_params["b"].shape)
        return corporation

    def pso_evolution(self, population):
        new_population = []
        for corp in population:
            new_corp = deepcopy(corp)
            for rule in new_corp.rules:
                rule.consequent_params["w"] += np.random.normal(
                    0, self.pso_params["inertia"], size=rule.consequent_params["w"].shape)
                rule.consequent_params["b"] += np.random.normal(
                    0, self.pso_params["inertia"], size=rule.consequent_params["b"].shape)
            new_population.append(new_corp)
        return new_population

# =============================================================================
# The Agent Combining All Components
# =============================================================================


class FuzzyLCSAgent:
    def __init__(self, env, fuzzy_kb, num_corporations=5, evolution_method="GA"):
        self.env = env
        self.fuzzy_kb = fuzzy_kb
        self.action_dim = env.action_space.shape[0]
        self.state_dim = env.observation_space.shape[0]
        self.corporations = self.initialize_corporations(num_corporations)
        self.evo_engine = EvolutionaryEngine(self.corporations, method=evolution_method)

    def initialize_corporations(self, num_corporations):
        corporations = []
        for _ in range(num_corporations):
            rules = []
            # Initialize each corporation with 3 random rules.
            for _ in range(10):
                antecedents = []
                for i in range(len(self.fuzzy_kb.feature_bounds)):
                    possible_sets = self.fuzzy_kb.fuzzy_sets[i]
                    rule_set = random.choice(possible_sets)
                    antecedents.append(rule_set.name)
                w = np.random.uniform(-1, 1, size=(self.action_dim, self.state_dim))
                b = np.random.uniform(-1, 1, size=(self.action_dim,))
                consequent_params = {"w": w, "b": b}
                rules.append(FuzzyRule(antecedents, consequent_params))
            corporations.append(Corporation(rules))
        return corporations

    def select_action(self, state):
        # Activation-based selection: choose the corporation with the highest total rule activation.
        activations = [corp.total_activation(state, self.fuzzy_kb) for corp in self.corporations]
        best_index = np.argmax(activations) if sum(
            activations) > 0 else random.randint(0, len(self.corporations)-1)
        best_corp = self.corporations[best_index]
        action = best_corp.decide_action(state, self.fuzzy_kb, self.action_dim)
        return np.clip(action, -1, 1)

    def update_fitnesses(self, trajectory):
        update_rule_fitness_td(trajectory)

    def evolve(self):
        self.corporations = self.evo_engine.evolve()

# =============================================================================
# Main Training and Testing Loop
# =============================================================================


def train_agent(episodes=500, evolution_interval=50, evolution_method="GA", num_fuzzy_sets=3):
    env = gym.make('LunarLanderContinuous-v2')
    feature_bounds = [(-1, 1)] * env.observation_space.shape[0]
    fuzzy_kb = FuzzyKnowledgeBase(feature_bounds, num_fuzzy_sets)
    agent = FuzzyLCSAgent(env, fuzzy_kb, num_corporations=10, evolution_method=evolution_method)

    for ep in range(episodes):
        state, info = env.reset()  # Updated reset with two variables.
        done = False
        trajectory = []
        total_reward = 0
        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)  # Updated step return values.
            done = terminated or truncated
            # For TD update, select a rule from the best corporation used.
            chosen_rule = random.choice(random.choice(agent.corporations).rules)
            trajectory.append((state, action, chosen_rule, reward))
            total_reward += reward
            state = next_state
        agent.update_fitnesses(trajectory)
        if ep % evolution_interval == 0 and ep > 0:
            agent.evolve()
        print(f"Episode {ep} reward: {total_reward}")
    return agent, env


def test_agent(agent, episodes=10):
    env = gym.make('LunarLanderContinuous-v2', render_mode='human')
    for ep in range(episodes):
        state, info = env.reset()
        done = False
        total_reward = 0
        while not done:
            env.render()  # Gymnasium's render call
            action = agent.select_action(state)
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
        print(f"Test Episode {ep} reward: {total_reward}")
    env.close()


if __name__ == "__main__":
    num_fuzzy_sets = 5  # User-specified number of fuzzy sets per feature.
    trained_agent, env = train_agent(episodes=500, evolution_interval=20,
                                     evolution_method="PSO", num_fuzzy_sets=num_fuzzy_sets)
    test_agent(trained_agent, episodes=10)
