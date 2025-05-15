#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 10 23:27:32 2025

@author: sanup
"""

import gymnasium as gym
import numpy as np
import pickle
import os
from stable_baselines3 import PPO, DDPG
from stable_baselines3.common.env_util import make_vec_env

# ===========================
# Configurations
# ===========================
ENV_NAME = "LunarLanderContinuous-v3"
AGENT_TYPE = "PPO"      # or "DDPG"
TOTAL_TIMESTEPS = 200_000
DATASET_SIZE = 50_000   # number of (s,a) pairs to collect
OUTPUT_DIR = "data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "lunarlander_sa_dataset.pkl")
MODEL_PATH = os.path.join("models", f"lunar_{AGENT_TYPE.lower()}_agent")

# ===========================
# 1. Create vectorized environment
# ===========================
# We use a single-instance VecEnv for training
train_env = make_vec_env(ENV_NAME, n_envs=1)

# ===========================
# 2. Train or load agent
# ===========================
if os.path.exists(MODEL_PATH + ".zip"):
    print(f"Loading existing {AGENT_TYPE} model from {MODEL_PATH}.zip")
    if AGENT_TYPE == "PPO":
        model = PPO.load(MODEL_PATH, env=train_env)
    else:
        model = DDPG.load(MODEL_PATH, env=train_env)
else:
    print(f"Training new {AGENT_TYPE} model for {TOTAL_TIMESTEPS} timesteps...")
    if AGENT_TYPE == "PPO":
        model = PPO("MlpPolicy", train_env, verbose=1)
    else:
        model = DDPG("MlpPolicy", train_env, verbose=1)
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)
    print(f"Saved trained model to {MODEL_PATH}.zip")

# ===========================
# 3. Rollout and collect data
# ===========================
# Use a fresh evaluation environment (non-vectorized) for deterministic rollouts
eval_env = gym.make(ENV_NAME, render_mode=None)
obs, _ = eval_env.reset()
states, actions = [], []

while len(states) < DATASET_SIZE:
    action, _ = model.predict(obs, deterministic=True)
    next_obs, reward, terminated, truncated, info = eval_env.step(action)
    states.append(obs.copy())
    actions.append(action.copy())
    obs = next_obs
    if terminated or truncated:
        obs, _ = eval_env.reset()

states = np.array(states)      # shape (N, 8)
actions = np.array(actions)    # shape (N, 2)

# ===========================
# 4. Save dataset
# ===========================
os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(OUTPUT_FILE, "wb") as f:
    pickle.dump({"states": states, "actions": actions}, f)

print(f"Saved dataset with {states.shape[0]} samples to {OUTPUT_FILE}")
