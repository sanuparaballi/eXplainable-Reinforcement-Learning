#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 11 11:08:45 2025

@author: sanup
"""


import os
import pickle
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from scipy.integrate import quad
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import gymnasium as gym
from stable_baselines3 import PPO
from deap import base, creator, tools, algorithms

# ===========================
# CONFIGURATION
# ===========================
DATA_PATH = "data/lunarlander_sa_dataset.pkl"
MODEL_DIR = "models"
RESULT_CSV = "results_summary.csv"
N_INPUTS = 8
N_OUTPUTS = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FID_TOL = 0.1
DTW_WEIGHT = 0.5

rule_counts = [4, 8, 16, 32]
mf_types = ["gaussian", "triangular", "trapezoidal"]

# Load dataset
with open(DATA_PATH, "rb") as f:
    data = pickle.load(f)
X = data["states"]
Y = data["actions"]
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

# Helper functions


def compute_fidelity(pred, true, tol=FID_TOL):
    d = np.linalg.norm(pred - true, axis=1)
    return 100.0 * np.mean(d < tol)


def rollout(policy, seed=42, n=500):
    env = gym.make("LunarLanderContinuous-v3")
    obs, _ = env.reset(seed=seed)
    traj = []
    for _ in range(n):
        a, _ = policy.predict(obs, deterministic=True)
        traj.append((obs.copy(), a.copy()))
        obs, _, term, trunc, _ = env.step(a)
        if term or trunc:
            break
    return traj

# FCS model definition


class FuzzyPolicyNet(torch.nn.Module):
    def __init__(self, n_inputs, n_rules, n_outputs, mf_type):
        super().__init__()
        self.n_inputs, self.n_rules = n_inputs, n_rules
        self.mf_type = mf_type
        if mf_type == "gaussian":
            self.centers = torch.nn.Parameter(torch.randn(n_rules, n_inputs))
            self.stds = torch.nn.Parameter(torch.ones(n_rules, n_inputs)*0.5)
        elif mf_type == "triangular":
            self.left = torch.nn.Parameter(torch.randn(n_rules, n_inputs)-0.5)
            self.center = torch.nn.Parameter(torch.randn(n_rules, n_inputs))
            self.right = torch.nn.Parameter(torch.randn(n_rules, n_inputs)+0.5)
        else:
            self.a = torch.nn.Parameter(torch.randn(n_rules, n_inputs)-1)
            self.b = torch.nn.Parameter(torch.randn(n_rules, n_inputs)-0.5)
            self.c = torch.nn.Parameter(torch.randn(n_rules, n_inputs)+0.5)
            self.d = torch.nn.Parameter(torch.randn(n_rules, n_inputs)+1)
        self.coefs = torch.nn.Parameter(torch.randn(n_rules, n_outputs, n_inputs))
        self.bias = torch.nn.Parameter(torch.zeros(n_rules, n_outputs, 1))

    def forward(self, x):
        # x: (batch, n_inputs)
        B = x.shape[0]
        x_exp = x.unsqueeze(1).expand(B, self.n_rules, self.n_inputs)
        # compute memberships
        if self.mf_type == "gaussian":
            diff = (x_exp - self.centers.unsqueeze(0)) / (self.stds.unsqueeze(0)+1e-6)
            phi = torch.exp(-0.5*diff**2)
        elif self.mf_type == "triangular":
            l, c, r = self.left.unsqueeze(0), self.center.unsqueeze(0), self.right.unsqueeze(0)
            phi = torch.clamp(torch.min((x_exp-l)/(c-l+1e-6), (r-x_exp)/(r-c+1e-6)), 0.0)
        else:
            a, b, c, d = [p.unsqueeze(0) for p in (self.a, self.b, self.c, self.d)]
            l_s = torch.clamp((x_exp-a)/(b-a+1e-6), 0, 1)
            r_s = torch.clamp((d-x_exp)/(d-c+1e-6), 0, 1)
            phi = torch.min(l_s, r_s)
        strength = phi.prod(dim=-1)                       # (B, R)
        norm = strength / (strength.sum(dim=1, keepdim=True)+1e-6)
        lin = torch.einsum('brk,rok->bro', x_exp, self.coefs) + self.bias.squeeze(-1).unsqueeze(0)
        out = (norm.unsqueeze(-1)*lin).sum(dim=1)          # (B, O)
        return out, strength

# GA training


def train_fcs_ga(mf_type, n_rules):
    model = FuzzyPolicyNet(N_INPUTS, n_rules, N_OUTPUTS, mf_type).to(DEVICE)
    param_size = sum(p.numel() for p in model.parameters())
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    toolbox = base.Toolbox()
    toolbox.register("attr_float", np.random.randn)
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                     toolbox.attr_float, n=param_size)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def set_params(ind):
        ptr = 0
        for p in model.parameters():
            num = p.numel()
            vals = torch.tensor(ind[ptr:ptr+num], dtype=p.dtype)
            p.data = vals.view_as(p).to(DEVICE)
            ptr += num

    def eval_ind(ind):
        set_params(ind)
        with torch.no_grad():
            pred, _ = model(torch.from_numpy(X_val).to(DEVICE))
            loss = torch.nn.functional.mse_loss(pred, torch.from_numpy(Y_val).to(DEVICE))
        return (loss.item(),)
    toolbox.register("evaluate", eval_ind)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)
    pop = toolbox.population(n=50)
    algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2, ngen=40, verbose=False)
    best = tools.selBest(pop, k=1)[0]
    set_params(best)
    return model

# Evaluate any model with .predict or forward


def evaluate_model(model, method, mf_type, n_rules):
    # Predictions
    with torch.no_grad():
        pred_train, str_train = model(torch.from_numpy(X_train).to(DEVICE))
        pred_val,   str_val = model(torch.from_numpy(X_val).to(DEVICE))
    pred_train = pred_train.cpu().numpy()
    pred_val = pred_val.cpu().numpy()
    train_mse = np.mean((pred_train - Y_train)**2)
    val_mse = np.mean((pred_val - Y_val)**2)
    fidelity = compute_fidelity(pred_val, Y_val)
    # FRAD
    frad_vals = np.array([np.sum((s.cpu().numpy() / (s.sum().cpu().numpy()+1e-6))**2) for s in str_val])
    frad_mean, frad_std = frad_vals.mean(), frad_vals.std()
    # FSC: average max membership per dim
    fsc_dims = []
    for d in range(N_INPUTS):
        max_m = []
        for b in range(str_val.shape[0]):
            # approximate membership per dim via phi^{1/n_inputs}
            max_m.append((str_val[b].cpu().numpy()**(1/N_INPUTS)).max())
        fsc_dims.append(np.mean(max_m))
    fsc = np.mean(fsc_dims)
    # ASG
    centers = [(model.coefs[r, 0] @ torch.zeros(N_INPUTS).to(DEVICE) + model.bias[r, 0]).item()
               for r in range(n_rules)]
    std = 0.5
    overlaps = []
    for i in range(len(centers)-1):
        def mf_i(x): return np.exp(-0.5*((x-centers[i])/std)**2)
        def mf_j(x): return np.exp(-0.5*((x-centers[i+1])/std)/std**2)
        low, high = min(centers[i], centers[i+1])-3*std, max(centers[i], centers[i+1])+3*std
        ov, _ = quad(lambda x: min(mf_i(x), mf_j(x)), low, high)
        overlaps.append(ov)
    asg = 1 - (np.mean(overlaps)/max(overlaps))
    # DTW
    ppo = PPO.load(os.path.join(MODEL_DIR, "lunar_ppo_agent.zip"))
    traj_nn = rollout(ppo)

    class W:
        def __init__(self, m): self.m = m

        def predict(self, s, deterministic=True):
            out, _ = self.m(torch.from_numpy(s).to(DEVICE))
            return out.cpu().numpy()[0], None
    traj_fcs = rollout(W(model))
    seq1 = [np.concatenate((s, a)) for s, a in traj_nn]
    seq2 = [np.concatenate((s, a)) for s, a in traj_fcs]
    dtw, _ = fastdtw(seq1, seq2, dist=lambda x, y: DTW_WEIGHT *
                     euclidean(x[:N_INPUTS], y[:N_INPUTS])+(1-DTW_WEIGHT)*euclidean(x[N_INPUTS:], y[N_INPUTS:]))
    return {
        "Method": method, "MF_Type": mf_type, "Rules": n_rules,
        "Train_MSE": train_mse, "Val_MSE": val_mse, "Fidelity": fidelity,
        "FRAD_mean": frad_mean, "FRAD_std": frad_std,
        "FSC": fsc, "ASG": asg, "DTW": dtw
    }


# Main loop
results = []
for mf in mf_types:
    for r in rule_counts:
        fcs = train_fcs_ga(mf, r)
        results.append(evaluate_model(fcs, "FCS_GA", mf, r))

# Decision Tree baseline
dt = DecisionTreeRegressor(max_leaf_nodes=16)
dt.fit(X_train, Y_train)


class DTW:
    def __init__(self, m): self.m = m

    def predict(self, s, deterministic=True):
        return self.m.predict(s.reshape(1, -1))[0], None


results.append(evaluate_model(DTW(dt), "DecisionTree", "DT", np.nan))

# Save results
df = pd.DataFrame(results)
df.to_csv(RESULT_CSV, index=False)
print("Results saved to", RESULT_CSV)
