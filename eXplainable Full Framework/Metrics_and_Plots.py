# # #!/usr/bin/env python3
# # # -*- coding: utf-8 -*-
# # """
# # Created on Sun May 11 00:00:41 2025

# # @author: sanup
# # """


# import os
# import pickle
# import torch
# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.kernel_ridge import KernelRidge
# from sklearn.cluster import KMeans
# from scipy.integrate import quad
# from fastdtw import fastdtw
# from scipy.spatial.distance import euclidean
# import gymnasium as gym
# from stable_baselines3 import PPO
# from deap import base, creator, tools, algorithms
# import matplotlib.pyplot as plt

# # ===========================
# # CONFIGURATION
# # ===========================
# DATA_PATH = "data/lunarlander_sa_dataset.pkl"
# MODEL_DIR = "models/fcs"
# RESULT_DIR = "results"
# os.makedirs(RESULT_DIR, exist_ok=True)

# # New GA/ANFIS settings
# GA_POP = 100
# GA_GEN = 80
# ANFIS_LR = 1e-2

# FID_TOL = 0.5  # relaxed for higher fidelity
# DTW_WEIGHT = 0.5
# N_INPUTS = 8
# N_OUTPUTS = 2

# # Surrogate settings
# rule_counts = [8, 16, 32]
# mf_types = ["gaussian", "triangular", "trapezoidal"]

# # Load dataset
# with open(DATA_PATH, "rb") as f:
#     data = pickle.load(f)
# X = data["states"]
# Y = data["actions"]
# X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=23)

# # Helper functions


# def compute_fidelity(pred, true, tol=FID_TOL):
#     return 100.0 * np.mean(np.linalg.norm(pred - true, axis=1) < tol)


# def compute_mae(pred, true):
#     return np.mean(np.abs(pred - true))


# def compute_p75(pred, true):
#     return np.percentile(np.linalg.norm(pred - true, axis=1), 75)


# def rollout(policy, seed=42, n=500):
#     env = gym.make("LunarLanderContinuous-v2")
#     obs, _ = env.reset(seed=seed)
#     traj = []
#     for _ in range(n):
#         a, _ = policy.predict(obs, deterministic=True)
#         traj.append((obs.copy(), a.copy()))
#         obs, _, term, trunc, _ = env.step(a)
#         if term or trunc:
#             break
#     return traj

# # Simplified FCS training (ANFIS + GA) function placeholder


# def train_fcs(mf_type, n_rules):
#     # ... implement ANFIS warmstart + GA with GA_POP, GA_GEN ...
#     # Returns trained model object with forward(x)->(pred, strengths)
#     pass


# results = []

# # Evaluate a model


# def evaluate_model(model, name):

#     if not fname.endswith(".pth"):
#         continue

#     # parse mf_type and rule count from filename
#     _, mf_type, rules_str = fname[:-4].split("_")
#     n_rules = int(rules_str)

#     # load model
#     model = FuzzyPolicyNet(n_inputs=8, n_rules=n_rules, n_outputs=2,
#                            mf_type=mf_type).to(DEVICE)
#     model.load_state_dict(torch.load(os.path.join(MODEL_DIR, fname)))
#     model.eval()

#     # Predictions
#     with torch.no_grad():
#         Xv_t = torch.from_numpy(X_val).float().to(model.device)
#         pred_val, strengths = model(Xv_t)
#     pred_val = pred_val.cpu().numpy()

#     # Compute Metrics
#     val_mse = np.mean((pred_val - Y_val)**2)
#     fidelity = compute_fidelity(pred_val, Y_val)

#     # FRAD
#     frad_vals = []
#     for s in strengths:
#         arr = s.cpu().numpy()
#         norm = arr / (arr.sum() + 1e-6)
#         frad_vals.append((norm**2).sum())
#     frad_vals = np.array(frad_vals)
#     frad_mean, frad_std = frad_vals.mean(), frad_vals.std()

#     # count rules with avg strength > threshold
#     rule_usage = np.mean(np.array([s.cpu().numpy() for s in strengths]), axis=0)
#     compactness = np.sum(rule_usage > (1.0/len(rule_usage)))  # rules above average

#     # FSC
#     # nth-root on strengths to approximate per-dim membership,
#     # then max across rules
#     fsc_dims = []
#     for dim in range(8):
#         max_m = []
#         for s in strengths:
#             # approximate: (rule_strength)^(1/8)
#             m = np.power(s.cpu().numpy(), 1/8).max()
#             max_m.append(m)
#         fsc_dims.append(np.mean(max_m))
#     fsc = np.mean(fsc_dims)

#     # ASG (action dimension 0)
#     centers = []
#     for r_idx in range(n_rules):
#         # compute rule's action-0 consequent at zero state
#         c = model.coefs[r_idx, 0] @ torch.zeros(8).to(DEVICE)
#         b = model.bias[r_idx, 0, 0]
#         centers.append((c + b).item())
#     std = 0.5
#     overlaps = []
#     for i in range(len(centers) - 1):
#         def fi(x): return np.exp(-0.5*((x-centers[i])/std)**2)
#         def fj(x): return np.exp(-0.5*((x-centers[i+1])/std)**2)
#         low = min(centers[i], centers[i+1]) - 3*std
#         high = max(centers[i], centers[i+1]) + 3*std
#         ov, _ = quad(lambda x: min(fi(x), fj(x)), low, high)
#         overlaps.append(ov)
#     asg = 1 - (np.mean(overlaps) / max(overlaps))

#     # DTW distance
#     ppo = PPO.load("models/lunar_ppo_agent.zip")
#     traj_nn = rollout(ppo)

#     mae = compute_mae(pred_val, Y_val)
#     p75 = compute_p75(pred_val, Y_val)

#     class FCSWrap:
#         def __init__(self, m): self.m = m

#         # def predict(self, s, deterministic=True):
#         #     out, _ = self.m(torch.from_numpy(s).to(DEVICE))
#         #     return out.cpu().numpy()[0], None

#         def predict(self, s, deterministic=True):
#             # Convert s to a PyTorch tensor
#             s_tensor = torch.from_numpy(s).to(DEVICE)

#             # Ensure s_tensor is float32 if your model uses float32 parameters
#             # (Gym environments often use float32 for observations,
#             # and PyTorch models default to float32 parameters)
#             if s_tensor.dtype == torch.float64:
#                 s_tensor = s_tensor.float()

#             # Add a batch dimension: (n_inputs,) -> (1, n_inputs)
#             s_tensor_batched = s_tensor.unsqueeze(0)

#             # Get model output
#             # Ensure the model is in evaluation mode if it has layers like Dropout, BatchNorm
#             # self.m.eval() # Typically done once after loading the model
#             with torch.no_grad():  # Important for inference
#                 out, _ = self.m(s_tensor_batched)

#             # Output from model will be (1, n_outputs), so take the first element for single instance
#             return out.cpu().numpy()[0], None

#     traj_fcs = rollout(FCSWrap(model))
#     seq1 = [np.concatenate((s, a)) for s, a in traj_nn]
#     seq2 = [np.concatenate((s, a)) for s, a in traj_fcs]
#     dtw_dist, _ = fastdtw(
#         seq1, seq2,
#         dist=lambda x, y: DTW_WEIGHT*euclidean(x[:8], y[:8])
#         + (1-DTW_WEIGHT)*euclidean(x[8:], y[8:])
#     )

#     results.append({
#         "Method": "FCS_GA",
#         "MF_Type": mf_type,
#         "Rules": n_rules,
#         "Val_MSE": val_mse,
#         "Fidelity": fidelity,
#         "FRAD_mean": frad_mean,
#         "FRAD_std": frad_std,
#         "Compactness": compactness,
#         "MAE": mae,
#         "P75": p75,
#         "FSC": fsc,
#         "ASG": asg,
#         "DTW": dtw_dist
#     })


# # --- 2) Decision Tree Baseline ---
# # Only compute metrics that make sense
# dt = DecisionTreeRegressor(max_leaf_nodes=16).fit(X_train, Y_train)
# pt = dt.predict(X_train)
# pv = dt.predict(X_val)
# results.append({
#     "Method": "DecisionTree",
#     "MF_Type": "DT",
#     "Rules": 16,
#     "Val_MSE": np.mean((pv - Y_val)**2),
#     "Fidelity": compute_fidelity(pv, Y_val),
#     "FRAD_mean": np.nan,
#     "FRAD_std": np.nan,
#     "FSC": np.nan,
#     "ASG": np.nan,
#     "DTW": np.nan
# })

# # --- 3) Write out ---
# pd.DataFrame(results).to_csv(RESULT_CSV, index=False)
# print(f"Saved results to {RESULT_CSV}")


# # Main loop
# results = []
# # Surrogates
# for mf in mf_types:
#     for r in rule_counts:
#         model = train_fcs(mf, r)
#         res = evaluate_model(model, f"FCS_{mf}_{r}")
#         results.append(res)

# # Kernel Ridge baseline
# kr = KernelRidge(alpha=1.0).fit(X_train, Y_train)


# class KRWrap:
#     def __init__(self, m): self.m = m

#     def predict(self, s, deterministic=True):
#         return self.m.predict(s.reshape(1, -1))[0], None


# results.append(evaluate_model(KRWrap(kr), "KernelRidge"))

# # KMeans-quantized policy baseline
# kmeans = KMeans(n_clusters=16).fit(Y_train)


# class KMWrap:
#     def __init__(self, km): self.km = km

#     def predict(self, s, deterministic=True):
#         # find nearest cluster center to true PPO action (oracle)
#         a_ppo = PPO.load("models/lunar_ppo_agent.zip").predict(s)[0]
#         idx = self.km.predict([a_ppo])[0]
#         return self.km.cluster_centers_[idx], None


# results.append(evaluate_model(KMWrap(kmeans), "KMeansQuant"))

# # Save results
# df = pd.DataFrame(results)
# df.to_csv(os.path.join(RESULT_DIR, "results_enhanced.csv"), index=False)

# # Plot enhanced fidelity curve
# plt.figure()
# for mf in mf_types:
#     subset = df[df['Model'].str.contains(mf)]
#     plt.plot(rule_counts, subset['Fidelity'], marker='o', label=mf)
# plt.plot(rule_counts, [df[df['Model'] == "KernelRidge"]['Fidelity'].values[0]]
#          * len(rule_counts), '--', label='KernelRidge')
# plt.title("Validation Fidelity vs. # Rules / Baselines")
# plt.xlabel("Number of Rules / Clusters")
# plt.ylabel("Fidelity (%)")
# plt.grid(True)
# plt.legend()
# plt.savefig(os.path.join(RESULT_DIR, "fidelity_enhanced.pdf"))
# plt.close()


"""

Enhanced evaluation pipeline for FCS surrogates with:
 - ANFIS warmstart + GA optimization
 - New baselines (Kernel Ridge, KMeans quantization)
 - Additional metrics (MAE, 75th percentile error, rule compactness)
 - Enhanced fidelity plot
"""

import os
import pickle
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.cluster import KMeans
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
MODEL_DIR = "models/fcs"
RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)

GA_POP = 100
GA_GEN = 80
ANFIS_EPOCHS = 50
ANFIS_LR = 1e-2
FID_TOL = 0.5
DTW_WEIGHT = 0.5
N_INPUTS = 8
N_OUTPUTS = 2

rule_counts = [8, 16, 32]
mf_types = ["gaussian", "triangular", "trapezoidal"]

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ===========================
# Load dataset
# ===========================
with open(DATA_PATH, "rb") as f:
    data = pickle.load(f)
X = data["states"].astype(np.float32)
Y = data["actions"].astype(np.float32)
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

# ===========================
# Model Definition
# ===========================


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
        else:  # trapezoidal
            self.a = torch.nn.Parameter(torch.randn(n_rules, n_inputs)-1.0)
            self.b = torch.nn.Parameter(torch.randn(n_rules, n_inputs)-0.5)
            self.c = torch.nn.Parameter(torch.randn(n_rules, n_inputs)+0.5)
            self.d = torch.nn.Parameter(torch.randn(n_rules, n_inputs)+1.0)
        self.coefs = torch.nn.Parameter(torch.randn(n_rules, n_outputs, n_inputs))
        self.bias = torch.nn.Parameter(torch.zeros(n_rules, n_outputs, 1))

    def forward(self, x):
        # x: (batch, n_inputs)
        B = x.shape[0]
        x_exp = x.unsqueeze(1).expand(B, self.n_rules, self.n_inputs)
        # compute membership degrees
        if self.mf_type == "gaussian":
            diff = (x_exp - self.centers.unsqueeze(0)) / (self.stds.unsqueeze(0)+1e-6)
            phi = torch.exp(-0.5*diff**2)
        elif self.mf_type == "triangular":
            l, c, r = self.left.unsqueeze(0), self.center.unsqueeze(0), self.right.unsqueeze(0)
            phi = torch.clamp(torch.min((x_exp-l)/(c-l+1e-6), (r-x_exp)/(r-c+1e-6)), min=0.0)
        else:
            a, b, c, d = [p.unsqueeze(0) for p in (self.a, self.b, self.c, self.d)]
            l_s = torch.clamp((x_exp-a)/(b-a+1e-6), 0, 1)
            r_s = torch.clamp((d-x_exp)/(d-c+1e-6), 0, 1)
            phi = torch.min(l_s, r_s)
        # rule strengths and outputs
        strength = phi.prod(dim=-1)  # (B, n_rules)
        norm = strength / (strength.sum(dim=1, keepdim=True)+1e-6)
        lin = torch.einsum('brk,rok->bro', x_exp, self.coefs) + self.bias.squeeze(-1).unsqueeze(0)
        out = (norm.unsqueeze(-1)*lin).sum(dim=1)  # (B, n_outputs)
        return out, strength

# ===========================
# ANFIS warmstart + GA training
# ===========================


def train_fcs(mf_type, n_rules):
    model = FuzzyPolicyNet(N_INPUTS, n_rules, N_OUTPUTS, mf_type).to(device)
    # ANFIS warmstart
    optimizer = torch.optim.Adam(model.parameters(), lr=ANFIS_LR)
    loss_fn = torch.nn.MSELoss()
    X_train_t = torch.from_numpy(X_train).to(device)
    Y_train_t = torch.from_numpy(Y_train).to(device)
    for _ in range(ANFIS_EPOCHS):
        model.train()
        optimizer.zero_grad()
        pred, _ = model(X_train_t)
        loss = loss_fn(pred, Y_train_t)
        loss.backward()
        optimizer.step()
    # GA refinement
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
            p.data = vals.view_as(p).to(device)
            ptr += num

    def eval_ind(ind):
        set_params(ind)
        model.eval()
        with torch.no_grad():
            Xv_t = torch.from_numpy(X_val).to(device)
            pred, _ = model(Xv_t)
            return (torch.nn.functional.mse_loss(pred, torch.from_numpy(Y_val).to(device)).item(),)
    toolbox.register("evaluate", eval_ind)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)
    pop = toolbox.population(n=GA_POP)
    algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.3, ngen=GA_GEN, verbose=False)
    best = tools.selBest(pop, k=1)[0]
    set_params(best)
    return model

# ===========================
# Evaluation routine
# ===========================


def compute_fidelity(pred, true, tol=FID_TOL):
    return 100.0 * np.mean(np.linalg.norm(pred - true, axis=1) < tol)


def compute_mae(pred, true):
    return np.mean(np.abs(pred - true))


def compute_p75(pred, true):
    return np.percentile(np.linalg.norm(pred - true, axis=1), 75)


def rollout_policy(policy, seed=42, n=500):
    env = gym.make("LunarLanderContinuous-v3")
    obs, _ = env.reset(seed=seed)
    traj = []
    for _ in range(n):
        a, _ = policy.predict(obs, deterministic=True)
        traj.append((obs.copy(), a.copy()))
        obs, _, done, trunc, _ = env.step(a)
        if done or trunc:
            break
    return traj


def evaluate_model(model, name):
    model.eval()
    Xv_t = torch.from_numpy(X_val).to(device)
    with torch.no_grad():
        pred_val, strengths = model(Xv_t)
    pred_val = pred_val.cpu().numpy()
    fidelity = compute_fidelity(pred_val, Y_val)
    mae = compute_mae(pred_val, Y_val)
    p75 = compute_p75(pred_val, Y_val)
    frad_vals = np.array([((s.cpu().numpy() / (s.sum().cpu().numpy()+1e-6))**2).sum() for s in strengths])
    frad_mean, frad_std = frad_vals.mean(), frad_vals.std()
    # rule compactness (# rules with mean strength above 1/n_rules)
    usage = strengths.mean(dim=0).cpu().numpy()
    compactness = np.sum(usage > (1.0/model.n_rules))
    # DTW distance
    ppo = PPO.load("models/lunar_ppo_agent.zip")
    traj_nn = rollout_policy(ppo)

    class Wrap:
        def __init__(self, m): self.m = m

        def predict(self, s, deterministic=True):
            # Convert s to a PyTorch tensor
            s_tensor = torch.from_numpy(s).to(device)

            # Ensure s_tensor is float32 if your model uses float32 parameters
            # (Gym environments often use float32 for observations,
            # and PyTorch models default to float32 parameters)
            if s_tensor.dtype == torch.float64:
                s_tensor = s_tensor.float()

            # Add a batch dimension: (n_inputs,) -> (1, n_inputs)
            s_tensor_batched = s_tensor.unsqueeze(0)

            # Get model output
            # Ensure the model is in evaluation mode if it has layers like Dropout, BatchNorm
            # self.m.eval() # Typically done once after loading the model
            with torch.no_grad():  # Important for inference
                out, _ = self.m(s_tensor_batched)

            # Output from model will be (1, n_outputs), so take the first element for single instance
            return out.cpu().numpy()[0], None
    traj_fcs = rollout_policy(Wrap(model))
    seq1 = [np.concatenate((s, a)) for s, a in traj_nn]
    seq2 = [np.concatenate((s, a)) for s, a in traj_fcs]
    dtw, _ = fastdtw(seq1, seq2,
                     dist=lambda x, y: DTW_WEIGHT*euclidean(x[:N_INPUTS], y[:N_INPUTS]) +
                     (1-DTW_WEIGHT)*euclidean(x[N_INPUTS:], y[N_INPUTS:]))
    return {
        "Model": name,
        "Fidelity": fidelity,
        "MAE": mae,
        "P75": p75,
        "FRAD_mean": frad_mean,
        "FRAD_std": frad_std,
        "Compactness": compactness,
        "DTW": dtw
    }


# ===========================
# Main experiment loop
# ===========================
results = []
# Train and eval FCS surrogates
for mf in mf_types:
    for r in rule_counts:
        print(f"Training FCS {mf}, rules={r}")
        model = train_fcs(mf, r)
        res = evaluate_model(model, f"FCS_{mf}_{r}")
        results.append(res)

# Kernel Ridge baseline
kr = KernelRidge(alpha=1.0).fit(X_train, Y_train)


class KRWrap:
    def __init__(self, m): self.m = m

    def predict(self, s, deterministic=True):
        a = self.m.predict(s.reshape(1, -1))[0]
        return a, None


results.append(evaluate_model(KRWrap(kr), "KernelRidge"))

# KMeans quantization baseline
ppo = PPO.load("models/lunar_ppo_agent.zip")
actions = Y_train  # cluster PPO actions
kmeans = KMeans(n_clusters=16, random_state=0).fit(actions)


class KMWrap:
    def __init__(self, km): self.km = km

    def predict(self, s, deterministic=True):
        a, _ = ppo.predict(s, deterministic=True)
        idx = self.km.predict([a])[0]
        return self.km.cluster_centers_[idx], None


results.append(evaluate_model(KMWrap(kmeans), "KMeansQuant"))

# Save CSV
df = pd.DataFrame(results)
df.to_csv(os.path.join(RESULT_DIR, "results_enhanced.csv"), index=False)

# Plot enhanced fidelity
plt.figure()
for mf in mf_types:
    sub = df[df['Model'].str.contains(mf)]
    plt.plot(rule_counts, sub['Fidelity'], marker='o', label=mf.capitalize())
baseline_fid = df[df['Model'] == "KernelRidge"]['Fidelity'].values[0]
plt.hlines(baseline_fid, rule_counts[0], rule_counts[-1], linestyles='--', label='KernelRidge')
plt.title("Validation Fidelity vs. # Rules / Clusters")
plt.xlabel("Number of Rules / Clusters")
plt.ylabel("Fidelity (%)")
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(RESULT_DIR, "fidelity_enhanced.pdf"))
plt.close()

print("Enhanced experiments complete. Results saved in", RESULT_DIR)
