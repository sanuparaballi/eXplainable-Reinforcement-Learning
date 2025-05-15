# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Created on Sat May 10 23:50:31 2025

# @author: sanup
# """

# import pickle
# import torch
# import os
# import torch.nn as nn
# import numpy as np
# from sklearn.model_selection import train_test_split

# # Evolutionary libs
# from deap import base, creator, tools, algorithms

# # ===========================
# # CONFIGURATION
# # ===========================
# DATA_PATH = "data/lunarlander_sa_dataset.pkl"
# # MF_TYPE = "gaussian"     # or "triangular", "trapezoidal"
# MF_TYPE = "triangular"
# N_RULES = 16             # number of fuzzy rules
# POP_SIZE = 50             # GA population size
# N_GEN = 40             # GA generations
# CX_PB = 0.7            # crossover probability
# MUT_PB = 0.2            # mutation probability

# DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# # OUTPUT_DIR = "data"
# # OUTPUT_FILE = os.path.join(OUTPUT_DIR, "lunarlander_sa_dataset.pkl")
# MODEL_PATH = os.path.join("models", f"fcs_{MF_TYPE.lower()}_best.pth")

# # ===========================
# # 1. LOAD DATA
# # ===========================
# with open(DATA_PATH, "rb") as f:
#     data = pickle.load(f)
# X = data["states"].astype(np.float32)
# Y = data["actions"].astype(np.float32)
# X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

# X_val_t = torch.from_numpy(X_val).to(DEVICE)
# Y_val_t = torch.from_numpy(Y_val).to(DEVICE)

# # ===========================
# # 2. MODEL DEFINITION
# # ===========================


# class FuzzyPolicyNet(nn.Module):
#     def __init__(self, n_inputs=8, n_rules=16, n_outputs=2, mf_type="gaussian"):
#         super().__init__()
#         self.n_inputs = n_inputs
#         self.n_rules = n_rules
#         self.n_outputs = n_outputs
#         self.mf_type = mf_type

#         # Antecedent parameters
#         if mf_type == "gaussian":
#             self.centers = nn.Parameter(torch.randn(n_rules, n_inputs))
#             self.stds = nn.Parameter(torch.ones(n_rules, n_inputs)*0.5)
#         elif mf_type == "triangular":
#             self.left = nn.Parameter(torch.randn(n_rules, n_inputs)-0.5)
#             self.center = nn.Parameter(torch.randn(n_rules, n_inputs))
#             self.right = nn.Parameter(torch.randn(n_rules, n_inputs)+0.5)
#         else:  # trapezoidal
#             self.a = nn.Parameter(torch.randn(n_rules, n_inputs)-1.0)
#             self.b = nn.Parameter(torch.randn(n_rules, n_inputs)-0.5)
#             self.c = nn.Parameter(torch.randn(n_rules, n_inputs)+0.5)
#             self.d = nn.Parameter(torch.randn(n_rules, n_inputs)+1.0)

#         # Consequent linear coefficients and biases
#         self.coefs = nn.Parameter(torch.randn(n_rules, n_outputs, n_inputs))
#         self.bias = nn.Parameter(torch.zeros(n_rules, n_outputs, 1))

#     def forward(self, x):
#         # Compute membership degrees
#         x_exp = x.unsqueeze(1).expand(-1, self.n_rules, -1)  # (B, R, I)
#         if self.mf_type == "gaussian":
#             diff = (x_exp - self.centers.unsqueeze(0)) / (self.stds.unsqueeze(0) + 1e-6)
#             phi = torch.exp(-0.5 * diff**2)
#         elif self.mf_type == "triangular":
#             l, c, r = [p.unsqueeze(0) for p in (self.left, self.center, self.right)]
#             phi = torch.clamp(torch.min((x_exp-l)/(c-l+1e-6), (r-x_exp)/(r-c+1e-6)), min=0)
#         else:  # trapezoidal
#             a, b, c, d = [p.unsqueeze(0) for p in (self.a, self.b, self.c, self.d)]
#             left = torch.clamp((x_exp - a)/(b - a + 1e-6), 0, 1)
#             right = torch.clamp((d - x_exp)/(d - c + 1e-6), 0, 1)
#             phi = torch.min(left, right)

#         # Rule strengths
#         strength = torch.prod(phi, dim=-1)               # (B, R)
#         norm_str = strength / (strength.sum(dim=1, keepdim=True) + 1e-6)

#         # Consequent outputs
#         lin = torch.einsum('brk,rok->bro', x_exp, self.coefs)  # (B, R, O)
#         lin = lin + self.bias.squeeze(-1).unsqueeze(0)
#         out = (norm_str.unsqueeze(-1) * lin).sum(dim=1)        # (B, O)
#         return out


# # Instantiate a “template” model for shape reference
# _template = FuzzyPolicyNet(n_inputs=X.shape[1], n_rules=N_RULES,
#                            n_outputs=Y.shape[1], mf_type=MF_TYPE).to(DEVICE)
# param_size = sum(p.numel() for p in _template.parameters())

# # ===========================
# # 3. GA SETUP (DEAP)
# # ===========================
# creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # minimize val loss
# creator.create("Individual", list, fitness=creator.FitnessMin)

# toolbox = base.Toolbox()
# # Attribute generator: uniform random init
# toolbox.register("attr_float", np.random.randn)
# # Individual: flatten all model params
# toolbox.register("individual", tools.initRepeat, creator.Individual,
#                  toolbox.attr_float, n=param_size)
# toolbox.register("population", tools.initRepeat, list, toolbox.individual)


# def set_model_params(model, flat_params):
#     """Load flat_params into model.parameters()."""
#     ptr = 0
#     for p in model.parameters():
#         num = p.numel()
#         vals = torch.tensor(flat_params[ptr:ptr+num], dtype=p.dtype)
#         p.data = vals.view_as(p).to(DEVICE)
#         ptr += num


# def evaluate(individual):
#     # load into model
#     model = FuzzyPolicyNet(n_inputs=X.shape[1], n_rules=N_RULES,
#                            n_outputs=Y.shape[1], mf_type=MF_TYPE).to(DEVICE)
#     set_model_params(model, individual)
#     # compute val MSE
#     with torch.no_grad():
#         pred = model(X_val_t)
#         loss = torch.nn.functional.mse_loss(pred, Y_val_t)
#     return (loss.item(),)


# # GA operators
# toolbox.register("evaluate", evaluate)
# toolbox.register("mate", tools.cxTwoPoint)
# toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.05)
# toolbox.register("select", tools.selTournament, tournsize=3)

# # ===========================
# # 4. RUN GA
# # ===========================
# pop = toolbox.population(n=POP_SIZE)
# algorithms.eaSimple(pop, toolbox, cxpb=CX_PB, mutpb=MUT_PB,
#                     ngen=N_GEN, verbose=True)

# # Extract best
# best = tools.selBest(pop, k=1)[0]
# # Load best into a final model
# final_model = FuzzyPolicyNet(n_inputs=X.shape[1], n_rules=N_RULES,
#                              n_outputs=Y.shape[1], mf_type=MF_TYPE).to(DEVICE)
# set_model_params(final_model, best)
# torch.save(final_model, MODEL_PATH)

# # ===========================
# # 5. RULE PRINTING
# # ===========================


# def print_rules(model):
#     """Prints human-readable IF–THEN rules."""
#     for r in range(model.n_rules):
#         antecedents = []
#         for i in range(model.n_inputs):
#             if model.mf_type == "gaussian":
#                 c = model.centers[r, i].item()
#                 s = model.stds[r, i].item()
#                 antecedents.append(f"x{i} IS gauss(center={c:.2f}, std={s:.2f})")
#             elif model.mf_type == "triangular":
#                 l = model.left[r, i].item()
#                 c = model.center[r, i].item()
#                 rr = model.right[r, i].item()
#                 antecedents.append(f"x{i} IS tri(left={l:.2f}, cen={c:.2f}, right={rr:.2f})")
#             else:
#                 a, b, c, d = [p[r, i].item() for p in (model.a, model.b, model.c, model.d)]
#                 antecedents.append(f"x{i} IS trap(a={a:.2f}, b={b:.2f}, c={c:.2f}, d={d:.2f})")
#         # consequent: linear function
#         w = model.coefs[r].cpu().detach().numpy()  # (n_outputs, n_inputs)
#         b = model.bias[r].cpu().detach().numpy().squeeze()
#         cons = []
#         for o in range(model.n_outputs):
#             terms = " + ".join(f"{w[o,j]:+.2f}*x{j}" for j in range(model.n_inputs))
#             terms += f" {b[o]:+.2f}"
#             cons.append(f"a{o} = {terms}")
#         print(f"Rule {r+1}: IF " + " AND ".join(antecedents) + " THEN " + "; ".join(cons))


# # Finally, dump rules
# print_rules(final_model)


import os
import pickle
import torch
import numpy as np
from deap import base, creator, tools, algorithms
from sklearn.model_selection import train_test_split
# from train_fcs_utils import FuzzyPolicyNet  # we’ll factor out the model definition

# ===========================
# CONFIGURATION
# ===========================
DATA_PATH = "data/lunarlander_sa_dataset.pkl"
MODEL_DIR = "models/fcs"
MF_TYPES = ["gaussian", "triangular", "trapezoidal"]
RULE_COUNTS = [4, 8, 16, 32]
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

os.makedirs(MODEL_DIR, exist_ok=True)

# Load data once
with open(DATA_PATH, "rb") as f:
    data = pickle.load(f)
X = data["states"]
Y = data["actions"]
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)


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


def train_fcs_ga(mf_type, n_rules):
    model = FuzzyPolicyNet(n_inputs=8, n_rules=n_rules, n_outputs=2, mf_type=mf_type).to(DEVICE)
    # flatten parameters
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

    def evaluate_ind(ind):
        set_params(ind)
        with torch.no_grad():
            Xv = torch.from_numpy(X_val).to(DEVICE)
            pred, _ = model(Xv)
            loss = torch.nn.functional.mse_loss(pred, torch.from_numpy(Y_val).to(DEVICE))
        return (loss.item(),)

    toolbox.register("evaluate", evaluate_ind)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=50)
    algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2, ngen=40, verbose=False)

    best = tools.selBest(pop, k=1)[0]
    set_params(best)
    return model


# Train & save
for mf in MF_TYPES:
    for r in RULE_COUNTS:
        print(f"Training FCS ({mf}, rules={r})...")
        m = train_fcs_ga(mf, r)
        path = os.path.join(MODEL_DIR, f"fcs_{mf}_{r}.pth")
        torch.save(m.state_dict(), path)
        print(f" -> saved to {path}")
