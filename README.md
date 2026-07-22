# Explainable Reinforcement Learning via Hierarchical Fuzzy Rules

This repository contains the code accompanying the paper **"Distilling Deep Reinforcement Learning into Interpretable Fuzzy Rules: An Explainable AI Framework,"** accepted to the **AAAI Spring Symposium Series 2026**.

**Authors:** Sanup S. Araballi¹, Simon Khan², Chilukuri K. Mohan¹
¹ Department of EECS, Syracuse University · ² Air Force Research Laboratory, Rome, NY

Contact: ssarabal@syr.edu

---

## What this is

Deep RL agents trained for continuous control (thrust vectors, torques, steering angles) make decisions that are effectively locked inside a neural network's weights. That opacity is a real problem for any domain that needs verification or certification before an agent can be trusted: aerospace, robotics, autonomous vehicles.

This project distills a trained continuous-control policy into a small set of human-readable IF-THEN rules using a **Hierarchical Takagi-Sugeno-Kang (TSK) Fuzzy Classifier System**, without giving up much of the original policy's accuracy.

The core idea is a two-level decomposition:

1. **State partitioning (Level 1):** K-Means clustering groups the state space into a handful of operational regions ("hovering," "correcting drift," and so on), each represented by a fuzzy membership function (Gaussian or Triangular).
2. **Local action inference (Level 2):** Within each region, a Ridge Regression model learns a simple linear function mapping state to action. Final actions are a weighted blend of whichever regions are active for the current state.

This decoupling is what lets the approach scale to continuous action spaces, where plain decision trees and classic Learning Classifier Systems tend to break down.

## Why it matters

Existing explainability approaches for RL fall into a few buckets, and each has a real limitation this work tries to address:

- **Local explanation methods** (SHAP, LIME) explain individual decisions but never add up to a picture of the policy as a whole.
- **Decision trees** are globally interpretable but approximate smooth control functions with piecewise-constant regions, forcing an uncomfortable trade-off between tree depth and accuracy.
- **Neuro-fuzzy methods** are interpretable by construction but have historically lagged behind modern deep RL on anything but toy problems.

The framework here is a post-hoc surrogate: train the neural policy however you like (this work uses PPO), then distill it into fuzzy rules afterward.

## Results

Evaluated on `LunarLanderContinuous-v3`, distilling a PPO teacher trained via Stable-Baselines3:

| Model | Fidelity | MSE | DTW distance | FRAD | FSC |
|---|---|---|---|---|---|
| Simple MLP (upper bound, not interpretable) | 96.84% | 0.0016 | 0.55 | N/A | N/A |
| FCS, Triangular membership, 16 rules | 81.48% | 0.0053 | 1.05 | 0.814 | 0.933 |
| FCS, Gaussian membership, 16 rules | 81.38% | 0.0037 | 0.87 | 0.723 | 0.974 |
| Decision Tree, 16 leaves | 60.14% | 0.0074 | 1.32 | N/A | N/A |

A few things worth calling out:

- The fuzzy surrogate beats the decision tree baseline by **21 points of fidelity** at the same rule count, since TSK rules output smooth linear functions instead of constant values per leaf.
- **Triangular membership functions produce meaningfully more focused explanations** than Gaussian ones (FRAD 0.814 vs. 0.723, paired t-test p < 0.001), with no fidelity cost, because they activate fewer rules at once for any given state.
- Reducing the rule count from 16 to 4 actually **improved** fidelity, up to 97.83%. The paper calls this the "less is more" finding: the underlying policy's logic seems to be governed by a small number of dominant operational modes, and forcing more rules than that just fragments them and adds noise.

Three metrics were introduced specifically to make "interpretability" measurable rather than a qualitative claim:

- **FRAD** (Fuzzy Rule Activation Density): how concentrated the rule activations are for a given state. Closer to 1 means one rule clearly dominates; closer to 1/N means many rules fire ambiguously.
- **FSC** (Fuzzy Set Coverage): whether the learned fuzzy sets actually cover the operational state space, or leave gaps that force unreliable extrapolation.
- **ASG** (Action Space Granularity): variance across rule outputs, confirming the rules capture genuinely different behaviors rather than collapsing to near-identical actions.

Dynamic Time Warping was also used to confirm the surrogate's rollouts track the teacher's trajectories over time, not just match actions at individual timesteps.

## Repository structure

> Note: these descriptions are based on file names and the paper's methodology; if any of these have drifted from what's actually inside them, let me know and I'll correct this section.

```
.
├── eXplainable Full Framework/   # Main hierarchical TSK-FCS pipeline described in the paper
├── models/                       # Trained teacher policy checkpoints
├── AC LunarLander.py             # Actor-Critic teacher training, Lunar Lander (Continuous)
├── AC Gumbel Lunar Lander.py     # Actor-Critic variant using Gumbel-softmax
├── ACLunarLander.py              # Earlier Actor-Critic Lunar Lander script
├── AC MCC.py                     # Actor-Critic teacher training, Mountain Car Continuous
├── FCS - Michigan.py             # Michigan-style learning classifier system experiments
├── FCS - Pittsburgh.py           # Pittsburgh-style learning classifier system experiments
├── FuzzyCorp - Corporation.py    # Fuzzy rule set / corpus generation utilities
├── FuzzyCorp-AprilUpdate.py      # Later revision of the above
└── XAI - Try1.py                 # Early prototype
```

## Setup

```bash
pip install stable-baselines3 scikit-learn gymnasium numpy torch
```

Train (or supply) a PPO teacher on `LunarLanderContinuous-v3`, collect state-action rollout data, then run the fuzzy distillation pipeline in `eXplainable Full Framework/` to fit the hierarchical TSK-FCS surrogate.

## Citation

```bibtex
@inproceedings{araballi2026distilling,
  title     = {Distilling Deep Reinforcement Learning into Interpretable Fuzzy Rules: An Explainable AI Framework},
  author    = {Araballi, Sanup S. and Khan, Simon and Mohan, Chilukuri K.},
  booktitle = {AAAI Spring Symposium Series},
  year      = {2026}
}
```

## Acknowledgments

This research was partially supported by the Air Force Research Laboratory and is approved under PA number AFRL-2026-0219.
