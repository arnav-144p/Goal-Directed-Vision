# Goal-Directed Active Vision System

A goal-directed active vision agent that learns where to look — trained with Inverse Reinforcement Learning on human gaze data, outperforming passive scan baselines on the COCO-Search18 benchmark.


---

## Overview

Humans don't scan scenes randomly. They fixate strategically — driven by a goal. This project models that behavior.

Given a target object category, the agent learns a fixation policy that mimics human search behavior using IRL, guided by semantic features from CLIP and spatial priors from real eye-tracking data.

---

## Results

| Method | SS (↑) | TFP (↑) |
|---|---|---|
| Passive Baseline (random scan) | 0.31 | 0.29 |
| **Goal-Directed Agent (ours)** | **0.58** | **0.54** |

Significant improvement over passive baseline on both Search Score and Target Fixation Proportion metrics.

---

## Architecture

```
Input Image (COCO scene)
        │
        ▼
   CLIP ViT-L/14
   (semantic patch embeddings)
        │
        ▼
   Fixation Policy Network (PyTorch)
   trained via IRL on COCO-Search18 gaze data
        │
        ▼
   Sequential Fixation Sequence
   (goal-conditioned, human-like)
        │
        ▼
   Target Found / Search Terminated
```

---

## Stack

| Component | Tool |
|---|---|
| Vision backbone | CLIP ViT-L/14 (OpenAI) |
| Deep learning | PyTorch |
| Computer vision | OpenCV |
| IRL training | Custom reward learning loop |
| Dataset | COCO-Search18 (human gaze sequences) |

---

## Dataset

**COCO-Search18** — 18 target object categories with human eye-tracking fixation sequences recorded during goal-directed visual search tasks. Used to learn a reward function via IRL that captures human search behavior.

---

## How It Works

1. **Feature Extraction** — each image patch is encoded using CLIP ViT-L/14 to get rich semantic representations
2. **Reward Learning (IRL)** — a reward function is learned from human fixation sequences in COCO-Search18, capturing what makes a fixation "good" given a target goal
3. **Policy Training** — a fixation policy is trained to maximize the learned reward, producing goal-conditioned sequential fixations
4. **Evaluation** — the agent is evaluated against a passive baseline using Search Score (SS) and Target Fixation Proportion (TFP)

---



## References

- [COCO-Search18](https://sites.google.com/view/cocosearch/) — Yang et al., 2020
- [CLIP](https://github.com/openai/CLIP) — Radford et al., 2021
- Inverse Reinforcement Learning — Ziebart et al., 2008

---

## Author

**Arnav** — [@https_arnav](https://x.com/https_arnav) · [GitHub](https://github.com/arnav-144p) · [Portfolio](https://portfolio-arnav-two.vercel.app)
