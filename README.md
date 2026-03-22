# ML_Valorant

# Valorant Match Outcome Prediction – Time-Aware ML Pipeline

## Overview

This project builds a **time-aware machine learning pipeline** to predict professional Valorant match outcomes using performance dynamics rather than raw match statistics.

Instead of using naive averages, the system models team strength evolution over time using **exponential decay feature engineering**, ensuring:

- No data leakage  
- Strict chronological training/testing split  
- Map-specific and global performance modeling  
- Proper experimental comparison of models  

The goal of this project is not to maximize leaderboard metrics, but to demonstrate strong applied ML engineering and statistical reasoning.

---

## Problem Statement

Predict whether **Team 1 wins a map** in a professional Valorant match using only information available *before* the match is played.

This is framed as a **binary classification problem**.

Target:
- `1` → Team 1 wins
- `0` → Team 1 loses

---

## Data Pipeline Architecture

Raw match data → Feature engineering → Model dataset → Time-based split → Model comparison

### 1. Base Dataset Construction

Match-level data is merged and cleaned to produce:

- Team name
- Opponent
- Map name
- Match date
- Kills / Deaths
- Average player rating
- Winner flag

Dates are converted to `datetime` and sorted chronologically to ensure time consistency.

---

## Feature Engineering

### Exponential Time Decay

Team performance evolves over time. Older matches should influence the current strength estimate less than recent matches.

For each team before each match:

\[
weight = e^{-\lambda \cdot \Delta t}
\]

Where:
- λ = 0.01
- Δt = days since previous match

This produces weighted historical averages.

---

### Engineered Features

For every team and map:

#### Map-Specific Decay Features
- `rating_decay_map`
- `kd_decay_map`
- `winrate_map_decay`

#### Global Decay Features
- `rating_decay_global`
- `kd_decay_global`
- `winrate_global_decay`

#### Experience
- Number of past matches before current match

All features are computed using **strictly past data only** (`date < current_date`).

No future leakage.

---

## Model Dataset

For each match, features are converted into **team1 – team2 differences**:

- rating_decay_map_diff
- kd_decay_map_diff
- rating_decay_global_diff
- kd_decay_global_diff
- winrate_map_decay_diff
- winrate_global_decay_diff
- experience_diff

This transforms the problem into modeling relative strength.

---

## Evaluation Strategy

### Time-Based Split

Train:
- Matches before October 1, 2025

Test:
- Matches on or after October 1, 2025

This simulates real-world forecasting.

No random shuffling.

---

## Models Implemented

### 1. Logistic Regression (with StandardScaler)

Evaluated across multiple feature sets:
- Baseline decay features
- + Experience
- + Winrate features

### 2. Random Forest

- 200 trees
- Max depth = 5
- Feature importance analysis

---

## Key Findings

- Decay-based features carry more signal than raw match stats.
- Global decay features were generally more stable than map-only features.
- Small dataset size (75 train / 13 test) leads to high metric variance.
- With limited samples, model instability is expected.
- Feature engineering was validated as conceptually correct despite metric fluctuation.

This project demonstrates:

- Temporal feature engineering
- Leakage prevention
- Model comparison discipline
- Bias–variance awareness
- Statistical reasoning under small-sample conditions

---

## Limitations

- Dataset size is small
- High variance in ROC-AUC due to limited test samples
- No cross-validation yet implemented

Future improvements:
- Larger historical dataset
- Rolling time validation
- ELO-style rating system
- Hyperparameter tuning

---

## Tech Stack

- Python
- Pandas
- NumPy
- scikit-learn
- VSCode

---

## Why This Project Matters

This project demonstrates the ability to:

- Design a full ML pipeline from scratch
- Engineer time-aware features
- Prevent data leakage
- Compare linear and tree-based models
- Diagnose overfitting and instability
- Think critically about evaluation

It reflects applied machine learning engineering rather than tutorial-style modeling.

---

## Author

Built as an applied ML engineering project focused on time-series reasoning and predictive modeling in esports analytics.