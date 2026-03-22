import pandas as pd
import numpy as np

df = pd.read_csv("data/processed/team_map_dataset.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

df["rating_decay_map"] = 0.0
df["kd_decay_map"] = 0.0
df["rating_decay_global"] = 0.0
df["kd_decay_global"] = 0.0
df["experience"] = 0
decay_rate = 0.01

for index, row in df.iterrows():
    current_date = row["date"]
    team = row["team"]

    # --- MAP-SPECIFIC HISTORY ---
    past_map_matches = df[
        (df["team"] == team) &
        (df["map_name"] == row["map_name"]) &
        (df["date"] < current_date)
    ]

    # --- GLOBAL HISTORY ---
    past_global_matches = df[
        (df["team"] == team) &
        (df["date"] < current_date)
    ]

    # Experience = number of past matches (global)
    experience = len(past_global_matches)

    # ---------- MAP-SPECIFIC DECAY ----------
    if past_map_matches.empty:
        rating_decay_map = 1.0
        kd_decay_map = 1.0
    else:
        delta_days_map = (current_date - past_map_matches["date"]).dt.days
        weights_map = np.exp(-decay_rate * delta_days_map)

        rating_decay_map = np.sum(past_map_matches["avg_rating"] * weights_map) / np.sum(weights_map)
        kd_decay_map = np.sum(past_map_matches["team_kd"] * weights_map) / np.sum(weights_map)

    # ---------- GLOBAL DECAY ----------
    if past_global_matches.empty:
        rating_decay_global = 1.0
        kd_decay_global = 1.0
    else:
        delta_days_global = (current_date - past_global_matches["date"]).dt.days
        weights_global = np.exp(-decay_rate * delta_days_global)

        rating_decay_global = np.sum(past_global_matches["avg_rating"] * weights_global) / np.sum(weights_global)
        kd_decay_global = np.sum(past_global_matches["team_kd"] * weights_global) / np.sum(weights_global)

    # Save values
    df.at[index, "rating_decay_map"] = rating_decay_map
    df.at[index, "kd_decay_map"] = kd_decay_map
    df.at[index, "rating_decay_global"] = rating_decay_global
    df.at[index, "kd_decay_global"] = kd_decay_global
    df.at[index, "experience"] = experience

df.to_csv("data/processed/team_map_with_decay.csv", index=False)
print("Decay features added and saved.")
