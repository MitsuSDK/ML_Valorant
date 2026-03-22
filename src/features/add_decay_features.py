import pandas as pd
import numpy as np

df = pd.read_csv("data/processed/team_map_dataset.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

df["rating_decay"] = 0.0
df["kd_decay"] = 0.0
decay_rate = 0.01

for index, row in df.iterrows():
    current_date = row["date"]
    team = row["team"]

    past_matches = df[
    (df["team"] == team) &
    (df["date"] < current_date)
    ]

    if past_matches.empty:
        # Neutral baseline to avoid leakage from future data
        rating_decay = 1.0
        kd_decay = 1.0
    else:
        delta_days = (current_date - past_matches["date"]).dt.days
        weights = np.exp(-decay_rate * delta_days)

        rating_decay = np.sum(past_matches["avg_rating"] * weights) / np.sum(weights)
        kd_decay = np.sum(past_matches["team_kd"] * weights) / np.sum(weights)

    df.at[index, "rating_decay"] = rating_decay
    df.at[index, "kd_decay"] = kd_decay

df.to_csv("data/processed/team_map_with_decay.csv", index=False)
print("Decay features added and saved.")
