import pandas as pd

from pathlib import Path

# Define project root dynamically
BASE_DIR = Path(__file__).resolve().parents[2]


input_path = BASE_DIR / "data" / "processed" / "team_map_with_decay.csv"
df = pd.read_csv(input_path)
df["date"] = pd.to_datetime(df["date"])


df = df.sort_values("date").reset_index(drop=True)

model_rows = []


for (match_id, map_name), group in df.groupby(["match_id", "map_name"]):
    
    if len(group) != 2:
        continue
    group = group.sort_values("team").reset_index(drop=True)
    
    team1 = group.iloc[0]
    team2 = group.iloc[1]
    
    rating_diff = team1["rating_decay"] - team2["rating_decay"]
    kd_diff = team1["kd_decay"] - team2["kd_decay"]
    
    target = team1["winner_flag"]
    
    model_rows.append({
        "match_id": match_id,
        "map_name": map_name,
        "date": team1["date"],
        "team1": team1["team"],
        "team2": team2["team"],
        "rating_diff": rating_diff,
        "kd_diff": kd_diff,
        "target": target
    })


model_df = pd.DataFrame(model_rows)


output_path = BASE_DIR / "data" / "processed" / "model_dataset.csv"
model_df.to_csv(output_path, index=False)

print(f"Model dataset created and saved at {output_path}")
print(model_df.head())