import pandas as pd

# Source dataset path
path = "data/processed/base_dataset.csv"

# Read base dataset with parsed dates
required_columns = ["match_id", "date", "map_name", "winner", "team1", "team2", "player_name", "player_team", "rating", "k", "d"]
df = pd.read_csv(path, parse_dates=["date"])

missing = [c for c in required_columns if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns in {path}: {missing}")

# Binary flag if this player's team won the match
# In case winner is team name. For some versions winner may be bool/flag.
if df["winner"].dtype == "bool" or set(df["winner"].dropna().unique()) <= {0, 1, True, False}:
    # If winner is 0/1, ensure player_team is aligned by team1/team2 and winner assignment
    if "team1" in df.columns and "team2" in df.columns:
        df["is_winner"] = ((df["player_team"] == df["team1"]) & (df["winner"] == 1)) | ((df["player_team"] == df["team2"]) & (df["winner"] == 0))
        df["is_winner"] = df["is_winner"].astype(int)
    else:
        df["is_winner"] = df["winner"].astype(int)
else:
    df["is_winner"] = (df["player_team"] == df["winner"]).astype(int)

# Compute opponent name per row if team1/team2 exist
def resolve_opponent(row):
    if row["player_team"] == row["team1"]:
        return row["team2"]
    if row["player_team"] == row["team2"]:
        return row["team1"]
    return None

if "team1" in df.columns and "team2" in df.columns:
    df["opponent_team"] = df.apply(resolve_opponent, axis=1)
else:
    df["opponent_team"] = None

# Compute per-player metrics first, per match+map+team
player_stats = (
    df
    .groupby(["match_id", "map_name", "player_team", "player_name"], as_index=False)
    .agg(
        appearances=("player_name", "size"),
        avg_rating=("rating", "mean"),
        total_k=("k", "sum"),
        total_d=("d", "sum"),
        winner_flag=("is_winner", "max"),
        date=("date", "first"),
        opponent_team=("opponent_team", "first"),
    )
)

# Select top 5 players per match/map/team by frequency (appearances), then rating as tie-breaker.

def top5_player_rows(subdf):
    return subdf.sort_values(["appearances", "avg_rating"], ascending=[False, False]).head(5)

top_df = (
    player_stats
    .groupby(["match_id", "map_name", "player_team"], group_keys=True)
    .apply(top5_player_rows)
    .reset_index(level=["match_id", "map_name", "player_team"])
    .reset_index(drop=True)
)

# Safety checks
assert (top_df.groupby(["match_id", "map_name", "player_team"])["player_name"].nunique() <= 5).all(), "top_df has >5 players in some group"
assert (top_df.groupby(["match_id", "map_name", "player_team"]).ngroups > 0), "top_df grouping has no groups"

# Aggregate to one row per (match,map,team)
agg = (
    top_df
    .groupby(["match_id", "map_name", "player_team"], as_index=False)
    .agg(
        date=("date", "first"),
        opponent_team=("opponent_team", "first"),
        winner_flag=("winner_flag", "max"),
        avg_rating=("avg_rating", "mean"),
        total_k=("total_k", "sum"),
        total_d=("total_d", "sum"),
        player_count=("player_name", "nunique"),
    )
)

# Calculate team K/D ratio
agg["team_kd"] = agg["total_k"] / agg["total_d"]

# Persist aggregated dataset
agg = agg.sort_values("date").reset_index(drop=True)
agg.to_csv("data/processed/team_map_dataset.csv", index=False)
print(f"Exported team_map_dataset with {len(agg)} rows to data/processed/team_map_dataset.csv")
