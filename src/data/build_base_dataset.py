import pandas as pd


df_maps     = pd.read_csv("data/raw/detailed_matches_maps.csv")
df_overview = pd.read_csv("data/raw/detailed_matches_overview.csv")
df_player     = pd.read_csv("data/raw/detailed_matches_player_stats.csv")

temp = pd.merge(df_overview[['match_id', 'date']],
                df_maps[['match_id', 'map_name', 'winner']],
                on='match_id',
                how='inner')

# Merge with player stats on both match_id AND map_name to avoid Cartesian product
output = pd.merge(temp,
                  df_player[['match_id', 'map_name', 'team1', 'team2', 'player_name', 'player_team', 'rating', 'k', 'd']],
                  on=['match_id', 'map_name'],
                  how='inner')

output['date'] = pd.to_datetime(output['date'])
output = output.sort_values("date").reset_index(drop=True)
print(output['date'].dtype)

# Displaying result
print(output.head())
print(output.shape)
# save
output.to_csv("data/processed/base_dataset.csv", index=False)