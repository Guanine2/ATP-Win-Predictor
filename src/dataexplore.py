import pandas as pd
import numpy as np
import os
import time

script_dir = os.path.dirname(__file__)
file_path_2022 = os.path.join(script_dir, "atp_matches_2022.csv")
df = pd.read_csv(file_path_2022)


import pandas as pd

# Mapping of winner/loser columns to standardized names
stat_mapping = {
    'w_ace': 'ace', 'l_ace': 'ace',
    'w_df': 'df', 'l_df': 'df',
    'w_svpt': 'svpt', 'l_svpt': 'svpt',
    'w_1stIn': '1stIn', 'l_1stIn': '1stIn',
    'w_1stWon': '1stWon', 'l_1stWon': '1stWon',
    'w_2ndWon': '2ndWon', 'l_2ndWon': '2ndWon',
    'w_SvGms': 'SvGms', 'l_SvGms': 'SvGms',
    'w_bpSaved': 'bpSaved', 'l_bpSaved': 'bpSaved',
    'w_bpFaced': 'bpFaced', 'l_bpFaced': 'bpFaced',
    'winner_rank': 'rank', 'loser_rank': 'rank',
    'winner_rank_points': 'rank_points', 'loser_rank_points': 'rank_points'
}

# Filter only columns that exist in df
existing_stats = [col for col in stat_mapping if col in df.columns]

# Separate winner and loser columns and rename to standardized names
winner_stats = df[[col for col in existing_stats if col.startswith('w') or col.startswith('winner')]].rename(columns={k: v for k, v in stat_mapping.items() if k in df.columns})
loser_stats = df[[col for col in existing_stats if col.startswith('l') or col.startswith('loser')]].rename(columns={k: v for k, v in stat_mapping.items() if k in df.columns})

# Get descriptive statistics
winner_summary = winner_stats.describe()
loser_summary = loser_stats.describe()

# Save both to a single CSV with separate sections
with open("match_stats_standardized.csv", "w") as f:
    f.write("Winner Stats\n")
    winner_summary.to_csv(f)
    f.write("\nLoser Stats\n")
    loser_summary.to_csv(f)


