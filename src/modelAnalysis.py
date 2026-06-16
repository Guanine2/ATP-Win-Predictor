import os
import time
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from xgboost import plot_importance
import matplotlib.pyplot as plt


clf_loaded = XGBClassifier()
clf_loaded.load_model("src/xgb_model.json")

feature_names = [
    "rank_diff",
    "seed_diff",
    "points_diff",
    "surface_win_pct_diff",
    "tourney_win_pct_diff",
    "h2h_win_diff",
    "avg_game_diff_diff",
    "avg_game_diff_bo5_diff",
    "aces_per_match_diff",
    "aces_per_match_bo5_diff",
    "df_per_match_diff",
    "df_per_match_bo5_diff",
    "first_in_pct_diff",
    "first_in_pct_bo5_diff",
    "first_won_pct_diff",
    "first_won_pct_bo5_diff",
    "second_won_pct_diff",
    "second_won_pct_bo5_diff",
    "bp_save_pct_diff",
    "bp_save_pct_bo5_diff",
    "breaks_per_match_diff",
    "breaks_per_match_bo5_diff",
    "last5_win_diff",
    "last5_game_diff",
    "bp_faced_diff",
    "h2h_last3_win_diff_1",
    "h2h_last3_win_diff_2",
    "h2h_last3_win_diff_3"
]


booster = clf_loaded.get_booster()
booster.feature_names = feature_names


# Path to the 'data' directory (same level as 'src')
data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(data_dir, exist_ok=True)

# Save importance plots for all types
for imp_type in ["gain", "weight", "cover"]:
    plt.figure(figsize=(10, 8))
    plot_importance(clf_loaded, importance_type=imp_type)
    plt.title(f"Feature Importance by {imp_type.capitalize()}")
    plt.tight_layout()
    
    save_path = os.path.join(data_dir, f"feature_importance_{imp_type}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved {save_path}")








'''
importance = booster.get_score(importance_type='weight')
importance_df = pd.DataFrame(list(importance.items()), columns=['Feature', 'FScore'])
importance_df['Normalized'] = importance_df['FScore'] / importance_df['FScore'].max()
importance_df = importance_df.sort_values('Normalized', ascending=False)

cutoff = 0.1 * importance_df['FScore'].max()  # 10% threshold
low_importance = importance_df[importance_df['FScore'] < cutoff]
print("Not relevant features:\n", low_importance)
'''