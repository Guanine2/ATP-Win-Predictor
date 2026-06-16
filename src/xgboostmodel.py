import pandas as pd
import numpy as np
import os
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

start_time = time.time()
# -----------------------------
# Load 2022 data for training
# -----------------------------
script_dir = os.path.dirname(__file__)

# Build file paths for each dataset
file_path_2020 = os.path.join(script_dir, "atp_matches_2020.csv")
file_path_2021 = os.path.join(script_dir, "atp_matches_2021.csv")
file_path_2022 = os.path.join(script_dir, "atp_matches_2022.csv")

# Read each CSV (ensure tourney_date is parsed as datetime)
df_2020 = pd.read_csv(file_path_2020, parse_dates=["tourney_date"])
df_2021 = pd.read_csv(file_path_2021, parse_dates=["tourney_date"])
df_2022 = pd.read_csv(file_path_2022, parse_dates=["tourney_date"])

# Combine all years into one DataFrame
df = pd.concat([df_2020, df_2021, df_2022], ignore_index=True)

df = df[~df['score'].str.contains("RET", na=False)]

# Optional: sort by date for convenience
df = df.sort_values("tourney_date").reset_index(drop=True)

MAJORS = ["Wimbledon", "Australian Open", "Us Open", "Roland Garros"]

# -----------------------------
# Helper functions
# -----------------------------

def add_prev5_features(df):
    df = df.sort_values("tourney_date").copy()

    # Containers for new columns
    df["winner_prev5_game_diff"] = np.nan
    df["winner_prev5_win_pct"] = np.nan
    df["winner_prev5_missing"] = 0
    df["loser_prev5_game_diff"] = np.nan
    df["loser_prev5_win_pct"] = np.nan
    df["loser_prev5_missing"] = 0

    # Dictionary to store rolling history per player
    player_history = {}

    def compute_game_diff(row, player):
        if row['winner_name'] == player:
            games_won = row['w_SvGms'] + (row['l_bpFaced'] - row['l_bpSaved'])
            games_lost = row['l_SvGms'] + (row['w_bpFaced'] - row['w_bpSaved'])
        else:
            games_won = row['l_SvGms'] + (row['w_bpFaced'] - row['w_bpSaved'])
            games_lost = row['w_SvGms'] + (row['l_bpFaced'] - row['l_bpSaved'])
        return games_won - games_lost

    for idx, row in df.iterrows():
        for role in ["winner", "loser"]:
            player = row[f"{role}_name"]

            # Initialize history if needed
            if player not in player_history:
                player_history[player] = []

            # Compute stats from history
            prev_matches = player_history[player][-5:]

            if len(prev_matches) > 0:
                avg_game_diff = np.mean([m["game_diff"] for m in prev_matches])
                win_pct = np.mean([m["win"] for m in prev_matches])
            else:
                avg_game_diff = 0
                win_pct = 0

            df.loc[idx, f"{role}_prev5_game_diff"] = avg_game_diff
            df.loc[idx, f"{role}_prev5_win_pct"] = win_pct
            df.loc[idx, f"{role}_prev5_missing"] = 1 if len(prev_matches) < 5 else 0

        # After assigning features, update history with current match
        winner = row["winner_name"]
        loser = row["loser_name"]

        gd_winner = compute_game_diff(row, winner)
        gd_loser = compute_game_diff(row, loser)

        player_history[winner].append({"game_diff": gd_winner, "win": 1})
        player_history[loser].append({"game_diff": gd_loser, "win": 0})

    return df

df = add_prev5_features(df)

def compute_player_history(player, date, df, surface=None, tourney_name=None):
    """Compute historical stats for a player before a given date."""
    hist = df[(df['winner_name'] == player) | (df['loser_name'] == player)]
    hist = hist[hist['tourney_date'] < date]

    if surface:
        hist_surface = hist[((hist['winner_name'] == player) & (hist['surface'] == surface)) |
                            ((hist['loser_name'] == player) & (hist['surface'] == surface))]
    else:
        hist_surface = hist

    total_matches = len(hist)
    wins = (hist['winner_name'] == player).sum()
    win_pct = wins / total_matches if total_matches > 0 else 0

    surface_wins = (hist_surface['winner_name'] == player).sum()
    surface_total = len(hist_surface)
    surface_win_pct = surface_wins / surface_total if surface_total > 0 else 0

    if tourney_name:
        tourney_hist = hist[hist['tourney_name'] == tourney_name]
        tourney_wins = (tourney_hist['winner_name'] == player).sum()
        tourney_total = len(tourney_hist)
        tourney_win_pct = tourney_wins / tourney_total if tourney_total > 0 else 0
    else:
        tourney_win_pct = 0

    # -----------------------------
    # Best-of-5 matches (majors)
    # -----------------------------
    bo5_hist = hist[hist['tourney_name'].isin(MAJORS)]
    bo5_total = len(bo5_hist)
    bo5_wins = (bo5_hist['winner_name'] == player).sum()
    win_pct_bo5 = bo5_wins / bo5_total if bo5_total > 0 else 0

    # -----------------------------
    # Serving stats
    # -----------------------------
    def compute_serving_stats(hist_sub):
        w_hist = hist_sub[hist_sub['winner_name'] == player]
        l_hist = hist_sub[hist_sub['loser_name'] == player]

        n = len(hist_sub) if len(hist_sub) > 0 else 1

        aces = (w_hist['w_ace'].fillna(0).sum() + l_hist['l_ace'].fillna(0).sum()) / n
        dfaults = (w_hist['w_df'].fillna(0).sum() + l_hist['l_df'].fillna(0).sum()) / n
        sv_gms = w_hist['w_SvGms'].fillna(0).sum() + l_hist['l_SvGms'].fillna(0).sum()
        first_in = (w_hist['w_1stIn'].fillna(0).sum() + l_hist['l_1stIn'].fillna(0).sum()) / max(1, sv_gms)
        first_won = (w_hist['w_1stWon'].fillna(0).sum() + l_hist['l_1stWon'].fillna(0).sum()) / max(1, w_hist['w_1stIn'].fillna(0).sum() + l_hist['l_1stIn'].fillna(0).sum())
        second_won = (w_hist['w_2ndWon'].fillna(0).sum() + l_hist['l_2ndWon'].fillna(0).sum()) / max(1, sv_gms - (w_hist['w_1stIn'].fillna(0).sum() + l_hist['l_1stIn'].fillna(0).sum()))
        return aces, dfaults, first_in, first_won, second_won

    aces_per_match, df_per_match, first_in_pct, first_won_pct, second_won_pct = compute_serving_stats(hist)
    aces_per_match_bo5, df_per_match_bo5, first_in_pct_bo5, first_won_pct_bo5, second_won_pct_bo5 = compute_serving_stats(bo5_hist)

    # -----------------------------
    # Break point stats
    # -----------------------------
    def compute_bp_stats(hist_sub):
        bp_faced = bp_saved = breaks_made = 0

        winner_hist = hist_sub[hist_sub['winner_name'] == player]
        bp_faced += winner_hist['w_bpFaced'].fillna(0).sum()
        bp_saved += winner_hist['w_bpSaved'].fillna(0).sum()
        breaks_made += (winner_hist['l_bpFaced'] - winner_hist['l_bpSaved']).fillna(0).sum()

        loser_hist = hist_sub[hist_sub['loser_name'] == player]
        bp_faced += loser_hist['l_bpFaced'].fillna(0).sum()
        bp_saved += loser_hist['l_bpSaved'].fillna(0).sum()
        breaks_made += (loser_hist['w_bpFaced'] - loser_hist['w_bpSaved']).fillna(0).sum()

        bp_save_pct = bp_saved / bp_faced if bp_faced > 0 else 0
        breaks_per_match = breaks_made / len(hist_sub) if len(hist_sub) > 0 else 0
        return bp_save_pct, breaks_per_match

    bp_save_pct, breaks_per_match = compute_bp_stats(hist)
    bp_save_pct_bo5, breaks_per_match_bo5 = compute_bp_stats(bo5_hist)

    # -----------------------------
    # Game differential stats
    # -----------------------------
    def compute_game_diff_stats(hist_sub):
        def compute_game_diff(row):
            if row['winner_name'] == player:
                games_won = row['w_SvGms'] + (row['l_bpFaced'] - row['l_bpSaved'])
                games_lost = row['l_SvGms'] + (row['w_bpFaced'] - row['w_bpSaved'])
            else:
                games_won = row['l_SvGms'] + (row['w_bpFaced'] - row['w_bpSaved'])
                games_lost = row['w_SvGms'] + (row['l_bpFaced'] - row['l_bpSaved'])
            return games_won, games_lost

        games_won = games_lost = 0
        for _, row in hist_sub.iterrows():
            gw, gl = compute_game_diff(row)
            games_won += gw
            games_lost += gl

        total = len(hist_sub) if len(hist_sub) > 0 else 1
        avg_game_diff = (games_won - games_lost) / total
        return avg_game_diff

    avg_game_diff = compute_game_diff_stats(hist)
    avg_game_diff_bo5 = compute_game_diff_stats(bo5_hist)

    return {
        "win_pct": win_pct,
        "surface_win_pct": surface_win_pct,
        "tourney_win_pct": tourney_win_pct,
        "bp_save_pct": bp_save_pct,
        "breaks_per_match": breaks_per_match,
        "avg_game_diff": avg_game_diff,
        "aces_per_match": aces_per_match,
        "df_per_match": df_per_match,
        "first_in_pct": first_in_pct,
        "first_won_pct": first_won_pct,
        "second_won_pct": second_won_pct,
        # Best-of-5
        "win_pct_bo5": win_pct_bo5,
        "bp_save_pct_bo5": bp_save_pct_bo5,
        "breaks_per_match_bo5": breaks_per_match_bo5,
        "avg_game_diff_bo5": avg_game_diff_bo5,
        "aces_per_match_bo5": aces_per_match_bo5,
        "df_per_match_bo5": df_per_match_bo5,
        "first_in_pct_bo5": first_in_pct_bo5,
        "first_won_pct_bo5": first_won_pct_bo5,
        "second_won_pct_bo5": second_won_pct_bo5
    }

def compute_h2h(playerA, playerB, date, df):
    """Compute historical head-to-head stats before date with last 3 matches."""
    h2h_matches = df[((df['winner_name'] == playerA) & (df['loser_name'] == playerB)) |
                     ((df['winner_name'] == playerB) & (df['loser_name'] == playerA))]
    h2h_matches = h2h_matches[h2h_matches['tourney_date'] < date]
    h2h_matches = h2h_matches.sort_values("tourney_date", ascending=False)

    last3 = h2h_matches.head(3)

    # Initialize arrays with np.nan if not enough matches
    win_diff_last3 = []
    game_diff_last3 = []

    for _, row in last3.iterrows():
        if row['winner_name'] == playerA:
            win_diff_last3.append(1)
            gw = row['w_SvGms'] + (row['l_bpFaced'] - row['l_bpSaved'])
            gl = row['l_SvGms'] + (row['w_bpFaced'] - row['w_bpSaved'])
        else:
            win_diff_last3.append(0)
            gw = row['l_SvGms'] + (row['w_bpFaced'] - row['w_bpSaved'])
            gl = row['w_SvGms'] + (row['l_bpFaced'] - row['l_bpSaved'])
        game_diff_last3.append(gw - gl)

    # Pad with np.nan if fewer than 3 matches
    while len(win_diff_last3) < 3:
        win_diff_last3.append(np.nan)
        game_diff_last3.append(np.nan)

    # Cumulative stats
    a_wins = (h2h_matches['winner_name'] == playerA).sum()
    b_wins = (h2h_matches['winner_name'] == playerB).sum()
    cum_win_diff = a_wins - b_wins

    cum_game_diff = 0
    for _, row in h2h_matches.iterrows():
        if row['winner_name'] == playerA:
            gw = row['w_SvGms'] + (row['l_bpFaced'] - row['l_bpSaved'])
            gl = row['l_SvGms'] + (row['w_bpFaced'] - row['w_bpSaved'])
        else:
            gw = row['l_SvGms'] + (row['w_bpFaced'] - row['w_bpSaved'])
            gl = row['w_SvGms'] + (row['l_bpFaced'] - row['l_bpSaved'])
        cum_game_diff += (gw - gl)

    return {
        "win_diff": cum_win_diff,
        "game_diff": cum_game_diff,
        "last3_win_diff": win_diff_last3,
        "last3_game_diff": game_diff_last3
    }

# -----------------------------
# Feature vector builder
# -----------------------------
def build_feature_vector(a_stats, b_stats, h2h, row, best5_flag):
    rank_diff = row['winner_rank'] - row['loser_rank']
    seed_diff = (row['winner_seed'] if not pd.isna(row['winner_seed']) else 0) - \
                (row['loser_seed'] if not pd.isna(row['loser_seed']) else 0)
    points_diff = row['winner_rank_points'] - row['loser_rank_points']

    last5_win_diff = (
        row["winner_prev5_win_pct"] - row["loser_prev5_win_pct"]
    ) if (row["winner_prev5_missing"] == 0 and row["loser_prev5_missing"] == 0) else np.nan

    last5_game_diff = (
        row["winner_prev5_game_diff"] - row["loser_prev5_game_diff"]
    ) if (row["winner_prev5_missing"] == 0 and row["loser_prev5_missing"] == 0) else np.nan

    # -----------------------------
    # Feature vector
    # -----------------------------
    features = [
        rank_diff,
        seed_diff,
        points_diff,
        a_stats['surface_win_pct'] - b_stats['surface_win_pct'],
        a_stats['tourney_win_pct'] - b_stats['tourney_win_pct'],
        h2h["win_diff"],
        a_stats['avg_game_diff'] - b_stats['avg_game_diff'],
        (a_stats['avg_game_diff_bo5'] - b_stats['avg_game_diff_bo5']) if best5_flag else np.nan,
        a_stats['aces_per_match'] - b_stats['aces_per_match'],
        (a_stats['aces_per_match_bo5'] - b_stats['aces_per_match_bo5']) if best5_flag else np.nan,
        a_stats['df_per_match'] - b_stats['df_per_match'],
        (a_stats['df_per_match_bo5'] - b_stats['df_per_match_bo5']) if best5_flag else np.nan,
        a_stats['first_in_pct'] - b_stats['first_in_pct'],
        (a_stats['first_in_pct_bo5'] - b_stats['first_in_pct_bo5']) if best5_flag else np.nan,
        a_stats['first_won_pct'] - b_stats['first_won_pct'],
        (a_stats['first_won_pct_bo5'] - b_stats['first_won_pct_bo5']) if best5_flag else np.nan,
        a_stats['second_won_pct'] - b_stats['second_won_pct'],
        (a_stats['second_won_pct_bo5'] - b_stats['second_won_pct_bo5']) if best5_flag else np.nan,
        a_stats['bp_save_pct'] - b_stats['bp_save_pct'],
        (a_stats['bp_save_pct_bo5'] - b_stats['bp_save_pct_bo5']) if best5_flag else np.nan,
        a_stats['breaks_per_match'] - b_stats['breaks_per_match'],
        (a_stats['breaks_per_match_bo5'] - b_stats['breaks_per_match_bo5']) if best5_flag else np.nan,
        last5_win_diff,
        last5_game_diff,
        a_stats.get("bp_faced", 0) - b_stats.get("bp_faced", 0),   # New: break points faced
        *h2h["last3_win_diff"],   
    ]

    return features


# -----------------------------
# Build features for 2022
# -----------------------------
features = []
labels = []

for idx, row in df.iterrows():
    playerA = row['winner_name']
    playerB = row['loser_name']
    date = row['tourney_date']
    surface = row['surface']
    tourney = row['tourney_name']
    best5_flag = 1 if tourney in MAJORS else 0

    a_stats = compute_player_history(playerA, date, df, surface, tourney)
    b_stats = compute_player_history(playerB, date, df, surface, tourney)
    h2h = compute_h2h(playerA, playerB, date, df)

    feature_vector = build_feature_vector(a_stats, b_stats, h2h, row, best5_flag)
    features.append(feature_vector)
    labels.append(1)

    # Reverse feature vector (preserve NaNs)
    feature_vector_rev = [-x if not pd.isna(x) else np.nan for x in feature_vector]
    features.append(feature_vector_rev)
    labels.append(0)

X = np.array(features)
y = np.array(labels)

# -----------------------------
# Train model with XGBoost
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = XGBClassifier(
    n_estimators=200,
    random_state=42,
    use_label_encoder=False,
    eval_metric="logloss"
)
clf.fit(X, y)

clf.save_model("xgb_model.json")

#y_pred = clf.predict(X_test)
#print("2022 Training Test Accuracy:", accuracy_score(y_test, y_pred))

# -----------------------------
# Rolling evaluation on 2023
# -----------------------------
'''
file_path_2023 = os.path.join(script_dir, "atp_matches_2023.csv")
df_2023 = pd.read_csv(file_path_2023, parse_dates=["tourney_date"])
df_2023 = df_2023.sort_values("tourney_date")
historical_df = df.copy()
test_results = []
df_2023 = add_prev5_features(df_2023)

for idx, row in df_2023.iterrows():
    playerA = row['winner_name']
    playerB = row['loser_name']
    date = row['tourney_date']
    surface = row['surface']
    tourney = row['tourney_name']
    best5_flag = 1 if tourney in MAJORS else 0

    a_stats = compute_player_history(playerA, date, historical_df, surface, tourney)
    b_stats = compute_player_history(playerB, date, historical_df, surface, tourney)
    h2h = compute_h2h(playerA, playerB, date, historical_df)

    feature_vector = build_feature_vector(a_stats, b_stats, h2h, row, best5_flag)

    prob = clf.predict_proba([feature_vector])[0][1]
    pred = clf.predict([feature_vector])[0]
    test_results.append({"date": date, "playerA": playerA, "playerB": playerB, "pred": pred, "prob": prob, "actual": 1})

    # Update historical_df with current match
    historical_df = pd.concat([historical_df, pd.DataFrame([row])], ignore_index=True)  
    
test_df = pd.DataFrame(test_results)
test_df["correct"] = (test_df["pred"] == test_df["actual"]).astype(int)
print("2023 Rolling Accuracy:", test_df["correct"].mean())
'''
print("Runtime: %.2f seconds" % (time.time() - start_time))

