import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score

# -----------------------------
# Load 2022 data for training
# -----------------------------
script_dir = os.path.dirname(__file__)
file_path_2022 = os.path.join(script_dir, "atp_matches_2022.csv")
df = pd.read_csv(file_path_2022, parse_dates=["tourney_date"])

# -----------------------------
# Helper functions
# -----------------------------
def compute_player_history(player, date, df, surface=None, tourney_name=None, last_n=5):
    hist = df[(df['winner_name'] == player) | (df['loser_name'] == player)]
    hist = hist[hist['tourney_date'] < date]

    if surface:
        hist_surface = hist[((hist['winner_name'] == player) & (hist['surface'] == surface)) |
                            ((hist['loser_name'] == player) & (hist['surface'] == surface))]
    else:
        hist_surface = hist

    # Win %
    wins = (hist['winner_name'] == player).sum()
    total_matches = len(hist)
    win_pct = wins / total_matches if total_matches > 0 else 0
    surface_wins = (hist_surface['winner_name'] == player).sum()
    surface_total = len(hist_surface)
    surface_win_pct = surface_wins / surface_total if surface_total > 0 else 0

    recent = hist.sort_values("tourney_date", ascending=False).head(last_n)
    recent_wins = (recent['winner_name'] == player).sum()
    recent_win_pct = recent_wins / last_n if last_n > 0 else 0

    if tourney_name:
        tourney_hist = hist[hist['tourney_name'] == tourney_name]
        tourney_wins = (tourney_hist['winner_name'] == player).sum()
        tourney_total = len(tourney_hist)
        tourney_win_pct = tourney_wins / tourney_total if tourney_total > 0 else 0
    else:
        tourney_win_pct = 0

    # Break point stats
    bp_faced = 0
    bp_saved = 0
    breaks_made = 0

    ace = 0
    double_fault = 0
    first_in = 0
    first_won = 0
    second_won = 0
    total_svpt = 0

    # winner side
    winner_hist = hist[hist['winner_name'] == player]
    bp_faced += winner_hist['w_bpFaced'].fillna(0).sum()
    bp_saved += winner_hist['w_bpSaved'].fillna(0).sum()
    breaks_made += (winner_hist['l_bpFaced'] - winner_hist['l_bpSaved']).fillna(0).sum()
    ace += winner_hist['w_ace'].fillna(0).sum()
    double_fault += winner_hist['w_df'].fillna(0).sum()
    first_in += winner_hist['w_1stIn'].fillna(0).sum()
    first_won += winner_hist['w_1stWon'].fillna(0).sum()
    second_won += winner_hist['w_2ndWon'].fillna(0).sum()
    total_svpt += winner_hist['w_svpt'].fillna(0).sum()

    # loser side
    loser_hist = hist[hist['loser_name'] == player]
    bp_faced += loser_hist['l_bpFaced'].fillna(0).sum()
    bp_saved += loser_hist['l_bpSaved'].fillna(0).sum()
    breaks_made += (loser_hist['w_bpFaced'] - loser_hist['w_bpSaved']).fillna(0).sum()
    ace += loser_hist['l_ace'].fillna(0).sum()
    double_fault += loser_hist['l_df'].fillna(0).sum()
    first_in += loser_hist['l_1stIn'].fillna(0).sum()
    first_won += loser_hist['l_1stWon'].fillna(0).sum()
    second_won += loser_hist['l_2ndWon'].fillna(0).sum()
    total_svpt += loser_hist['l_svpt'].fillna(0).sum()

    # percentages
    bp_save_pct = bp_saved / bp_faced if bp_faced > 0 else 0
    breaks_per_match = breaks_made / total_matches if total_matches > 0 else 0
    ace_pct = ace / total_svpt if total_svpt > 0 else 0
    df_pct = double_fault / total_svpt if total_svpt > 0 else 0
    first_in_pct = first_in / total_svpt if total_svpt > 0 else 0
    first_won_pct = first_won / first_in if first_in > 0 else 0
    second_won_pct = second_won / (total_svpt - first_in) if (total_svpt - first_in) > 0 else 0

    # Game differential
    def compute_game_diff(row, player):
        if row['winner_name'] == player:
            games_won = row['w_SvGms'] + (row['l_bpFaced'] - row['l_bpSaved'])
            games_lost = row['l_SvGms'] + (row['w_bpFaced'] - row['w_bpSaved'])
        else:
            games_won = row['l_SvGms'] + (row['w_bpFaced'] - row['w_bpSaved'])
            games_lost = row['w_SvGms'] + (row['l_bpFaced'] - row['l_bpSaved'])
        return games_won, games_lost

    if total_matches > 0:
        game_stats = hist.apply(lambda r: compute_game_diff(r, player), axis=1)
        games_won = sum(g[0] for g in game_stats)
        games_lost = sum(g[1] for g in game_stats)
        avg_game_diff = (games_won - games_lost) / total_matches
    else:
        avg_game_diff = 0

    if len(recent) > 0:
        recent_game_stats = recent.apply(lambda r: compute_game_diff(r, player), axis=1)
        rec_games_won = sum(g[0] for g in recent_game_stats)
        rec_games_lost = sum(g[1] for g in recent_game_stats)
        recent_game_diff = (rec_games_won - rec_games_lost) / len(recent)
    else:
        recent_game_diff = 0

    return {
        "win_pct": win_pct,
        "surface_win_pct": surface_win_pct,
        "recent_win_pct": recent_win_pct,
        "tourney_win_pct": tourney_win_pct,
        "bp_save_pct": bp_save_pct,
        "breaks_per_match": breaks_per_match,
        "ace_pct": ace_pct,
        "df_pct": df_pct,
        "first_in_pct": first_in_pct,
        "first_won_pct": first_won_pct,
        "second_won_pct": second_won_pct,
        "avg_game_diff": avg_game_diff,
        "recent_game_diff": recent_game_diff
    }

def compute_h2h(playerA, playerB, date, df):
    h2h_matches = df[((df['winner_name'] == playerA) & (df['loser_name'] == playerB)) |
                     ((df['winner_name'] == playerB) & (df['loser_name'] == playerA))]
    h2h_matches = h2h_matches[h2h_matches['tourney_date'] < date]

    if len(h2h_matches) == 0:
        return {"win_diff": 0, "game_diff": 0}

    a_wins = (h2h_matches['winner_name'] == playerA).sum()
    b_wins = (h2h_matches['winner_name'] == playerB).sum()
    win_diff = a_wins - b_wins

    game_diff = 0
    def compute_game_diff(row, player):
        if row['winner_name'] == player:
            games_won = row['w_SvGms'] + (row['l_bpFaced'] - row['l_bpSaved'])
            games_lost = row['l_SvGms'] + (row['w_bpFaced'] - row['w_bpSaved'])
        else:
            games_won = row['l_SvGms'] + (row['w_bpFaced'] - row['w_bpSaved'])
            games_lost = row['w_SvGms'] + (row['l_bpFaced'] - row['l_bpSaved'])
        return games_won, games_lost

    for _, row in h2h_matches.iterrows():
        gw, gl = compute_game_diff(row, playerA)
        game_diff += (gw - gl)

    return {"win_diff": win_diff, "game_diff": game_diff}

# -----------------------------
# Build feature matrix for 2022
# -----------------------------
features = []
labels = []

for idx, row in df.iterrows():
    playerA = row['winner_name']
    playerB = row['loser_name']
    date = row['tourney_date']
    surface = row['surface']
    tourney = row['tourney_name']

    a_stats = compute_player_history(playerA, date, df, surface, tourney)
    b_stats = compute_player_history(playerB, date, df, surface, tourney)
    h2h = compute_h2h(playerA, playerB, date, df)

    rank_diff = row['winner_rank'] - row['loser_rank']
    winner_seed = row['winner_seed'] if not pd.isna(row['winner_seed']) else 0
    loser_seed = row['loser_seed'] if not pd.isna(row['loser_seed']) else 0
    seed_diff = winner_seed - loser_seed

    feature_vector = [
        rank_diff,
        seed_diff,
        a_stats['surface_win_pct'] - b_stats['surface_win_pct'],
        a_stats['recent_win_pct'] - b_stats['recent_win_pct'],
        a_stats['tourney_win_pct'] - b_stats['tourney_win_pct'],
        h2h["win_diff"],
        h2h["game_diff"],
        a_stats['avg_game_diff'] - b_stats['avg_game_diff'],
        a_stats['recent_game_diff'] - b_stats['recent_game_diff'],
        a_stats['bp_save_pct'] - b_stats['bp_save_pct'],
        a_stats['breaks_per_match'] - b_stats['breaks_per_match'],
        a_stats['ace_pct'] - b_stats['ace_pct'],
        a_stats['df_pct'] - b_stats['df_pct'],
        a_stats['first_in_pct'] - b_stats['first_in_pct'],
        a_stats['first_won_pct'] - b_stats['first_won_pct'],
        a_stats['second_won_pct'] - b_stats['second_won_pct']
    ]
    features.append(feature_vector)
    labels.append(1)

    # Reverse
    feature_vector_rev = [-f if isinstance(f, (int, float)) else f for f in feature_vector]
    features.append(feature_vector_rev)
    labels.append(0)

X = np.array(features)
y = np.array(labels)
# Replace NaN with 0
X = np.nan_to_num(X, nan=0.0)
# -----------------------------
# Train logistic regression
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("2022 Test Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))

# -----------------------------
# Rolling evaluation on 2023
# -----------------------------
file_path_2023 = os.path.join(script_dir, "atp_matches_2023.csv")
df_2023 = pd.read_csv(file_path_2023, parse_dates=["tourney_date"])
df_2023 = df_2023.sort_values("tourney_date")

historical_df = df.copy()
test_results = []

for idx, row in df_2023.iterrows():
    playerA = row['winner_name']
    playerB = row['loser_name']
    date = row['tourney_date']
    surface = row['surface']
    tourney = row['tourney_name']

    a_stats = compute_player_history(playerA, date, historical_df, surface, tourney)
    b_stats = compute_player_history(playerB, date, historical_df, surface, tourney)
    h2h = compute_h2h(playerA, playerB, date, historical_df)

    rank_diff = row['winner_rank'] - row['loser_rank']
    winner_seed = row['winner_seed'] if not pd.isna(row['winner_seed']) else 0
    loser_seed = row['loser_seed'] if not pd.isna(row['loser_seed']) else 0
    seed_diff = winner_seed - loser_seed

    feature_vector = [
        rank_diff,
        seed_diff,
        a_stats['surface_win_pct'] - b_stats['surface_win_pct'],
        a_stats['recent_win_pct'] - b_stats['recent_win_pct'],
        a_stats['tourney_win_pct'] - b_stats['tourney_win_pct'],
        h2h["win_diff"],
        h2h["game_diff"],
        a_stats['avg_game_diff'] - b_stats['avg_game_diff'],
        a_stats['recent_game_diff'] - b_stats['recent_game_diff'],
        a_stats['bp_save_pct'] - b_stats['bp_save_pct'],
        a_stats['breaks_per_match'] - b_stats['breaks_per_match'],
        a_stats['ace_pct'] - b_stats['ace_pct'],
        a_stats['df_pct'] - b_stats['df_pct'],
        a_stats['first_in_pct'] - b_stats['first_in_pct'],
        a_stats['first_won_pct'] - b_stats['first_won_pct'],
        a_stats['second_won_pct'] - b_stats['second_won_pct']
    ]

    pred = clf.predict([feature_vector])[0]
    prob = clf.predict_proba([feature_vector])[0][1]
    actual = 1  # playerA is winner

    test_results.append({
        "date": date,
        "playerA": playerA,
        "playerB": playerB,
        "pred": pred,
        "prob": prob,
        "actual": actual
    })

    # Update history
    historical_df = pd.concat([historical_df, pd.DataFrame([row])], ignore_index=True)

results_df = pd.DataFrame(test_results)
rolling_accuracy = (results_df['pred'] == results_df['actual']).mean()
rolling_precision = precision_score(results_df['actual'], results_df['pred'])
rolling_recall = recall_score(results_df['actual'], results_df['pred'])

print("Rolling Accuracy on 2023:", rolling_accuracy)
print("Rolling Precision on 2023:", rolling_precision)
print("Rolling Recall on 2023:", rolling_recall)
