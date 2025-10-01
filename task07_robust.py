import pandas as pd
import numpy as np
from pathlib import Path

# --------------------
# Load data
# --------------------
games = pd.read_csv(Path(r"C:/Users/Sathvik/Downloads/syracuse_lacrosse_2025_cleaned.csv"))
players = pd.read_csv(Path(r"C:/Users/Sathvik/Downloads/syracuse_lacrosse_2025_player_stats.csv"))

# --------------------
# Setup
# --------------------
games["Margin"] = games["SU_Score"] - games["Opponent_Score"]
players["PPG"] = players["Points"] / players["Games_Played"].replace(0, np.nan)

# --------------------
# 1. Remove top scorer
# --------------------
top_scorer = players.loc[players["Goals"].idxmax(), "Player"]
without_top = players[players["Player"] != top_scorer].copy()
ranked_without_top = without_top.sort_values("Goals", ascending=False).head(5)

print("=== Remove Top Scorer ===")
print(f"Top scorer removed: {top_scorer}")
print("New top 5 goal scorers:")
print(ranked_without_top[["Player","Goals"]])

# --------------------
# 2. Remove top 10% of games (largest wins)
# --------------------
n_remove = max(1, int(0.1 * len(games)))
games_perturbed = games.sort_values("Margin", ascending=False).iloc[n_remove:]
win_rate_full = (games["SU_Score"] > games["Opponent_Score"]).mean()
win_rate_perturbed = (games_perturbed["SU_Score"] > games_perturbed["Opponent_Score"]).mean()

print("\n=== Remove Top 10% of Games (Best Wins) ===")
print(f"Original win rate: {win_rate_full:.3f}")
print(f"After removing {n_remove} games: {win_rate_perturbed:.3f}")

# --------------------
# 3. Normalization change (totals vs PPG)
# --------------------
totals_rank = players.sort_values("Points", ascending=False)[["Player","Points"]].head(5)
ppg_rank = players.sort_values("PPG", ascending=False)[["Player","PPG"]].head(5)

print("\n=== Normalization Change (Totals vs PPG) ===")
print("Top 5 by Total Points:")
print(totals_rank)
print("\nTop 5 by Points per Game:")
print(ppg_rank)

# --------------------
# Optional: Rank stability measure
# --------------------
from scipy.stats import spearmanr

# Spearman correlation between total-points and PPG rankings for players with >=5 games
valid_players = players[players["Games_Played"] >= 5]
rank_total = valid_players["Points"].rank(ascending=False)
rank_ppg = valid_players["PPG"].rank(ascending=False)
rho, _ = spearmanr(rank_total, rank_ppg)

print("\n=== Rank Stability ===")
print(f"Spearman correlation between Totals and PPG rankings: {rho:.2f}")
