import pandas as pd
import numpy as np
from pathlib import Path
import math

# --------------------
# Load data
# --------------------
games = pd.read_csv(Path(r"C:/Users/Sathvik/Downloads/syracuse_lacrosse_2025_cleaned.csv"))
players = pd.read_csv(Path(r"C:/Users/Sathvik/Downloads/syracuse_lacrosse_2025_player_stats.csv"))

# --------------------
# Missingness checks
# --------------------
print("=== Missingness Check ===")
print("Game data missing values per column:")
print(games.isna().sum())
print("\nPlayer data missing values per column:")
print(players.isna().sum())

# --------------------
# Outlier checks
# --------------------
players["Goals_per_Game"] = players["Goals"] / players["Games_Played"].replace(0, np.nan)
gpg_mean = players["Goals_per_Game"].mean()
gpg_std = players["Goals_per_Game"].std()

players["Outlier_flag"] = (players["Goals_per_Game"] > gpg_mean + 4*gpg_std)

print("\n=== Outlier Check ===")
print(f"Mean GPG: {gpg_mean:.2f}, Std: {gpg_std:.2f}")
print("Potential outliers (Goals/Game > mean + 4*std):")
print(players[players["Outlier_flag"]][["Player","Goals_per_Game"]])

# --------------------
# Statistical test: Syracuse vs Opponent goals
# Using one-sample t-test implemented manually
# --------------------
diff = games["SU_Score"] - games["Opponent_Score"]
n = len(diff)
mean_diff = diff.mean()
std_diff = diff.std(ddof=1)
t_stat = mean_diff / (std_diff / math.sqrt(n))
# two-sided p-value approximation using survival function of t ~ normal for large n
from math import erf, sqrt
def normal_cdf(x):  # approximation for p-value
    return (1.0 + erf(x / sqrt(2.0))) / 2.0

p_val = 2 * (1 - normal_cdf(abs(t_stat)))

cohen_d = mean_diff / std_diff

print("\n=== Statistical Test ===")
print(f"Mean margin: {mean_diff:.2f}")
print(f"t-statistic (approx): {t_stat:.2f}, p-value (approx): {p_val:.4f}")
print(f"Cohen's d: {cohen_d:.2f}")

# --------------------
# Data leakage checks
# --------------------
team_goals = games["SU_Score"].sum()
player_goals = players["Goals"].sum()

print("\n=== Data Consistency Check ===")
print(f"Total team goals (from games): {team_goals}")
print(f"Total goals (from players): {player_goals}")
print(f"Difference: {team_goals - player_goals}")
