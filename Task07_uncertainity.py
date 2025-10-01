import pandas as pd
import numpy as np
from pathlib import Path
import random

# --------------------
# Settings
# --------------------
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

games_path = Path(r"C:/Users/Sathvik/Downloads/syracuse_lacrosse_2025_cleaned.csv")
players_path = Path(r"C:/Users/Sathvik/Downloads/syracuse_lacrosse_2025_player_stats.csv")

games = pd.read_csv(games_path)
players = pd.read_csv(players_path)

# --------------------
# Helper functions
# --------------------
def wilson_ci(successes, n, z=1.96):
    if n == 0:
        return (0, 0, 0)
    p = successes / n
    denom = 1 + z**2/n
    center = (p + z**2/(2*n)) / denom
    margin = z * np.sqrt((p*(1-p))/n + z**2/(4*n**2)) / denom
    return (p, max(0, center - margin), min(1, center + margin))

def bootstrap_ci(data, B=2000, alpha=0.05):
    """Bootstrap CI for mean of data"""
    n = len(data)
    sims = [np.mean(np.random.choice(data, n, replace=True)) for _ in range(B)]
    sims = np.sort(sims)
    mean = np.mean(sims)
    lo = sims[int((alpha/2)*B)]
    hi = sims[int((1-alpha/2)*B)]
    return mean, lo, hi

# --------------------
# Team-level uncertainty
# --------------------
games["_WIN"] = (games["SU_Score"] > games["Opponent_Score"]).astype(int)
wins = games["_WIN"].sum()
total = len(games)

# Wilson CI
p, lo, hi = wilson_ci(wins, total)
print(f"Win rate: {p:.3f} ({wins}/{total})")
print(f"Wilson 95% CI: [{lo:.3f}, {hi:.3f}]")

# Bootstrap CI for win rate
boot_means = []
B = 2000
for _ in range(B):
    sample = games["_WIN"].sample(n=total, replace=True)
    boot_means.append(sample.mean())
boot_means = np.sort(boot_means)
boot_mean = np.mean(boot_means)
boot_lo, boot_hi = boot_means[int(0.025*B)], boot_means[int(0.975*B)]
print(f"Bootstrap mean win rate: {boot_mean:.3f}")
print(f"Bootstrap 95% CI: [{boot_lo:.3f}, {boot_hi:.3f}]")

# --------------------
# Player-level PPG uncertainty
# --------------------
players["_PPG"] = players["Points"] / players["Games_Played"]

results = []
for _, row in players.iterrows():
    if row["Games_Played"] > 0:
        lam = row["Points"] / row["Games_Played"]
        # bootstrap by Poisson sampling
        sims = [np.mean(np.random.poisson(lam, row["Games_Played"])) for _ in range(B)]
        sims = np.sort(sims)
        mean = np.mean(sims)
        lo, hi = sims[int(0.025*B)], sims[int(0.975*B)]
        results.append((row["Player"], lam, mean, lo, hi))

ppg_ci = pd.DataFrame(results, columns=["Player", "Observed_PPG", "Bootstrap_Mean", "CI_Lo", "CI_Hi"])
print("\nTop 5 players by PPG with CI:")
print(ppg_ci.sort_values("Observed_PPG", ascending=False).head(5))
