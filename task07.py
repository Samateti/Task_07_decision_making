#!/usr/bin/env python3
"""
Task 07 — Syracuse WLAX (Full Descriptives + Visuals + Bootstrap + Robustness)
**Fixed to support columns: SU_Score (GF) and Opponent_Score (GA)**

Reads:
  - C:/Users/Sathvik/Downloads/syracuse_lacrosse_2025_cleaned.csv  (game-level)
  - C:/Users/Sathvik/Downloads/syracuse_lacrosse_2025_player_stats.csv (player-level)

Outputs (created under ./results/ and ./figures/ next to this script):
  - results/summary_games.json
  - results/top_scorers.csv
  - results/ppg_ci_bootstrap.csv
  - results/rank_stability.csv
  - figures/top10_goal_scorers_bar.png
  - figures/win_loss_trend.png
  - figures/shots_vs_goals_scatter.png
  - figures/ppg_with_ci.png
  - figures/gf_vs_ga_scatter.png
  - figures/cumulative_goal_margin.png

Notes:
  * matplotlib only (no seaborn), no custom colors.
  * Random seed = 42 for reproducibility.
"""

import json, math, random
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------
# Reproducibility
# ---------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGS = HERE / "figures"
RESULTS.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

# ---------------------------
# Helpers
# ---------------------------
def wilson_ci(successes: int, n: int, z: float = 1.96):
    if n <= 0:
        return (0.0, 0.0, 0.0)
    p = successes / n
    denom = 1.0 + (z*z)/n
    center = (p + (z*z)/(2*n)) / denom
    half = (z * ((p*(1-p))/n + (z*z)/(4*n*n))**0.5) / denom
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return (p, lo, hi)

def find_col(df, candidates):
    cols = {c.strip().lower(): c for c in df.columns}
    for cand in candidates:
        if cand in cols:
            return cols[cand]
    return None

# ---------------------------
# Load data (your Downloads folder)
# ---------------------------
games_path = Path(r"C:/Users/Sathvik/Downloads/syracuse_lacrosse_2025_cleaned.csv")
players_path = Path(r"C:/Users/Sathvik/Downloads/syracuse_lacrosse_2025_player_stats.csv")

if not games_path.exists():
    raise FileNotFoundError(f"Missing file: {games_path}")
if not players_path.exists():
    raise FileNotFoundError(f"Missing file: {players_path}")

games = pd.read_csv(games_path)
players = pd.read_csv(players_path)

# ---------------------------
# Game-level columns
# ---------------------------
gf_col = find_col(games, [
    "gf","goals_for","su_goals","su_score","syracuse_score"
])
ga_col = find_col(games, [
    "ga","goals_against","opp_goals","opponent_score"
])
res_col = find_col(games, ["result","win","won","w"])
date_col = find_col(games, ["date","game_date"])

if gf_col is None or ga_col is None:
    raise ValueError("Could not find GF/GA columns. Expected SU_Score and Opponent_Score in your file.")

games["_GF"] = pd.to_numeric(games[gf_col], errors="coerce").fillna(0).astype(int)
games["_GA"] = pd.to_numeric(games[ga_col], errors="coerce").fillna(0).astype(int)
games["_MARGIN"] = games["_GF"] - games["_GA"]

if res_col is None:
    games["_WIN"] = (games["_GF"] > games["_GA"]).astype(int)
else:
    rc = games[res_col].astype(str).str.upper().str.strip()
    mapped = rc.map({"W":1, "L":0, "1":1, "0":0, "TRUE":1, "FALSE":0})
    games["_WIN"] = mapped.fillna((games["_GF"] > games["_GA"]).astype(int)).astype(int)

wins = int(games["_WIN"].sum())
total = int(len(games))
losses = total - wins
win_rate, lo, hi = wilson_ci(wins, total)

(RESULTS/"summary_games.json").write_text(
    json.dumps({
        "wins": wins,
        "losses": losses,
        "games": total,
        "win_rate": round(win_rate,3),
        "win_rate_wilson95": (round(lo,3), round(hi,3)),
        "generated_at_utc": datetime.utcnow().isoformat()+"Z",
        "seed": SEED
    }, indent=2)
)

# ---------------------------
# Game figures
# ---------------------------
plt.figure()
plt.plot(games.index.values, games["_WIN"].values, marker="o")
plt.title("Game Results Across Season (1=Win, 0=Loss)")
plt.xlabel("Game Index")
plt.ylabel("Win (1) / Loss (0)")
plt.tight_layout()
plt.savefig(FIGS/"win_loss_trend.png")
plt.close()

plt.figure()
plt.scatter(games["_GF"].values, games["_GA"].values)
plt.xlabel("Goals For (GF)")
plt.ylabel("Goals Against (GA)")
plt.title("Per-Game GF vs GA")
plt.tight_layout()
plt.savefig(FIGS/"gf_vs_ga_scatter.png")
plt.close()

plt.figure()
cum_margin = games["_MARGIN"].cumsum()
plt.plot(cum_margin.values, marker="o")
plt.title("Cumulative Goal Margin Over Season")
plt.xlabel("Game Index")
plt.ylabel("Cumulative Margin")
plt.tight_layout()
plt.savefig(FIGS/"cumulative_goal_margin.png")
plt.close()

# ---------------------------
# Player-level columns
# ---------------------------
player_col = find_col(players, ["player"])
goals_col  = find_col(players, ["goals"])
assists_col= find_col(players, ["assists"])
shots_col  = find_col(players, ["shots"])
games_col  = find_col(players, ["games_played","gp"])

players["_Player"] = players[player_col].astype(str)
players["_G"] = pd.to_numeric(players[goals_col], errors="coerce").fillna(0).astype(int)
players["_A"] = pd.to_numeric(players[assists_col], errors="coerce").fillna(0).astype(int) if assists_col else 0
players["_PTS"] = players["_G"] + players["_A"]
players["_SH"] = pd.to_numeric(players[shots_col], errors="coerce") if shots_col else np.nan
players["_GP"] = pd.to_numeric(players[games_col], errors="coerce") if games_col else np.nan

grp = players.groupby("_Player", as_index=False).agg({
    "_G":"sum","_A":"sum","_PTS":"sum","_SH":"sum","_GP":"sum"
})
grp["_PPG"] = grp["_PTS"] / grp["_GP"].replace(0, np.nan)

# Save top scorers
top = grp.sort_values("_G", ascending=False).head(10)
top.to_csv(RESULTS/"top_scorers.csv", index=False)

plt.figure()
plt.bar(top["_Player"], top["_G"])
plt.title("Top 10 Goal Scorers — Syracuse WLAX 2025")
plt.xticks(rotation=45, ha="right")
plt.ylabel("Goals")
plt.tight_layout()
plt.savefig(FIGS/"top10_goal_scorers_bar.png")
plt.close()

# Shots vs Goals
if not grp["_SH"].isna().all():
    plt.figure()
    plt.scatter(grp["_SH"], grp["_G"])
    plt.xlabel("Shots")
    plt.ylabel("Goals")
    plt.title("Shots vs Goals per Player")
    plt.tight_layout()
    plt.savefig(FIGS/"shots_vs_goals_scatter.png")
    plt.close()

# ---------------------------
# Bootstrap CI for PPG (Poisson approx)
# ---------------------------
B = 2000
rows = []
for _, r in grp.iterrows():
    gp = int(r["_GP"]) if not pd.isna(r["_GP"]) else 0
    if gp <= 0:
        continue
    lam = float(r["_PTS"]) / gp
    sims = np.random.poisson(lam, size=(B, gp)).mean(axis=1)
    sims.sort()
    mean = float(sims.mean())
    lo = float(sims[int(0.025*B)])
    hi = float(sims[int(0.975*B)])
    rows.append([r["_Player"], lam, lo, hi, gp, "poisson-approx"])

ppg_ci = pd.DataFrame(rows, columns=["Player","PPG_mean","CI95_lo","CI95_hi","Games","Method"])\
           .sort_values("PPG_mean", ascending=False)
ppg_ci.to_csv(RESULTS/"ppg_ci_bootstrap.csv", index=False)

topN = ppg_ci.head(10)
if not topN.empty:
    y = np.arange(len(topN))
    plt.figure()
    plt.errorbar(topN["PPG_mean"], y,
                 xerr=[topN["PPG_mean"]-topN["CI95_lo"], topN["CI95_hi"]-topN["PPG_mean"]],
                 fmt='o', capsize=5)
    plt.yticks(y, topN["Player"])
    plt.xlabel("Points per Game (PPG)")
    plt.title("PPG with 95% CI (Top 10)")
    plt.tight_layout()
    plt.savefig(FIGS/"ppg_with_ci.png")
    plt.close()

# Rank stability check
players_order = grp.sort_values("_G", ascending=False)["_Player"].tolist()
if len(grp) >= 3:
    top1 = players_order[0]
    base = grp.set_index("_Player")["_G"]
    pert = grp[grp["_Player"] != top1].set_index("_Player")["_G"]
    common = pert.index.intersection(base.index)
    def rank(v): return v.rank(method="average")
    rho = np.corrcoef(rank(base.loc[common]).values, rank(pert.loc[common]).values)[0,1]
else:
    rho = float("nan")

pd.DataFrame({"metric":["spearman_rho_after_remove_top1"], "value":[rho]}).to_csv(RESULTS/"rank_stability.csv", index=False)

print("Done. Results in:", RESULTS, "Figures in:", FIGS, "| Seed:", SEED)
