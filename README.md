# Task_07_decision_making 

## 📌 Project Purpose  
This project analyzes the 2025 Syracuse Women’s Lacrosse season data to produce validated descriptive results, quantify uncertainty, run robustness/sanity checks, and deliver actionable, risk-tiered recommendations in a **Stakeholder Report**.  

---

## 📂 Repository Structure  

```
.
├── data/
│   ├── syracuse_lacrosse_2025_cleaned.csv
│   ├── syracuse_lacrosse_2025_player_stats.csv
│
├── code/
│   ├── task07_full_analysis.py        # Main analysis script (stats, visuals, uncertainty, robustness)
│   ├── bootstrap_uncertainty.py       # Bootstrap estimates for win rate & PPG
│   ├── sanity_checks.py               # Missingness, outliers, consistency checks
│   └── robustness_tests.py            # Perturbation analysis (remove top N players, blowouts, etc.)
│
├── reports/
│   ├── Syracuse_WLAX_Full_Stakeholder_Report.docx
│   ├── Syracuse_WLAX_Full_Stakeholder_Report.pdf
│   └── Syracuse_WLAX_Full_Stakeholder_Report_With_Visuals.pdf
│
├── visuals/
│   ├── win_loss_trend.png
│   ├── gf_vs_ga_scatter.png
│   ├── cumulative_goal_margin.png
│   ├── top10_goal_scorers_bar.png
│   ├── shots_vs_goals_scatter.png
│   ├── ppg_with_ci.png
│   ├── robust.png
│   ├── sanity.png
│   └── UNicertain.png
│
└── README.md
```

---

## ⚙️ How to Run the Analysis  

1. Clone the repository:  
   ```bash
   git clone <repo-url>
   cd repo-name
   ```

2. Install dependencies:  
   ```bash
   pip install pandas matplotlib numpy
   ```

3. Run the main analysis script:  
   ```bash
   python code/task07_full_analysis.py
   ```

4. Outputs:  
   - Figures   
   - Summary statistics + logs  
   - Stakeholder Report  

---

## 📊 Key Deliverables  

- **Descriptive Stats & Visualizations:** Win/loss trend, GF vs GA scatter, PPG distributions.  
- **Uncertainty Analysis:** Wilson CI, bootstrap estimates.  
- **Sanity Checks:** Missing data, outlier detection, data leakage checks.  
- **Robustness Tests:** Removing top scorers, blowouts, normalization changes.  
- **Stakeholder Report:** Actionable recommendations (operational, investigatory, high-stakes), ethical/legal concerns, next steps.  

---

## 🔍 Reproducibility Notes  

- Random seed fixed at `42` for all bootstrap and resampling procedures.  
- LLM-generated content is clearly labeled and annotated in reports.  
- Code, prompts, and outputs are archived for transparency.  

---

## ✅ Next Steps  

- Extend analysis with opponent-adjusted efficiency metrics.  
- Automate A/B practice block evaluations.  
- Re-estimate uncertainty after next 4–6 games.  
