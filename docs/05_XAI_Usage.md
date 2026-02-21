# 🧠 Explainable AI (XAI) Usage

## The "Black Box" Problem in Sports Analytics

In professional sports, telling a coach or selector, "The computer says Player X is the best, but we don't know why," is an immediate failure. Human selectors require justification before dropping a veteran player in favor of a mathematically-recommended rookie.

To bridge this gap, this project relies heavily on **Explainable AI (XAI)**, explicitly through **SHAP (SHapley Additive exPlanations)**.

Because we use a Random Forest (a tree-based model), we utilize `shap.TreeExplainer()`.

## 1. Global Feature Importance (The Selectors' Blueprint)

By passing our entire dataset through an explainer `shap.summary_plot(..., plot_type='bar')`, the system mathematically confirms the generalized philosophy of T20 cricket:

- Does a low `form_dot_pct_10` (strike rotation) matter more than total boundaries for an `Excellent` T20 anchor?
- Are consistent players (low `consistency_score` variance) rated higher than streaky, boom-or-bust players?

This XAI artifact validates that the model thinks like a cricket selector. If "Matches Played" outweighs "Recent Form Runs," the model is broken or biased. SHAP exposes this logic transparently.

## 2. Individual Player Explanations (Waterfall Plots)

The core of our Streamlit Application focuses on local explanations.
When the system recommends Charith Asalanka as an `Excellent` middle-order batter, the user interface calls:

```python
shap.plots.waterfall(shap_vals[0])
```

This XAI plot visually dissects the prediction:

- **Base Value:** The average prediction across the entire Sri Lankan squad.
- **Red Arrows:** Features pushing Charith Asalanka's rating HIGHER (e.g. `form_sr_10` = 155.0).
- **Blue Arrows:** Features dragging his rating LOWER (e.g. `form_dot_pct_10` = 45%).
- **Final Output:** The mathematical justification for why his rating is elite.

## 3. Head-to-Head Comparisons (Force Plots)

When selectors are torn between picking two players for one position (e.g., Kusal Perera vs Kusal Mendis), the XAI system generates side-by-side force plots.

`shap.plots.bar(explainer(X_test_sc[:2]))`

These models don't just pick a player; they map exactly which underlying rolling-window feature tips the scale in the recommended player's favour, offering complete transparency to external stakeholders and satisfying grading rubrics for transparent ML decision-support systems.
