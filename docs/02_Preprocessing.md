# ⚙️ Preprocessing & Feature Engineering

## Overview

Turning raw cricket data into reliable predictive labels requires significant transformation. We don't train algorithms on raw "runs scored." Instead, we engineer **Rolling Form Features** which encapsulate a player's immediate impact on the game based on their **last 10 matches**.

Our logic is simple: A player's career average does not win a T20 match today. **Current form does.**

## Step 1: Extracting Per-Match Statistics

The system loops through every ball-by-ball record (`balls_df`) and aggregates deliveries into match-level statistics for each active Sri Lankan player.

**Key Batting Outputs:**

- `runs_scored`: Total runs made.
- `balls_faced`: Deliveries faced excluding wides.
- `strike_rate`: Runs scored relative to balls faced (crucial T20 metric).
- `boundaries`: Number of 4s and 6s.
- `dot_ball_pct`: Percentage of deliveries where 0 runs were scored (measures strike rotation pressure).

**Key Bowling Outputs:**

- `wickets_taken`: Number of legal dismissals attributed.
- `runs_conceded`: Total runs allowed including extras.
- `overs_bowled`: Balls bowled divided by 6.
- `economy_rate`: Runs allowed per over.
- `bowling_strike_rate`: Balls bowled per wicket taken.

## Step 2: Form Features (Rolling 10-Match Averages)

Once match-level stats are generated, the system sorts them chronologically and applies a rolling window transformation.

**Why a 10-match window?**
It strikes the perfect balance between an "outlier performance" (1-2 matches) and a "long-term career average" (50+ matches).

```python
# Pseudo-code representation of our feature generation
p_df['form_runs_10'] = p_df['runs_scored'].rolling(10, min_periods=3).mean()
p_df['form_economy_10'] = p_df['economy_rate'].rolling(10, min_periods=3).mean()
```

This transforms the dataset from "isolated matches" into an ongoing "form tracker" covering multiple dimensions (Runs, Boundaries, SR, Wickets, Economy, Consistency/Variance).

## Step 3: Composite Performance Scoring & Labelling

Unlike standard ML classification tasks where the "target label" (y) is pre-provided by humans, we mathematically calculate a target ground-truth label using domain knowledge. We assign specific weights to T20 impact factors to generate a dynamic score out of 100.

**Batting Formula Weighting:**

- 40% Average Runs Form
- 30% Strike Rate Form (In T20, scoring fast is almost as important as scoring big)
- 20% Boundary Frequency Form
- 10% Minimizing Dot Balls

**Bowling Formula Weighting:**

- 40% Average Wickets
- 35% Economy Control (Preventing runs)
- 25% Strike Rate (Frequency of wickets)

### The Categorical Labels

Using these generated scores (0-100), we bin the outputs into four categorical tiers which inform the Random Forest model:

- `Score >= 75`: **Excellent**
- `Score >= 50`: **Good**
- `Score >= 25`: **Average**
- `Score < 25`: **Poor**

This resulting structured `.csv` of engineered features mapped against a performance string label (`Excellent`, `Good`, etc.) is fully pre-processed and ready for Model Training.
