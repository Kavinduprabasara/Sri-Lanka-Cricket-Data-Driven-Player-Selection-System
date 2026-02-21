# 📊 Dataset Documentation

## Overview

The Sri Lanka Cricket Data-Driven Player Selection System leverages high-quality, open-source international cricket data to evaluate player performance objectively. Instead of relying on proprietary or hidden scouting data, we use strictly mathematical ball-by-ball records from the global standard in cricket datasets: **Cricsheet.org**.

## Source & Scope

- **Source:** [Cricsheet Downloads (JSON & CSV formats)](https://cricsheet.org/downloads/)
- **Data Series:**
  - **T20 Internationals (Men):** All historical men's T20I matches, capturing international-level pressure.
  - **Lanka Premier League (LPL):** High-level domestic franchise cricket, excellent for evaluating emerging Sri Lankan talent who haven't yet secured a permanent spot in the national T20I squad.
- **Filtering Scope:** The data is specifically filtered to focus on active Sri Lankan national players (extracting players where `batting_team` == 'Sri Lanka' or `bowling_team` == 'Sri Lanka').

## File Structure & Formats

Cricsheet provides match records in the highly structured "Ashwin" CSV format. Every match is represented by two distinct files:

### 1. Ball-by-Ball File (`<match_id>.csv`)

This file is the core engine of the system. It records every single delivery bowled in a match, mapping exactly what happened. This granularity is what allows our system to calculate advanced metrics like dot-ball percentages, boundaries hit, and bowling strike rates.

**Key Columns:**

```csv
match_id, season, start_date, venue, innings, ball,
batting_team, bowling_team, striker, non_striker, bowler,
runs_off_bat, extras, wides, noballs, byes, legbyes,
wicket_type, player_dismissed, other_wicket_type, other_player_dismissed
```

### 2. Match Info File (`<match_id>_info.csv`)

Provides the metadata wrapper for the match, crucial for aggregating team-level statistics and understanding match outcomes (e.g. rolling win rates).

**Key Columns:**

```csv
match_id, season, date, venue, team1, team2,
toss_winner, toss_decision, winner, player_of_match
```

## Why Ball-by-Ball Data?

Traditional scorecards only provide the total runs a player scored or the total wickets they took. Ball-by-ball data allows our Machine Learning system to look deeper into the _context_ of a player's form:

- Did a batter score 30 runs off 15 balls (excellent impact) or 30 runs off 35 balls (poor impact)?
- Did a bowler bowl 12 dot balls in their 4 overs (excellent pressure building), forcing a wicket at the other end?
- What is the frequency of boundaries for a specific player?

By loading this data, we transform raw cricket events into quantifiable metrics over rolling match windows, creating the foundation for our predictive Random Forest pipeline.
