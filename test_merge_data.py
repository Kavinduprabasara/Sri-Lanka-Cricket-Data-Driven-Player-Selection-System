import pandas as pd
import numpy as np

def load_real_player_data():
    df_bat = pd.read_csv('data/processed/player_labeled_batting.csv')
    df_bat['match_date'] = pd.to_datetime(df_bat['match_date'])
    
    df_bowl = pd.read_csv('data/processed/player_labeled_bowling.csv')
    df_bowl['match_date'] = pd.to_datetime(df_bowl['match_date'])
    
    # Merge
    df = pd.merge(df_bat, df_bowl, on=['match_id', 'match_date', 'player'], how='outer', suffixes=('_bat', '_bowl'))
    
    # Fill NAs
    for col in df.columns:
        if ('runs' in col or 'wickets' in col or 'balls' in col or 'boundaries' in col or 'dismissed' in col):
            df[col] = df[col].fillna(0)
    
    # Combine Performance Scores
    # For now, let's use the actual labels generated in the pipeline.
    # We can take the max or average of bat/bowl performance scores.
    df['performance_score_bat'] = df['performance_score_bat'].fillna(0)
    df['performance_score_bowl'] = df['performance_score_bowl'].fillna(0)
    df['performance_score'] = df[['performance_score_bat', 'performance_score_bowl']].max(axis=1)
    
    # Assign labels based on score
    conditions = [
        (df['performance_score'] >= 75),
        (df['performance_score'] >= 50),
        (df['performance_score'] >= 25)
    ]
    choices = ['Excellent', 'Good', 'Average']
    df['performance_label'] = np.select(conditions, choices, default='Poor')
    
    # Needs opponent, venue, match_result, role, player_of_match (Mock for now to satisfy UI)
    np.random.seed(42)
    opponents = ["India", "Australia", "England", "Pakistan", "South Africa", "New Zealand", "Bangladesh", "West Indies", "Afghanistan"]
    
    df['opponent'] = np.random.choice(opponents, size=len(df))
    df['venue'] = np.random.choice(["Home", "Away", "Neutral"], size=len(df), p=[0.4, 0.4, 0.2])
    df['match_result'] = np.random.choice(["Win", "Loss"], size=len(df), p=[0.45, 0.55])
    df['player_of_match'] = (df['performance_score'] > 80) & (df['match_result'] == 'Win')
    
    # Load PLAYER_ROLES from select_team.py or define it
    # We can just define a fallback
    
    df['role'] = 'allrounder' # Fallback
    
    print(df.head())
    return df

load_real_player_data()
