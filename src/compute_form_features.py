import pandas as pd
import os

def compute_batting_form(df):
    df = df.sort_values(['player', 'match_date'])
    rolling_cols = ['runs_scored', 'strike_rate', 'boundaries', 'dot_ball_pct', 'dismissed']
    
    for col in rolling_cols:
        new_col = col
        if col == 'runs_scored': new_col = 'runs'
        if col == 'strike_rate': new_col = 'sr'
        if col == 'dot_ball_pct': new_col = 'dot_pct'
        
        df[f'form_{new_col}_10'] = (
            df.groupby('player')[col]
            .transform(lambda x: x.rolling(10, min_periods=3).mean())
        )
        
    df['consistency_score'] = df.groupby('player')['runs_scored'].transform(lambda x: x.rolling(10, min_periods=3).std())
    df['matches_played_total'] = df.groupby('player').cumcount() + 1
    df['recent_50s'] = df.groupby('player')['runs_scored'].transform(
        lambda x: (x >= 50).rolling(10, min_periods=3).sum()
    )
    
    df.rename(columns={'form_dismissed_10': 'form_dismissals_10'}, inplace=True)
    return df

def compute_bowling_form(df):
    df = df.sort_values(['player', 'match_date'])
    rolling_cols = ['wickets_taken', 'economy_rate', 'bowling_strike_rate', 'dot_ball_pct']
    
    for col in rolling_cols:
        new_col = col
        if col == 'wickets_taken': new_col = 'wickets'
        if col == 'economy_rate': new_col = 'economy'
        if col == 'bowling_strike_rate': new_col = 'sr_bowl'
        if col == 'dot_ball_pct': new_col = 'dot_pct_bowl'
            
        df[f'form_{new_col}_10'] = (
            df.groupby('player')[col]
            .transform(lambda x: x.rolling(10, min_periods=3).mean())
        )
        
    df['form_maidens_10'] = 0 # Maidens not extracted directly in Phase 3 script

    df['consistency_wickets'] = df.groupby('player')['wickets_taken'].transform(lambda x: x.rolling(10, min_periods=3).std())
    df['recent_3fers'] = df.groupby('player')['wickets_taken'].transform(
        lambda x: (x >= 3).rolling(10, min_periods=3).sum()
    )
    return df

def main():
    if not os.path.exists("data/processed/player_batting_stats.csv"):
        print("Data not found. Run extract_player_stats.py first.")
        return
        
    batting_df = pd.read_csv("data/processed/player_batting_stats.csv")
    bowling_df = pd.read_csv("data/processed/player_bowling_stats.csv")
    
    batting_df['match_date'] = pd.to_datetime(batting_df['match_date'])
    bowling_df['match_date'] = pd.to_datetime(bowling_df['match_date'])
    
    bat_form = compute_batting_form(batting_df)
    bowl_form = compute_bowling_form(bowling_df)
    
    bat_form.to_csv("data/processed/batting_form_features.csv", index=False)
    bowl_form.to_csv("data/processed/bowling_form_features.csv", index=False)
    print("Saved form features to data/processed/")

if __name__ == "__main__":
    main()
