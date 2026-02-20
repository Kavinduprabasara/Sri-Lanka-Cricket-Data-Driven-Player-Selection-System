import pandas as pd
import glob
import os
from tqdm import tqdm

def load_data():
    all_balls = []
    # Load T20I matches
    print("Loading T20I CSVs...")
    for f in tqdm(glob.glob("data/raw/t20s_male_csv2/*.csv")):
        if '_info' not in f:
            df = pd.read_csv(f, low_memory=False)
            all_balls.append(df)
            
    # Load LPL matches
    print("Loading LPL CSVs...")
    for f in tqdm(glob.glob("data/raw/lpl_male_csv2/*.csv")):
        if '_info' not in f:
            df = pd.read_csv(f, low_memory=False)
            all_balls.append(df)

    balls_df = pd.concat(all_balls, ignore_index=True)
    
    # Convert start_date to datetime to allow sorting later
    balls_df['start_date'] = pd.to_datetime(balls_df['start_date'])
    return balls_df

def get_sl_players(balls_df):
    sl_batsmen = balls_df[balls_df['batting_team'] == 'Sri Lanka']['striker'].unique()
    sl_bowlers = balls_df[balls_df['bowling_team'] == 'Sri Lanka']['bowler'].unique()
    sl_players = set(sl_batsmen) | set(sl_bowlers)
    return list(sl_players)

def extract_batting_stats(balls_df, match_id, player):
    """Extract batting stats for one player in one match."""
    bat = balls_df[
        (balls_df['match_id'] == match_id) &
        (balls_df['striker'] == player)
    ]

    if len(bat) == 0:
        return None  # did not bat

    runs = bat['runs_off_bat'].sum()
    balls = len(bat[bat['wides'].isna()])  # exclude wides
    boundaries = ((bat['runs_off_bat'] == 4) | (bat['runs_off_bat'] == 6)).sum()
    dismissed = int(bat['player_dismissed'].notna().any())
    dot_balls = (bat['runs_off_bat'] == 0).sum()
    match_date = bat['start_date'].iloc[0]

    return {
        'match_id': match_id,
        'match_date': match_date,
        'player': player,
        'runs_scored': runs,
        'balls_faced': balls,
        'strike_rate': (runs / balls * 100) if balls > 0 else 0,
        'boundaries': int(boundaries),
        'dot_ball_pct': (dot_balls / balls) if balls > 0 else 0,
        'dismissed': dismissed,
    }

def extract_bowling_stats(balls_df, match_id, player):
    """Extract bowling stats for one player in one match."""
    bowl = balls_df[
        (balls_df['match_id'] == match_id) &
        (balls_df['bowler'] == player)
    ]

    if len(bowl) == 0:
        return None  # did not bowl

    wickets = bowl['wicket_type'].notna().sum()
    # Don't count run outs as bowler's wickets
    wickets -= bowl['wicket_type'].isin(['run out']).sum()

    runs = bowl['runs_off_bat'].sum() + bowl['extras'].fillna(0).sum()
    balls = len(bowl[bowl['wides'].isna()])
    overs = balls / 6
    dot_balls = (bowl['runs_off_bat'] == 0).sum()
    match_date = bowl['start_date'].iloc[0]

    return {
        'match_id': match_id,
        'match_date': match_date,
        'player': player,
        'wickets_taken': int(wickets),
        'runs_conceded': int(runs),
        'overs_bowled': round(overs, 2),
        'economy_rate': round(runs / overs, 2) if overs > 0 else 0,
        'bowling_strike_rate': round(balls / wickets, 2) if wickets > 0 else 999,
        'dot_ball_pct': round(dot_balls / balls, 2) if balls > 0 else 0,
    }

def main():
    if not os.path.exists("data/processed"):
        os.makedirs("data/processed")
        
    print("Loading data...")
    balls_df = load_data()
    
    print("Identifying Sri Lanka players...")
    sl_players = get_sl_players(balls_df)
    print(f"Unique Sri Lanka players found: {len(sl_players)}")
    
    # We only care about matches where an SL player played
    sl_matches = balls_df[
        (balls_df['striker'].isin(sl_players)) | 
        (balls_df['bowler'].isin(sl_players))
    ]['match_id'].unique()
    
    print(f"Total matches to process: {len(sl_matches)}")
    
    batting_records = []
    bowling_records = []
    
    print("Extracting per-match stats for Sri Lanka players...")
    # To speed up, we can iterate over grouped data
    grouped = balls_df[balls_df['match_id'].isin(sl_matches)].groupby('match_id')
    
    for match_id, match_df in tqdm(grouped, total=len(sl_matches)):
        # Further optimization: only check players who were actually in this match
        match_strikers = set(match_df['striker'].unique()).intersection(sl_players)
        match_bowlers = set(match_df['bowler'].unique()).intersection(sl_players)
        
        for player in match_strikers:
            bat_stats = extract_batting_stats(match_df, match_id, player)
            if bat_stats:
                batting_records.append(bat_stats)
                
        for player in match_bowlers:
            bowl_stats = extract_bowling_stats(match_df, match_id, player)
            if bowl_stats:
                bowling_records.append(bowl_stats)
                
    print("Saving processed data...")
    batting_df = pd.DataFrame(batting_records)
    bowling_df = pd.DataFrame(bowling_records)
    
    batting_df.to_csv("data/processed/player_batting_stats.csv", index=False)
    bowling_df.to_csv("data/processed/player_bowling_stats.csv", index=False)
    
    print(f"Saved {len(batting_df)} batting records and {len(bowling_df)} bowling records.")

if __name__ == "__main__":
    main()
