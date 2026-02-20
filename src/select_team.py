import pandas as pd
import json

PLAYER_ROLES = {
    'P Nissanka': 'opener',
    'Kusal Mendis': 'opener_wk',
    'KIC Asalanka': 'middle_order',
    'PHKD Mendis': 'middle_order',
    'S Samarawickrama': 'middle_order_wk',
    'MD Shanaka': 'allrounder',
    'PWH de Silva': 'allrounder_spin',
    'DM de Silva': 'allrounder',
    'M Theekshana': 'spinner',
    'M Pathirana': 'pacer',
    'PVD Chameera': 'pacer',
    'D Madushanka': 'pacer',
    'N Thushara': 'pacer',
    'DN Wellalage': 'allrounder_spin',
    'MDKJ Perera': 'opener_wk',
    'PBB Rajapaksa': 'middle_order',
    'AD Mathews': 'allrounder',
    'L Kumara': 'pacer'
}

def load_player_ratings():
    bat_df = pd.read_csv("data/processed/player_labeled_batting.csv")
    bowl_df = pd.read_csv("data/processed/player_labeled_bowling.csv")
    
    bat_df['match_date'] = pd.to_datetime(bat_df['match_date'])
    bowl_df['match_date'] = pd.to_datetime(bowl_df['match_date'])
    
    # Filter active players (played within the last 365 days of the latest match date in the dataset)
    if not bat_df.empty:
        latest_date = max(bat_df['match_date'].max(), bowl_df['match_date'].max())
        cutoff_date = latest_date - pd.Timedelta(days=365)
        
        active_batters = bat_df[bat_df['match_date'] >= cutoff_date]['player'].unique()
        active_bowlers = bowl_df[bowl_df['match_date'] >= cutoff_date]['player'].unique()
        active_players = set(active_batters) | set(active_bowlers)
    else:
        active_players = set(PLAYER_ROLES.keys())

    bat_last = bat_df.sort_values('match_date').groupby('player').last().reset_index()
    bowl_last = bowl_df.sort_values('match_date').groupby('player').last().reset_index()
    
    bat_ratings = bat_last.set_index('player')['performance_score'].to_dict()
    bowl_ratings = bowl_last.set_index('player')['performance_score'].to_dict()
    return bat_ratings, bowl_ratings, active_players

def select_best_xi(batting_ratings, bowling_ratings, player_roles, active_players):
    selected = []
    
    # Filter roles dict to only include active players
    active_roles = {p: r for p, r in player_roles.items() if p in active_players}
    
    openers = [p for p, r in active_roles.items() if 'opener' in r]
    openers_sorted = sorted(openers, key=lambda p: batting_ratings.get(p, 0), reverse=True)
    selected.extend(openers_sorted[:2])
    
    middle = [p for p, r in active_roles.items() if 'middle_order' in r]
    middle = [p for p in middle if p not in selected]
    middle_sorted = sorted(middle, key=lambda p: batting_ratings.get(p, 0), reverse=True)
    selected.extend(middle_sorted[:3])
    
    allrounders = [p for p, r in active_roles.items() if 'allrounder' in r]
    allrounder_sorted = sorted(allrounders, key=lambda p: batting_ratings.get(p, 0) + bowling_ratings.get(p, 0), reverse=True)
    selected.extend(allrounder_sorted[:2])
    
    spinners = [p for p, r in active_roles.items() if 'spin' in r and p not in selected]
    spinners_sorted = sorted(spinners, key=lambda p: bowling_ratings.get(p, 0), reverse=True)
    if spinners_sorted:
        selected.append(spinners_sorted[0])
    
    pacers = [p for p, r in active_roles.items() if 'pacer' in r and p not in selected]
    pacers_sorted = sorted(pacers, key=lambda p: bowling_ratings.get(p, 0), reverse=True)
    selected.extend(pacers_sorted[:3])
    
    # Fill remaining spots if needed with top rated remaining players
    remaining = [p for p in active_roles.keys() if p not in selected]
    remaining_sorted = sorted(remaining, key=lambda p: batting_ratings.get(p, 0) + bowling_ratings.get(p, 0), reverse=True)
    
    while len(selected) < 11 and remaining_sorted:
        selected.append(remaining_sorted.pop(0))
        
    return selected[:11]

if __name__ == "__main__":
    bat_ratings, bowl_ratings = load_player_ratings()
    xi = select_best_xi(bat_ratings, bowl_ratings, PLAYER_ROLES)
    print("Recommended Playing XI based on recent form:")
    for i, player in enumerate(xi, 1):
        role = PLAYER_ROLES.get(player, 'Unknown')
        bat_sc = bat_ratings.get(player, 0)
        bowl_sc = bowl_ratings.get(player, 0)
        print(f"{i}. {player} ({role}) - Bat: {bat_sc:.1f} | Bowl: {bowl_sc:.1f}")
