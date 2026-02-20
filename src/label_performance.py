import pandas as pd
import os

def batting_score(row):
    score = (
        0.40 * row['form_runs_10'] / 50 +        # normalize runs (50 is a good score)
        0.30 * row['form_sr_10'] / 150 +          # normalize SR (150 is excellent in T20)
        0.20 * row['form_boundaries_10'] / 8 +    # boundaries contribution
        0.10 * (1 - row['form_dot_pct_10'])       # penalize dot ball %
    )
    return round(score * 100, 2)

def bowling_score(row):
    score = (
        0.40 * row['form_wickets_10'] / 3 +          # 3 wickets per match = excellent
        0.35 * (1 - min(row['form_economy_10'] / 12, 1)) +  # economy < 6 = excellent
        0.25 * (1 - min(row['form_sr_bowl_10'] / 24, 1))    # SR < 12 = excellent
    )
    return round(score * 100, 2)

def assign_label(score):
    if score >= 75:   return 'Excellent'
    elif score >= 50: return 'Good'
    elif score >= 25: return 'Average'
    else:             return 'Poor'

def main():
    if not os.path.exists("data/processed/batting_form_features.csv"):
        print("Data not found. Run compute_form_features.py first.")
        return
        
    batting_df = pd.read_csv("data/processed/batting_form_features.csv")
    bowling_df = pd.read_csv("data/processed/bowling_form_features.csv")
    
    # Needs to handle missing values or NaNs from rolling windows
    batting_df.fillna(0, inplace=True)
    bowling_df.fillna(0, inplace=True)

    batting_df['performance_score'] = batting_df.apply(batting_score, axis=1)
    batting_df['performance_label'] = batting_df['performance_score'].apply(assign_label)
    
    bowling_df['performance_score'] = bowling_df.apply(bowling_score, axis=1)
    bowling_df['performance_label'] = bowling_df['performance_score'].apply(assign_label)
    
    print("Batting class distribution:")
    print(batting_df['performance_label'].value_counts(normalize=True))
    
    print("Bowling class distribution:")
    print(bowling_df['performance_label'].value_counts(normalize=True))
    
    # Save the labeled datasets
    batting_df.to_csv("data/processed/player_labeled_batting.csv", index=False)
    bowling_df.to_csv("data/processed/player_labeled_bowling.csv", index=False)
    
    # To support the Streamlit app later which expects player_form_features.csv:
    # We can just copy the batting one to it for now (assuming the app primarily looks at batting features as per Phase 10)
    batting_df.to_csv("data/processed/player_form_features.csv", index=False)
    print("Saved labeled datasets to data/processed/")

if __name__ == "__main__":
    main()
