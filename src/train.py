import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

def train_batsman_model():
    print("Training Batsman Model...")
    df = pd.read_csv("data/processed/player_labeled_batting.csv")
    
    bat_features = ['form_runs_10', 'form_sr_10', 'form_boundaries_10',
                    'form_dot_pct_10', 'form_dismissals_10', 'consistency_score',
                    'matches_played_total', 'recent_50s']
    
    X = df[bat_features].fillna(0)
    y = df['performance_label']
    
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )
    
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    
    # Baseline Model
    lr = LogisticRegression(max_iter=1000, class_weight='balanced')
    lr.fit(X_train_sc, y_train)
    
    # Hyperparameter tuning for Random Forest
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'max_features': ['sqrt']
    }
    
    print("Tuning RandomForest Hyperparameters (Batsman)...")
    grid = GridSearchCV(
        RandomForestClassifier(class_weight='balanced', random_state=42),
        param_grid, cv=3, scoring='f1_weighted', n_jobs=-1
    )
    grid.fit(X_train_sc, y_train)
    best_rf = grid.best_estimator_
    print("Best params:", grid.best_params_)
    
    # Save models
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_rf, "models/rf_batsman_classifier.pkl")
    joblib.dump(lr, "models/lr_batsman_baseline.pkl")
    joblib.dump(scaler, "models/scaler_bat.pkl")
    joblib.dump(le, "models/label_encoder_bat.pkl")
    
    return best_rf, lr, scaler, le, X_test_sc, y_test, bat_features

def train_bowler_model():
    print("\nTraining Bowler Model...")
    df = pd.read_csv("data/processed/player_labeled_bowling.csv")
    
    bowl_features = ['form_wickets_10', 'form_economy_10', 'form_sr_bowl_10',
                     'form_dot_pct_bowl_10', 'form_maidens_10', 'consistency_wickets',
                     'recent_3fers']
    
    X = df[bowl_features].fillna(0)
    y = df['performance_label']
    
    le = LabelEncoder()
    
    try:
        y_enc = le.fit_transform(y)
        X_train, X_test, y_train, y_test = train_test_split(
             X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
        )
    except ValueError:
        print("Warning: stratify failed for bowlers due to rare classes. Falling back to non-stratified split.")
        X_train, X_test, y_train, y_test = train_test_split(
             X, y_enc, test_size=0.2, random_state=42
        )
    
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    
    # Baseline Model
    lr = LogisticRegression(max_iter=1000, class_weight='balanced')
    lr.fit(X_train_sc, y_train)
    
    # Hyperparameter tuning for Random Forest
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'max_features': ['sqrt']
    }
    
    print("Tuning RandomForest Hyperparameters (Bowler)...")
    grid = GridSearchCV(
        RandomForestClassifier(class_weight='balanced', random_state=42),
        param_grid, cv=3, scoring='f1_weighted', n_jobs=-1
    )
    grid.fit(X_train_sc, y_train)
    best_rf = grid.best_estimator_
    print("Best params:", grid.best_params_)
    
    # Save models
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_rf, "models/rf_bowler_classifier.pkl")
    joblib.dump(lr, "models/lr_bowler_baseline.pkl")
    joblib.dump(scaler, "models/scaler_bowl.pkl")
    joblib.dump(le, "models/label_encoder_bowl.pkl")
    
    return best_rf, lr, scaler, le, X_test_sc, y_test, bowl_features

def main():
    train_batsman_model()
    train_bowler_model()
    print("\nModels trained and saved successfully.")

if __name__ == "__main__":
    main()
