import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

def evaluate_model(model_path, data_path, scaler_path, le_path, features, title, out_png):
    print(f"\nEvaluating {title}...")
    
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}. Run train.py first.")
        return
        
    best_rf = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    le = joblib.load(le_path)
    
    df = pd.read_csv(data_path)
    X = df[features].fillna(0)
    y = df['performance_label']
    
    y_enc = le.transform(y)
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_enc, test_size=0.2, random_state=42
        )
        
    X_test_sc = scaler.transform(X_test)
    y_pred = best_rf.predict(X_test_sc)
    
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    os.makedirs("outputs/plots", exist_ok=True)
    disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred,
                                             display_labels=le.classes_,
                                             cmap='Blues')
    plt.title(f"{title} Classification — Confusion Matrix")
    plt.savefig(out_png)
    plt.close()
    print(f"Saved confusion matrix to {out_png}")

def main():
    bat_features = ['form_runs_10', 'form_sr_10', 'form_boundaries_10',
                    'form_dot_pct_10', 'form_dismissals_10', 'consistency_score',
                    'matches_played_total', 'recent_50s']
    
    evaluate_model(
        "models/rf_batsman_classifier.pkl",
        "data/processed/player_labeled_batting.csv",
        "models/scaler_bat.pkl",
        "models/label_encoder_bat.pkl",
        bat_features,
        "Batsman Performance",
        "outputs/plots/confusion_matrix_bat.png"
    )
    
    bowl_features = ['form_wickets_10', 'form_economy_10', 'form_sr_bowl_10',
                     'form_dot_pct_bowl_10', 'form_maidens_10', 'consistency_wickets',
                     'recent_3fers']
                     
    evaluate_model(
        "models/rf_bowler_classifier.pkl",
        "data/processed/player_labeled_bowling.csv",
        "models/scaler_bowl.pkl",
        "models/label_encoder_bowl.pkl",
        bowl_features,
        "Bowler Performance",
        "outputs/plots/confusion_matrix_bowl.png"
    )

if __name__ == "__main__":
    main()
