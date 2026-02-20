import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
import os
import warnings

warnings.filterwarnings('ignore')

def generate_shap_plots():
    print("Generating SHAP explanations...")
    os.makedirs("outputs/plots", exist_ok=True)
    
    if not os.path.exists("models/rf_batsman_classifier.pkl"):
        print("Model not found. Run train.py first.")
        return
        
    # Load Models and Data
    model = joblib.load("models/rf_batsman_classifier.pkl")
    scaler = joblib.load("models/scaler_bat.pkl")
    le = joblib.load("models/label_encoder_bat.pkl")
    
    df = pd.read_csv("data/processed/player_labeled_batting.csv")
    bat_features = ['form_runs_10', 'form_sr_10', 'form_boundaries_10',
                    'form_dot_pct_10', 'form_dismissals_10', 'consistency_score',
                    'matches_played_total', 'recent_50s']
    
    X = df[bat_features].fillna(0)
    X_sc = scaler.transform(X)
    
    explainer = shap.TreeExplainer(model)
    subset_size = min(500, len(X_sc))
    X_sc_sub = X_sc[:subset_size]
    
    try:
        class_idx = list(le.classes_).index('Excellent')
        shap_values = explainer.shap_values(X_sc_sub)
        
        plt.figure()
        if isinstance(shap_values, list):
            vals = shap_values[class_idx]
        else:
            vals = shap_values[:, :, class_idx]
            
        shap.summary_plot(vals, X_sc_sub, feature_names=bat_features, plot_type='bar', show=False)
        plt.title("What drives an 'Excellent' rating?")
        plt.tight_layout()
        plt.savefig("outputs/plots/shap_excellent_drivers.png")
        plt.close()
        print("Saved SHAP summary plot.")
    except Exception as e:
        print(f"Could not generate summary plot: {e}")

    try:
        player_df = df[df['player'].str.contains('Asalanka', na=False)]
        if len(player_df) > 0:
            player_row = X.loc[player_df.index[-1:]]
            player_row_sc = scaler.transform(player_row)
            
            shap_vals_obj = explainer(player_row_sc)
            
            plt.figure()
            if len(shap_vals_obj.shape) == 3:
                shap.plots.waterfall(shap_vals_obj[0, :, class_idx], show=False)
            elif isinstance(shap_vals_obj, list):
                # old shap behavior
                pass
            else:
                shap.plots.waterfall(shap_vals_obj[0], show=False)
                
            plt.title("SHAP Waterfall Explanation")
            plt.tight_layout()
            plt.savefig("outputs/plots/shap_asalanka.png")
            plt.close()
            print("Saved SHAP waterfall plot.")
    except Exception as e:
        print(f"Could not generate waterfall plot: {e}")
        
if __name__ == "__main__":
    generate_shap_plots()
