# 🧠 Model Training

## Algorithm Choice

This project utilizes a **Random Forest Classifier**. While simpler models like Logistic Regression provide strong baselines, Random Forests are perfectly suited for our multidimensional performance data:

1. **Non-Linear Interactions:** In sports, stats don't scale linearly. A strike rate of 150 vs 130 defines an elite T20 player entirely differently than a jump from 110 to 130. Tree-based splits capture these thresholds mathematically.
2. **Robustness:** Forests natively handle outliers (like a random 100-run knock in 1 bad year) by creating aggregated ensemble logic.
3. **No Extensive Scaling Requirements:** Unlike SVMs or deep learning, tree models don't structurally require complex normalization, though we still use standard scaling as best practice.

## Separate Models for Roles

A bowler who takes 3 wickets and scores 0 runs is an `Excellent` performer. A batsman who scores 0 runs is `Poor`. Since the features dictating "Good" or "Poor" performance are fundamentally different between disciplines, rolling all players into a single model leads to conflicting weight assignments.

To resolve this, the system isolates two independent pipelines:

1. `rf_batsman_classifier.pkl` (Trained exclusively on `form_runs_10`, `form_sr_10`, boundaries, dots, etc.)
2. `rf_bowler_classifier.pkl` (Trained exclusively on `form_wickets_10`, `form_economy_10`, strike rates, etc.)

## Model Configuration & Tuning

- **Standardization:** Features (`X_train`) are scaled using `StandardScaler()` to standardise distributions (subtracting the mean and scaling to unit variance).
- **Stratified Splits:** We utilize SMOTE or `stratify=y_bat_enc` in our `train_test_split(test_size=0.2)` to ensure rare `Excellent` labels aren't clustered purely into training or testing.
- **Handling Class Imbalance:** In cricket, `Average` and `Poor` players vastly outnumber `Excellent` ones. `class_weight='balanced'` algorithmically adjusts weights inversely proportional to class frequencies to aggressively penalize missing an elite player.

### Hyperparameter Tuning (GridSearchCV)

We execute a Cartesian grid search over parameters to select the optimal model architecture before deployment:

- `n_estimators`: [100, 200, 300] — How many distinct decision trees vote on the classification?
- `max_depth`: [None, 10, 20] — How complex can the rules get before overfitting?
- `min_samples_split`: The minimum number of samples required to split an internal node.
- `max_features`: The number of features considered when looking for the best split (`sqrt`, `log2`).

The pipeline outputs the absolute best-performing permutation (measured by weighted F-1 scores) and saves the serialized object for real-time frontend inference (`joblib.dump(best_rf, "models/...")`).
