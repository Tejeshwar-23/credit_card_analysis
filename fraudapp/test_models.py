import pandas as pd
import joblib
import os
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

# --- 1. Load your pretrained models ---
# Using absolute paths as provided in the project
DT_MODEL_PATH = "dt_model.pkl"
LOG_MODEL_PATH = "log_model.pkl"
FEATURES_PATH = "required_features.pkl"
SCALER_PATH = "scaler.pkl"

print("Loading models and scaler...")
dt_model = joblib.load(DT_MODEL_PATH)
log_model = joblib.load(LOG_MODEL_PATH)
required_features = joblib.load(FEATURES_PATH)
scaler = joblib.load(SCALER_PATH)
print("Models and sealer loaded successfully.\n")

# --- 2. Load the new dataset ---
dataset_path = "test_data.csv"  # Default to project's test data
if not os.path.exists(dataset_path):
    print(f"File {dataset_path} not found. Please ensure it exists or change the path in the script.")
    exit()

data = pd.read_csv(dataset_path)

# --- 3. Separate features and target ---
# Using normalization to handle different casings as implemented in the web app
df_cols_lower = {col.lower(): col for col in data.columns}

# Identify target column (case-insensitive)
target_col = None
if 'class' in df_cols_lower:
    target_col = df_cols_lower['class']
elif 'target' in df_cols_lower:
    target_col = df_cols_lower['target']

X_test = data.copy()
if target_col:
    X_test = X_test.drop(columns=[target_col])
    y_test = data[target_col]
else:
    y_test = None

# --- 4. Preprocess features ---
# Use the required features the model was trained on
# Fill missing features with 0 for robustness
for feat in required_features:
    feat_lower = feat.lower()
    if feat_lower in df_cols_lower:
        orig_name = df_cols_lower[feat_lower]
        X_test = X_test.rename(columns={orig_name: feat})

X_model_input = X_test.reindex(columns=required_features, fill_value=0)

# Apply Scaling
X_scaled = scaler.transform(X_model_input)
X_scaled_df = pd.DataFrame(X_scaled, columns=required_features)

# Sanity Check
print("--- Model Input Sanity Check (Scaled) ---")
print(X_scaled_df.describe().loc[['mean', 'std', 'max']])

# --- 5. Make predictions ---
print("\nGenerating predictions using Refined Model Roles...")
import numpy as np

# Get raw probabilities
dt_probs = dt_model.predict_proba(X_scaled_df)[:, 1]
log_probs = log_model.predict_proba(X_scaled_df)[:, 1]

# 1. Calibrated Alerting (Logistic Regression Only)
# Prediction thresholds were calibrated to match the observed fraud prevalence 
# of approximately 0.17%, ensuring realistic alert volumes.
target_rate = 0.0017 # 0.17%
log_threshold = np.quantile(log_probs, 1 - target_rate)
data["log_alerts"] = (log_probs >= log_threshold).astype(int)

# 2. Rule Pattern Matching (Decision Tree)
# Used for expert interpretation only (Uncalibrated, High Precision)
dt_threshold = 0.99
data["dt_rule_matches"] = (dt_probs >= dt_threshold).astype(int)

print(f"Calibrated LR Alert Threshold: {log_threshold:.4f}")
print(f"Total Calibrated Alerts (LR): {data['log_alerts'].sum()}")
print(f"Total Rule Matches (DT): {data['dt_rule_matches'].sum()}")

# Keep data frame names consistent for evaluation if needed
data["log_predictions"] = data["log_alerts"]
data["dt_predictions"] = data["dt_rule_matches"]

# --- 6. Evaluate if labels exist ---
if y_test is not None:
    # Ensure y_test is numeric if it contains 'Fraud'/'Legit' strings
    y_numeric = y_test.apply(lambda x: 1 if str(x).lower() in ['1', 'fraud'] else 0)
    
    print("--- Evaluation Results ---")
    print("Decision Tree Accuracy:", accuracy_score(y_numeric, data["dt_predictions"]))
    print("Logistic Regression Accuracy:", accuracy_score(y_numeric, data["log_predictions"]))
    
    print("\nDecision Tree Confusion Matrix:\n", confusion_matrix(y_numeric, data["dt_predictions"]))
    print("\nLogistic Regression Confusion Matrix:\n", confusion_matrix(y_numeric, data["log_predictions"]))
    
    if len(set(y_numeric)) == 2:  # binary classification
        print("\nDecision Tree ROC-AUC:", roc_auc_score(y_numeric, data["dt_predictions"]))
        print("Logistic Regression ROC-AUC:", roc_auc_score(y_numeric, data["log_predictions"]))

# --- 7. Save predictions ---
output_file = "predictions_comparison.csv"
data.to_csv(output_file, index=False)
print(f"\nPredictions saved to {output_file}")
