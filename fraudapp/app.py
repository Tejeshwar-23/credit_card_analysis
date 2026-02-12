from flask import Flask, request, jsonify, render_template
import pandas as pd
import os
from model_loader import load_model, load_features

app = Flask(__name__)

# Config
DT_MODEL_PATH = "dt_model.pkl"
LOG_MODEL_PATH = "log_model.pkl"
FEATURES_PATH = "required_features.pkl"
SCALER_PATH = "scaler.pkl"

# Load models globally
models = {}
try:
    models['dt'] = load_model(DT_MODEL_PATH)
    models['log'] = load_model(LOG_MODEL_PATH)
    required_features = load_features(FEATURES_PATH)
    scaler = load_model(SCALER_PATH)
except Exception as e:
    print(f"Error loading models or scaler: {e}")
    models = None
    required_features = None
    scaler = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not models or not models.get('dt') or not scaler:
        return jsonify({'error': 'Models or Scaler not loaded on server'}), 500
        
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
        
    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'Please upload a CSV file'}), 400
        
    try:
        df = pd.read_csv(file)
        
        # Flexible Column Matching (Case-insensitive)
        df_cols_lower = {col.lower(): col for col in df.columns}
        
        if required_features:
            # Reindex to match requested columns, filling missing with 0
            # Normalize column names first
            for feat in required_features:
                feat_lower = feat.lower()
                if feat_lower in df_cols_lower:
                    orig_name = df_cols_lower[feat_lower]
                    df = df.rename(columns={orig_name: feat})
            
            # Robust reindexing
            X_df = df.reindex(columns=required_features, fill_value=0)
            
            # Sanity Check
            print("\n--- Model Input Sanity Check (Pre-Scaling) ---")
            print(X_df.describe().loc[['mean', 'std', 'max']])
            
            # Scaling
            X_scaled = scaler.transform(X_df)
            X_scaled_df = pd.DataFrame(X_scaled, columns=required_features)
            
            print("\n--- Model Input Sanity Check (Post-Scaling) ---")
            print(X_scaled_df.describe().loc[['mean', 'std', 'max']])
            
            X = X_scaled_df
        else:
            # Fallback
            X = df.drop(columns=['Class', 'class'], errors='ignore')

        # Predictions
        # Get raw probabilities
        dt_probs = models['dt'].predict_proba(X)[:, 1]
        log_probs = models['log'].predict_proba(X)[:, 1]

        # Calibrated Alerting (Logistic Regression only)
        # Prediction thresholds were calibrated to match the observed fraud prevalence 
        # of approximately 0.17%, ensuring realistic alert volumes.
        import numpy as np
        target_rate = 0.0017 # 0.17% prevalence
        log_threshold = float(np.quantile(log_probs, 1 - target_rate))
        log_preds = (log_probs >= log_threshold).astype(int)

        # Rule Pattern Matching (Decision Tree)
        # DT is used for rule interpretation only (Uncalibrated, High Precision)
        dt_threshold = 0.99 
        dt_rules_matched = (dt_probs >= dt_threshold).astype(int)
        
        # Summary Stats
        total_rows = len(df)
        log_alerts = int(sum(log_preds)) # LR is the alerting source
        
        # Combine results safely
        df['log_alert'] = log_preds
        df['dt_rules'] = dt_rules_matched
        
        # Detect Class column for comparison (case-insensitive)
        has_ground_truth = False
        real_status_col = None
        if 'class' in df_cols_lower:
            real_status_col = df_cols_lower['class']
            has_ground_truth = True
        elif 'target' in df_cols_lower:
            real_status_col = df_cols_lower['target']
            has_ground_truth = True

        # Filter for display (Report only detected frauds)
        # We show rows flagged by either, but LR is the professional volume source
        fraud_df = df[(df['log_alert'] == 1) | (df['dt_rules'] == 1)].copy()
        
        results = []
        display_limit = 1000
        for idx, row in fraud_df.head(display_limit).iterrows():
            res_item = {
                'row_index': int(idx) + 1,
                'risk_rank': 'High Risk' if row['log_alert'] == 1 else 'Standard',
                'rule_match': 'Rule Match' if row['dt_rules'] == 1 else 'No Match'
            }
            if has_ground_truth:
                res_item['real_status'] = 'Fraud' if row[real_status_col] == 1 else 'Legit'
            results.append(res_item)
            
        return jsonify({
            'summary': {
                'total_analyzed': total_rows,
                'calibrated_alerts': log_alerts,
                'fraud_rate_applied': "0.17%",
                'log_threshold': round(log_threshold, 4),
                'dt_role': "Rule Interpretation"
            },
            'has_ground_truth': has_ground_truth,
            'results': results
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
