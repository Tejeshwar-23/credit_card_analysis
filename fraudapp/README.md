# Fraud Detection Web Application

This is a Flask-based web application that provides an interface for real-time credit card fraud detection. 

## Features
- Upload/Input transaction data.
- Predict fraud using pre-trained Logistic Regression and Decision Tree models.
- Interactive user interface.

## How to Run

1. **Navigate to this directory**:
   ```bash
   cd fraudapp
   ```

2. **Install Dependencies**:
   Ensure you have Flask and other required libraries installed:
   ```bash
   pip install flask pandas scikit-learn
   ```

3. **Run the Application**:
   ```bash
   python app.py
   ```

4. **Access the App**:
   Open your web browser and go to `http://127.0.0.1:5000`.

## Model Files
- `log_model.pkl`: Trained Logistic Regression model.
- `dt_model.pkl`: Trained Decision Tree model.
- `scaler.pkl`: StandardScaler used for feature normalization.
- `required_features.pkl`: List of features required by the models.
