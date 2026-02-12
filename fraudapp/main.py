from model_loader import load_model
from model_inspector import inspect_model
import os

def main():
    # Define paths
    dt_path = r"C:\Users\Administrator\Desktop\credit card fraud\dt_model.pkl"
    log_path = r"C:\Users\Administrator\Desktop\credit card fraud\log_model.pkl"
    
    print("Starting Model Inspection...")
    
    # Load and inspect Decision Tree Model
    try:
        dt_model = load_model(dt_path)
        inspect_model(dt_model, "Decision Tree Model")
    except Exception as e:
        print(f"Failed to process Decision Tree: {e}")
        
    # Load and inspect Logistic Regression Model
    try:
        log_model = load_model(log_path)
        inspect_model(log_model, "Logistic Regression Model")
    except Exception as e:
        print(f"Failed to process Logistic Regression: {e}")

if __name__ == "__main__":
    main()
