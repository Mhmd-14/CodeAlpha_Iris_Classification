# src/train_model.py
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from preprocessing import load_raw_data,preprocess_data, save_processed_data
from train_model import split_data, train_logistic_regression
from evaluate_train import evaluate_model_train
from evaluate_test import evaluate_model

def run_training_pipeline(raw_data_path: str, processed_data_path: str, target_col: str = 'Species') -> LogisticRegression:
    
    """
    Complete training pipeline: load data, split, train, and evaluate logistic regression.

    Args:
        filepath (str): Path to raw CSV file.
        target_col (str, optional): Name of target column. Defaults to 'target'.

    Returns:
        LogisticRegression: Trained logistic regression model.
    """
    print("Starting full pipeline...")

    # 1. Load raw data
    df_raw = load_raw_data(raw_data_path)

    # 2. Preprocess data (encodes target, scales features, saves scaler inside)
    df_processed = preprocess_data(df_raw, target_col=target_col)

    # 3. Save processed data for records
    save_processed_data(df_processed, processed_data_path)
    print(f"Processed data saved to {processed_data_path}")

    # 4. Split Data
    X_train, X_test, y_train, y_test = split_data(df_processed, target_col=target_col)

    # 5. Train Model
    model = train_logistic_regression(X_train, y_train)

    # 6. Evaluate Model
    evaluate_model_train(model,X_train,y_train)
    evaluate_model(model, X_test, y_test)

   # 7. Save trained model
    model_path = '../models/logistic_model.pkl'
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

    return model
