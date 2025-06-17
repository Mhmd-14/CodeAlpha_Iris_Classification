# src/train_model.py

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score


def evaluate_model_train(model: LogisticRegression, X_train, y_train) -> None:
    """
    Evaluate model performance and print classification report and accuracy.

    Args:
        model (LogisticRegression): Trained model.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): True test labels.

    Returns:
        None
    """
    y_pred_train = model.predict(X_train)
    acc = accuracy_score(y_train, y_pred_train)
    print(f"Accuracy on training set: {acc:.4f}")
    print("\nClassification Report:\n", classification_report(y_train, y_pred_train))
