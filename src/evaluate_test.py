# src/train_model.py

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score


def evaluate_model(model: LogisticRegression, X_test, y_test) -> None:
    """
    Evaluate model performance and print classification report and accuracy.

    Args:
        model (LogisticRegression): Trained model.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): True test labels.

    Returns:
        None
    """
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy on testing set: {acc:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
