import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join('..', 'src')))
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split
from src.train_model import train_logistic_regression 
from src.evaluate_test import evaluate_model  
from tests.test_utils import get_sample_df

# Sample dataset
sample_df = get_sample_df()


@pytest.fixture
def trained_model_and_test_data():
    """
    Fixture that trains a logistic regression model and returns the trained model,
    test features, and test labels.

    Returns:
        Tuple: (model, X_test, y_test)
    """
    feature_cols = ['feature1', 'feature2']
    target_col = 'target'

    X_train, X_test, y_train, y_test = train_test_split(
        sample_df[feature_cols], sample_df[target_col],
        test_size=0.5, random_state=42
    )

    model = train_logistic_regression(X_train, y_train)
    return model, X_test, y_test


def test_evaluate_model_prints_metrics(capsys, trained_model_and_test_data):
    """
    Test that evaluate_model prints the accuracy and classification report
    to standard output.
    """
    model, X_test, y_test = trained_model_and_test_data

    evaluate_model(model, X_test, y_test)
    captured = capsys.readouterr()

    assert "Accuracy on testing set:" in captured.out
    assert "Classification Report:" in captured.out
    assert "precision" in captured.out  # sanity check for classification report content
