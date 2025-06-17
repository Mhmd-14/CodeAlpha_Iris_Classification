import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join('..', 'src')))
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from src.train_model import train_logistic_regression
from src.evaluate_train import evaluate_model_train 
from tests.test_utils import get_sample_df

# Sample data
sample_df = get_sample_df()

@pytest.fixture
def trained_model_and_data():
    """
    Fixture that trains a simple logistic regression model on sample data
    and returns the model and training set.
    
    Returns:
        Tuple: (trained_model, X_train, y_train)
    """
    feature_cols = ['feature1', 'feature2']
    target_col = 'target'
    X_train, _, y_train, _ = train_test_split(
        sample_df[feature_cols], sample_df[target_col], test_size=0.5, random_state=42
    )
    model = train_logistic_regression(X_train, y_train)
    return model, X_train, y_train


def test_evaluate_model_train_output(capsys, trained_model_and_data):
    """
    Test that evaluate_model_train prints accuracy and classification report
    to standard output without raising exceptions.
    """
    model, X_train, y_train = trained_model_and_data

    # Call function and capture stdout
    evaluate_model_train(model, X_train, y_train)
    captured = capsys.readouterr()

    # Check that accuracy string and classification report appear in output
    assert "Accuracy on training set:" in captured.out
    assert "Classification Report:" in captured.out
    assert "precision" in captured.out  # basic sanity check for report content
