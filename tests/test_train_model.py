import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join('..', 'src')))
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from src.train_model import split_data, train_logistic_regression
from tests.test_utils import get_sample_df
import math

# Sample dataset used for testing
sample_df = get_sample_df()

def test_split_data():
    """
    Test that split_data correctly splits features and target into training and test sets.
    """
    X_train, X_test, y_train, y_test = split_data(sample_df, target_col='target')

    # Check shapes
    total_samples = sample_df.shape[0]
    expected_test_size = math.ceil(total_samples * 0.2)

    assert len(X_test) == expected_test_size
    assert len(X_train) == total_samples - expected_test_size
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)

    # Check that target column is not in features
    assert 'target' not in X_train.columns
    assert 'target' not in X_test.columns


def test_train_logistic_regression():
    """
    Test that train_logistic_regression trains a model and returns a fitted LogisticRegression instance.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        sample_df.drop(columns='target'),
        sample_df['target'],
        test_size=0.2,
        random_state=42
    )

    model = train_logistic_regression(X_train, y_train)

    # Check model is instance of LogisticRegression
    assert isinstance(model, LogisticRegression)

    # Check that the model is fitted and can predict
    preds = model.predict(X_test)
    assert len(preds) == len(y_test)
