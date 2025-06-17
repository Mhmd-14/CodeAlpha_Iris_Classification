# src/train_model.py
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def split_data(df: pd.DataFrame,target_col: str, test_size: float = 0.2, random_state: int = 42):
    """
    Split dataset into training and testing sets.

    Args:
        df (pd.DataFrame): Dataframe containing features and target.
        target_col (str): Target column name.
        test_size (float, optional): Proportion of test data. Defaults to 0.2.
        random_state (int, optional): Random seed for reproducibility. Defaults to 42.

    Returns:
        Tuple: X_train, X_test, y_train, y_test
    """
    feature_cols = df.columns.drop(target_col).tolist()

    X = df[feature_cols]
    y = df[target_col]

    
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def train_logistic_regression(X_train, y_train,random_state: int = 42) -> LogisticRegression:
    """
    Train a multinomial logistic regression model.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        random_state (int, optional): Random seed for reproducibility. Defaults to 42.

    Returns:
        LogisticRegression: Trained logistic regression model.
    """
    model = LogisticRegression(random_state=random_state,max_iter = 200)
    model.fit(X_train, y_train)


    return model


