import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_raw_data(path: str) -> pd.DataFrame:
    """
    Load raw dataset from a CSV file.

    Args:
        path (str): Path to the raw CSV file.

    Returns:
        pd.DataFrame: Loaded dataframe.
    """
    return pd.read_csv(path)


def preprocess_data(df: pd.DataFrame, target_col: str = 'target') -> pd.DataFrame:
    """
    Preprocess the dataset: encode target and scale features.

    Args:
        df (pd.DataFrame): Raw dataframe.
        target_col (str): Target column name.

    Returns:
        pd.DataFrame: Preprocessed dataframe.
    """
    df = df.copy()

    # Encode the target variable
    encoder = LabelEncoder()
    df[target_col] = encoder.fit_transform(df[target_col])
    df = df.drop(columns= 'Id',axis=1)

    # Scale features (excluding the target)
    features = df.columns.drop(target_col)
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])
    joblib.dump(scaler, '../models/scaler.pkl')

    return df


def save_processed_data(df: pd.DataFrame, path: str) -> None:
    """
    Save the processed dataframe to a CSV file.

    Args:
        df (pd.DataFrame): Processed dataframe.
        path (str): Path where to save the CSV.

    Returns:
        None
    """
    df.to_csv(path, index=False)
