import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join('..', 'src')))
from unittest.mock import patch
import pandas as pd
import pytest
from src.preprocessing import load_raw_data, preprocess_data, save_processed_data
from tests.test_utils import get_sample_df


# Sample dataframe for testing
sample_df = get_sample_df()


@pytest.fixture
def tmp_csv_file(tmp_path):
    """
    Creates a temporary CSV file with sample data for testing load_raw_data.
    
    Returns:
        Path to the temporary CSV file.
    """
    file_path = tmp_path / "test_data.csv"
    sample_df.to_csv(file_path, index=False)
    return file_path


def test_load_raw_data(tmp_csv_file):
    """
    Test that load_raw_data correctly loads a CSV file into a DataFrame
    with the expected shape and column names.
    """
    df = load_raw_data(tmp_csv_file)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == sample_df.shape
    assert list(df.columns) == list(sample_df.columns)


@patch('src.preprocessing.joblib.dump')
def test_preprocess_data(mock_dump):
    """
    Test that preprocess_data:
    - Drops the 'Id' column
    - Encodes the target column correctly
    - Standardizes feature columns (mean ≈ 0, std ≈ 1)
    - Returns a DataFrame with the correct shape
    """
    processed_df = preprocess_data(sample_df.copy(), target_col='target')
    assert isinstance(processed_df, pd.DataFrame)
    mock_dump.assert_called_once()

    # Check 'Id' is dropped
    assert 'Id' not in processed_df.columns

    # Check target column is integer-encoded
    assert pd.api.types.is_integer_dtype(processed_df['target'])

    # Check feature scaling
    features = processed_df.drop(columns='target')
    means = features.mean().round()
    stds = features.std().round()
    assert all(means.abs() <= 1)
    assert all((stds - 1).abs() <= 1)


def test_save_processed_data(tmp_path):
    """
    Test that save_processed_data correctly saves a DataFrame to CSV,
    and that the saved file matches the original input.
    """
    df_to_save = sample_df.drop(columns='Id')  # Since preprocess_data drops 'Id'
    test_path = tmp_path / "output.csv"
    save_processed_data(df_to_save, test_path)

    # Load and verify
    saved_df = pd.read_csv(test_path)
    assert saved_df.shape == df_to_save.shape
    assert list(saved_df.columns) == list(df_to_save.columns)
    pd.testing.assert_frame_equal(saved_df, df_to_save)
