import pandas as pd

def get_sample_df():
    """
    Returns a small sample DataFrame for testing that mimics the Iris dataset
    with all three target classes included.
    """
    return pd.DataFrame({
        'Id': [1,2,3,4],
        'feature1': [5.1, 7.0, 6.3,5.5],
        'feature2': [3.5, 3.2, 3.3,2.9],
        'feature3': [1.4, 4.7, 6.0,1.2],
        'feature4': [0.2, 1.4, 2.5,0.4],
        'target': ['setosa', 'versicolor', 'virginica','setosa']
    })