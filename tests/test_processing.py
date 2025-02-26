import pytest
import os
import pandas as pd
from src.preprocessing.preprocess import preprocess_data


def test_preprocess_data(tmpdir):
    # Create dummy data
    train_data = pd.DataFrame({
        'SalePrice': [100000, 200000],
        'LotArea': [8000, 9000],
        'OverallQual': [5, 6]
    })
    test_data = pd.DataFrame({
        'LotArea': [8500],
        'OverallQual': [5]
    })

    # Save dummy data
    os.makedirs(os.path.join(tmpdir, 'raw'), exist_ok=True)
    train_data.to_csv(os.path.join(tmpdir, 'raw', 'train.csv'), index=False)
    test_data.to_csv(os.path.join(tmpdir, 'raw', 'test.csv'), index=False)

    # Run preprocessing
    preprocessor = preprocess_data(
        os.path.join(tmpdir, 'raw', 'train.csv'),
        os.path.join(tmpdir, 'raw', 'test.csv'),
        os.path.join(tmpdir, 'processed')
    )

    # Check if files were created
    assert os.path.exists(os.path.join(tmpdir, 'processed', 'X_train.npy'))
    assert os.path.exists(os.path.join(tmpdir, 'processed', 'y_train.npy'))
    assert os.path.exists(os.path.join(tmpdir, 'processed', 'X_val.npy'))
    assert os.path.exists(os.path.join(tmpdir, 'processed', 'y_val.npy'))
    assert os.path.exists(os.path.join(tmpdir, 'processed', 'preprocessor.joblib'))