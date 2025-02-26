import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
import os


def preprocess_data(input_train_path, input_test_path, output_dir):
    # Load data
    train_data = pd.read_csv(input_train_path)
    test_data = pd.read_csv(input_test_path)

    # Separate target, the feature 'SalePrice' which is the target variable needs to be dropped
    # from the training features to avoid data leakage
    X_train = train_data.drop('SalePrice', axis=1)
    y_train = train_data['SalePrice']

    # Identify numeric and categorical columns
    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X_train.select_dtypes(include=['object']).columns

    # Create preprocessing pipelines
    # Create pipeline to handle missing data in numeric features using SimpleImputer() with strategy as median
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    # Create pipeline to handle missing data in categorical features using the 'most_frequent' strategy
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    # Fit and transform the training data
    X_train_processed = preprocessor.fit_transform(X_train)

    X_test_processed = preprocessor.transform(test_data) if test_data is not None else None

    # Split into train and validation sets
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train_processed, y_train, test_size=0.2, random_state=42
    )

    X_train_final = np.asarray(X_train_final)
    X_val = np.asarray(X_val)
    y_train_final = np.asarray(y_train_final)
    y_val = np.asarray(y_val)
    # Save the preprocessed data
    # A .npy file is a binary file format used by the numpy library in Python to store numpy arrays.
    # This format is efficient and preserves the array structure, data type, and shape.
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, 'X_train.npy'), X_train_final)
    np.save(os.path.join(output_dir, 'X_val.npy'), X_val)
    np.save(os.path.join(output_dir, 'y_train.npy'), y_train_final)
    np.save(os.path.join(output_dir, 'y_val.npy'), y_val)

    if X_test_processed is not None:
        np.save(os.path.join(output_dir, 'X_test.npy'), X_test_processed)

    # Save the preprocessor
    joblib.dump(preprocessor, os.path.join(output_dir, 'preprocessor.joblib'))

    try:
        feature_names = (numeric_features +
                         [f"{feat}_{val}" for feat, vals in
                          zip(categorical_features,
                              preprocessor.named_transformers_['cat']['onehot'].categories_)
                          for val in vals[1:]])

        with open(os.path.join(output_dir, 'feature_names.txt'), 'w') as f:
            f.write('\n'.join(feature_names))
    except Exception as e:
        print(f"Warning: Could not save feature names: {str(e)}")

    return preprocessor


if __name__ == "__main__":
    preprocess_data(
        'data/raw/train.csv',
        'data/raw/test.csv',
        'data/processed'
    )