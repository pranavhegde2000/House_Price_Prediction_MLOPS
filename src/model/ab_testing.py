import numpy as np
import mlflow
import joblib
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import os
import pandas as pd


def train_ab_models(X_train_path, y_train_path, X_val_path, y_val_path, model_dir):
    # Load data
    X_train = np.load(X_train_path)
    y_train = np.load(y_train_path)
    X_val = np.load(X_val_path)
    y_val = np.load(y_val_path)

    # Define model configurations
    models = {
        'XGBoost': {
            'model': xgb.XGBRegressor(
                objective='reg:squarederror',
                eval_metric='rmse',
                learning_rate=0.1,
                max_depth=6,
                n_estimators=1000,
                early_stopping_rounds=50,
                colsample_bytree=0.8,
                subsample=0.8,
                min_child_weight=1,
                seed=42
            ),
            'params': {
                'eval_set': [(X_val, y_val)],
                'verbose': 100
            }
        },
        'RandomForest': {
            'model': RandomForestRegressor(
                n_estimators=1000,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=42
            ),
            'params': {}
        }
    }

    results = {}
    metrics_df = pd.DataFrame()

    for name, model_config in models.items():
        # Start MLflow run
        mlflow.set_experiment("house-price-ab-testing")

        with mlflow.start_run(run_name=name):
            model = model_config['model']

            # Train model
            model.fit(X_train, y_train, **model_config['params'])

            # Make predictions
            y_pred = model.predict(X_val)

            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            mae = mean_absolute_error(y_val, y_pred)
            r2 = r2_score(y_val, y_pred)

            # Log parameters
            if hasattr(model, 'get_params'):
                params = model.get_params()
                for param, value in params.items():
                    mlflow.log_param(param, value)

            # Log metrics
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)

            # Log feature importance
            if hasattr(model, 'feature_importances_'):
                feature_importance = model.feature_importances_
                mlflow.log_dict(
                    {"feature_importance": feature_importance.tolist()},
                    f"{name}_feature_importance.json"
                )

            # Log model
            if isinstance(model, xgb.XGBRegressor):
                mlflow.xgboost.log_model(model, name)
            else:
                mlflow.sklearn.log_model(model, name)

            # Save model locally
            os.makedirs(model_dir, exist_ok=True)
            if isinstance(model, xgb.XGBRegressor):
                model.save_model(os.path.join(model_dir, f"{name}.json"))
            else:
                joblib.dump(model, os.path.join(model_dir, f"{name}.joblib"))

            # Store results
            results[name] = {
                "rmse": rmse,
                "mae": mae,
                "r2": r2
            }

            # Add to metrics DataFrame
            metrics_df = pd.concat([
                metrics_df,
                pd.DataFrame({
                    'Model': [name],
                    'RMSE': [rmse],
                    'MAE': [mae],
                    'R2': [r2]
                })
            ])

    # Print comparison
    print("\nModel Comparison:")
    print(metrics_df.to_string(index=False))

    # Determine best model
    best_model = metrics_df.loc[metrics_df['RMSE'].idxmin(), 'Model']
    print(f"\nBest performing model: {best_model}")

    return results, metrics_df


if __name__ == "__main__":
    results, metrics_df = train_ab_models(
        'data/processed/X_train.npy',
        'data/processed/y_train.npy',
        'data/processed/X_val.npy',
        'data/processed/y_val.npy',
        'models'
    )