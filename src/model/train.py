import numpy as np
import mlflow
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import os
import joblib

def train_model(X_train_path, y_train_path, X_val_path, y_val_path, model_dir):
    # Load data
    X_train = np.load(X_train_path)
    y_train = np.load(y_train_path)
    X_val = np.load(X_val_path)
    y_val = np.load(y_val_path)

    # Convert to DMatrix for better performance
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    # Define XGBoost parameters
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'learning_rate': 0.01,  # Reduced learning rate
        'max_depth': 4,  # Reduced depth
        'min_child_weight': 5,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'eta': 0.01,
        'seed': 42
    }

    # Start MLflow run
    mlflow.set_experiment("house-price-prediction")

    with mlflow.start_run():
        # Train model with early stopping
        evallist = [(dtrain, 'train'), (dval, 'val')]
        num_round = 10000

        model = xgb.train(
            params,
            dtrain,
            num_round,
            evallist,
            early_stopping_rounds=100,
            verbose_eval=100
        )

        # Make predictions (still in log space)
        y_pred = model.predict(dval)

        # Calculate metrics in log space first
        rmse_log = np.sqrt(mean_squared_error(y_val, y_pred))
        r2_log = r2_score(y_val, y_pred)

        # Safely transform predictions back to original scale
        try:
            # Clip predictions to avoid overflow
            y_val_clipped = np.clip(y_val, -100, 100)
            y_pred_clipped = np.clip(y_pred, -100, 100)

            # Transform back to original scale
            y_val_orig = np.expm1(y_val_clipped)
            y_pred_orig = np.expm1(y_pred_clipped)

            # Calculate metrics in original scale
            rmse = np.sqrt(mean_squared_error(y_val_orig, y_pred_orig))
            mae = mean_absolute_error(y_val_orig, y_pred_orig)
            r2 = r2_score(y_val_orig, y_pred_orig)

        except (RuntimeWarning, ValueError) as e:
            print(f"Warning: Error calculating metrics in original scale: {str(e)}")
            print("Falling back to log-space metrics")
            rmse = rmse_log
            mae = 0.0  # Cannot calculate MAE in this case
            r2 = r2_log

        # Log parameters and metrics
        for param, value in params.items():
            mlflow.log_param(param, value)

        mlflow.log_metric("rmse_log", rmse_log)
        mlflow.log_metric("r2_log", r2_log)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        # Save model
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(model, os.path.join(model_dir, "model.joblib"))

        print("\nModel Training Results:")
        print(f"Log-space metrics:")
        print(f"RMSE (log): {rmse_log:.4f}")
        print(f"R² (log): {r2_log:.4f}")
        print(f"\nOriginal-space metrics:")
        print(f"RMSE: ${rmse:,.2f}")
        print(f"MAE: ${mae:,.2f}")
        print(f"R²: {r2:.4f}")
        print(f"\nBest iteration: {model.best_iteration}")

        return model


if __name__ == "__main__":
    train_model(
        'data/processed/X_train.npy',
        'data/processed/y_train.npy',
        'data/processed/X_val.npy',
        'data/processed/y_val.npy',
        'models'
    )