import numpy as np
import mlflow
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import os
import joblib
import warnings


def train_model(X_train_path, y_train_path, X_val_path, y_val_path, model_dir):
    # Load data
    X_train = np.load(X_train_path)
    y_train = np.load(y_train_path)
    X_val = np.load(X_val_path)
    y_val = np.load(y_val_path)

    # First, check the range of your target variable
    print(f"Target variable statistics:")
    print(f"Min: {np.min(y_train)}, Max: {np.max(y_train)}")
    print(f"Mean: {np.mean(y_train)}, Median: {np.median(y_train)}")

    # Determine if data is already log-transformed
    # If min value is negative or close to zero, and data doesn't have extremely large values,
    # it's likely already log-transformed
    min_val = np.min(y_train)
    max_val = np.max(y_train)
    is_log_transformed = min_val < 0 or (min_val >= 0 and min_val < 1 and max_val < 30)

    print(f"Data appears to be {'log-transformed' if is_log_transformed else 'in original scale'}")

    # If not log-transformed, transform it
    if not is_log_transformed:
        # Handle zeros if present
        if np.min(y_train) == 0:
            print("Warning: Zero values found in target. Adding small constant before log transform.")
            y_train_log = np.log1p(y_train)
            y_val_log = np.log1p(y_val)
        else:
            y_train_log = np.log(y_train)
            y_val_log = np.log(y_val)
        print("Applied log transformation to target variable")
    else:
        y_train_log = y_train
        y_val_log = y_val
        print("Using target variable as-is (already log-transformed)")

    # Convert to DMatrix for better performance
    dtrain = xgb.DMatrix(X_train, label=y_train_log)
    dval = xgb.DMatrix(X_val, label=y_val_log)

    # Define XGBoost parameters
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'learning_rate': 0.01,
        'max_depth': 4,
        'min_child_weight': 5,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
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

        # Make predictions (in log space)
        y_pred_log = model.predict(dval)

        # Calculate metrics in log space
        rmse_log = np.sqrt(mean_squared_error(y_val_log, y_pred_log))
        r2_log = r2_score(y_val_log, y_pred_log)

        # Try to transform predictions back to original scale
        original_metrics_successful = False

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                if is_log_transformed:
                    # Data was already log-transformed, use appropriate inverse
                    # Check if values are natural log or log1p
                    if min_val < 0 or (min_val >= 0 and min_val < 1 and max_val < 20):
                        # Likely log1p transform
                        y_val_orig = np.expm1(y_val_log)
                        y_pred_orig = np.expm1(y_pred_log)
                        print("Using expm1 to convert back to original scale")
                    else:
                        # Likely natural log transform
                        y_val_orig = np.exp(y_val_log)
                        y_pred_orig = np.exp(y_pred_log)
                        print("Using exp to convert back to original scale")
                else:
                    # We applied log1p above
                    y_val_orig = np.expm1(y_val_log)
                    y_pred_orig = np.expm1(y_pred_log)

                # Check if transformation produced valid values
                if np.isfinite(y_pred_orig).all() and np.isfinite(y_val_orig).all():
                    # Calculate metrics in original scale
                    rmse = np.sqrt(mean_squared_error(y_val_orig, y_pred_orig))
                    mae = mean_absolute_error(y_val_orig, y_pred_orig)
                    r2 = r2_score(y_val_orig, y_pred_orig)

                    # Additional check for reasonable values
                    if r2 > -10 and r2 <= 1:  # R² should be less than or equal to 1
                        original_metrics_successful = True
                        metrics_source = "original scale"
                    else:
                        print(f"Calculated R² value ({r2}) is outside reasonable range.")
                else:
                    print("Transformation produced non-finite values.")
            except Exception as e:
                print(f"Error during transformation: {str(e)}")

        if not original_metrics_successful:
            print("Falling back to log-space metrics only")
            rmse = rmse_log
            mae = 0.0  # Cannot calculate MAE in this case
            r2 = r2_log
            metrics_source = "log scale (fallback)"

            # Save predictions in log space for later analysis
            np.save(os.path.join(model_dir, "y_val_log.npy"), y_val_log)
            np.save(os.path.join(model_dir, "y_pred_log.npy"), y_pred_log)

        # Log parameters and metrics
        for param, value in params.items():
            mlflow.log_param(param, value)

        mlflow.log_metric("rmse_log", rmse_log)
        mlflow.log_metric("r2_log", r2_log)

        if original_metrics_successful:
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)

        # Save model
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(model, os.path.join(model_dir, "model.joblib"))

        # Save whether the model was trained on log-transformed data
        with open(os.path.join(model_dir, "transformation_info.txt"), "w") as f:
            f.write(f"is_log_transformed={is_log_transformed}\n")
            if original_metrics_successful:
                f.write(
                    f"transform_method={'expm1' if min_val < 0 or (min_val >= 0 and min_val < 1 and max_val < 20) else 'exp'}\n")

        print("\nModel Training Results:")
        print(f"Log-space metrics:")
        print(f"RMSE (log): {rmse_log:.4f}")
        print(f"R² (log): {r2_log:.4f}")

        if original_metrics_successful:
            print(f"\nOriginal-space metrics ({metrics_source}):")
            print(f"RMSE: ${rmse:,.2f}")
            print(f"MAE: ${mae:,.2f}")
            print(f"R²: {r2:.4f}")
        else:
            print("\nOriginal-space metrics: Not available (using log-space metrics instead)")

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