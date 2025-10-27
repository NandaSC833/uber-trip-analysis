# src/train.py
import joblib
import numpy as np
import pandas as pd
import sys
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from xgboost import XGBRegressor

# ---------------- Metrics ----------------
def mape(y_true, y_pred):
    """Mean Absolute Percentage Error"""
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100

def evaluate_model(model, X_test, y_test):
    """Evaluate model using MAE, RMSE, and MAPE"""
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape_score = mape(y_test, y_pred)
    return mae, rmse, mape_score

# ---------------- Main Script ----------------
def main():
    # Default paths (so you can just run: python src/train.py)
    train_path = Path("data/processed/features/train.joblib")
    test_path = Path("data/processed/features/test.joblib")
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    # Load feature sets
    print("📂 Loading training and test data...")
    X_train, y_train = joblib.load(train_path)
    X_test, y_test = joblib.load(test_path)
    print(f"✅ Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # Define cross-validation
    tscv = TimeSeriesSplit(n_splits=3)

    # ---------------- XGBoost ----------------
    print("\n🚀 Training XGBoost...")
    xgb = XGBRegressor(objective="reg:squarederror", random_state=42)
    xgb_grid = {
        "n_estimators": [100, 300],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.8, 1.0],
    }
    grid_xgb = GridSearchCV(
        xgb, xgb_grid, scoring="neg_mean_absolute_error",
        cv=tscv, n_jobs=-1, verbose=1
    )
    grid_xgb.fit(X_train, y_train)
    best_xgb = grid_xgb.best_estimator_
    joblib.dump(best_xgb, models_dir / "xgb_best.joblib")
    print("✅ Best XGBoost params:", grid_xgb.best_params_)

    # ---------------- Random Forest ----------------
    print("\n🌲 Training Random Forest...")
    rf = RandomForestRegressor(random_state=42)
    rf_grid = {
        "n_estimators": [200, 500],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5],
    }
    grid_rf = GridSearchCV(
        rf, rf_grid, scoring="neg_mean_absolute_error",
        cv=tscv, n_jobs=-1, verbose=1
    )
    grid_rf.fit(X_train, y_train)
    best_rf = grid_rf.best_estimator_
    joblib.dump(best_rf, models_dir / "rf_best.joblib")
    print("✅ Best Random Forest params:", grid_rf.best_params_)

    # ---------------- Gradient Boosting ----------------
    print("\n🌼 Training Gradient Boosting...")
    gbr = GradientBoostingRegressor(random_state=42)
    gbr_grid = {
        "n_estimators": [100, 300],
        "learning_rate": [0.05, 0.1],
        "max_depth": [3, 5],
    }
    grid_gbr = GridSearchCV(
        gbr, gbr_grid, scoring="neg_mean_absolute_error",
        cv=tscv, n_jobs=-1, verbose=1
    )
    grid_gbr.fit(X_train, y_train)
    best_gbr = grid_gbr.best_estimator_
    joblib.dump(best_gbr, models_dir / "gbr_best.joblib")
    print("✅ Best Gradient Boosting params:", grid_gbr.best_params_)

    # ---------------- Evaluate Models ----------------
    print("\n📊 Evaluating models on test set...")
    results = {}
    models = {"XGBoost": best_xgb, "RandomForest": best_rf, "GBR": best_gbr}

    for name, model in models.items():
        mae, rmse, mape_score = evaluate_model(model, X_test, y_test)
        results[name] = {"MAE": mae, "RMSE": rmse, "MAPE": mape_score}
        print(f"{name:15s} | MAE: {mae:.2f} | RMSE: {rmse:.2f} | MAPE: {mape_score:.2f}%")

    # Choose best performer (lowest MAPE)
    best_model_name = min(results, key=lambda k: results[k]["MAPE"])
    best_model = models[best_model_name]
    joblib.dump(best_model, models_dir / "best_model.joblib")

    print(f"\n🏆 Best Performer: {best_model_name}")
    print(f"✅ Model saved to: {models_dir / 'best_model.joblib'}")

if __name__ == "__main__":
    main()
