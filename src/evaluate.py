# src/evaluate.py
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ---------- helper metrics ----------
def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100

# ---------- main ----------
def main():
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    # Load data and model
    X_test, y_test = joblib.load("data/processed/features/test.joblib")
    model = joblib.load("models/best_model.joblib")
    model_name = type(model).__name__
    print(f"✅ Loaded best model: {model_name}")

    # Make predictions
    y_pred = model.predict(X_test)

    # Compute metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape_score = mape(y_test, y_pred)

    metrics = {"Model": model_name, "MAE": mae, "RMSE": rmse, "MAPE": mape_score}
    print("\n📊 Final Test Metrics")
    print(json.dumps(metrics, indent=4))

    # Save metrics to JSON
    with open(reports_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    # ------------- Plot 1: Actual vs Predicted -------------
    plt.figure(figsize=(10, 5))
    plt.plot(y_test.values, label="Actual", marker="o")
    plt.plot(y_pred, label="Predicted", marker="x")
    plt.title("Actual vs Predicted Uber Trips")
    plt.xlabel("Time Steps (Test Period)")
    plt.ylabel("Trip Count")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(reports_dir / "actual_vs_predicted.png")
    plt.close()

    # ------------- Plot 2: Residuals -------------
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 4))
    plt.plot(residuals, marker="o", color="red")
    plt.axhline(0, color="black", linestyle="--")
    plt.title("Residuals (Actual - Predicted)")
    plt.xlabel("Time Steps (Test Period)")
    plt.ylabel("Error")
    plt.tight_layout()
    plt.savefig(reports_dir / "residuals.png")
    plt.close()

    # ------------- Plot 3: Feature Importance -------------
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        feat_imp = pd.Series(importances, index=X_test.columns).sort_values(ascending=False)

        plt.figure(figsize=(8, 5))
        feat_imp.head(10).plot(kind="bar")
        plt.title("Top 10 Feature Importances")
        plt.ylabel("Importance")
        plt.tight_layout()
        plt.savefig(reports_dir / "feature_importance.png")
        plt.close()

    print("✅ Evaluation complete — results saved in 'reports/'")

if __name__ == "__main__":
    main()
