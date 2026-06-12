# ─────────────────────────────────────────────────────────────
# Phase 3 — Model Training
# Project : Dubai Apartment Rent Predictor
# Input   : dubai_clean.csv
# Output  : model.pkl, encoders.pkl, shap_explainer.pkl, feature_cols.pkl
# Run     : python model.py
# ─────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import pickle
import warnings
import os

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import shap

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 120
pd.set_option("display.float_format", "{:,.0f}".format)

# ─────────────────────────────────────────────────────────────
# PATHS — all files sit in the same folder as this script
# ─────────────────────────────────────────────────────────────
BASE              = os.path.dirname(os.path.abspath(__file__))
CLEAN_PATH        = os.path.join(BASE, "dubai_clean.csv")
MODEL_PATH        = os.path.join(BASE, "model.pkl")
ENCODER_PATH      = os.path.join(BASE, "encoders.pkl")
SHAP_PATH         = os.path.join(BASE, "shap_explainer.pkl")
FEATURE_META_PATH = os.path.join(BASE, "feature_cols.pkl")

# ─────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────
print("=" * 55)
print("PHASE 3 — MODEL TRAINING")
print("=" * 55)

df = pd.read_csv(CLEAN_PATH)
print(f"\nLoaded: {df.shape[0]:,} rows, {df.shape[1]} columns")

# ─────────────────────────────────────────────────────────────
# 2. FEATURE PREPARATION
# ─────────────────────────────────────────────────────────────
NUMERIC_FEATURES = ["Area_in_sqft", "Beds", "Baths", "Beds_x_Baths"]
CAT_FEATURES     = ["Location", "Furnishing", "Type", "Frequency"]
TARGET           = "log_Rent"

# Create derived columns if missing
if "log_Rent" not in df.columns:
    df["log_Rent"] = np.log(df["Rent"])
    print("log_Rent column created.")

if "Beds_x_Baths" not in df.columns:
    df["Beds_x_Baths"] = df["Beds"] * df["Baths"]
    print("Beds_x_Baths column created.")

# Drop rows with nulls in any model column
before = len(df)
df = df.dropna(subset=NUMERIC_FEATURES + CAT_FEATURES + [TARGET])
print(f"Dropped {before - len(df)} rows with nulls. Remaining: {len(df):,}")

# ─────────────────────────────────────────────────────────────
# 3. LABEL ENCODE CATEGORICALS
# ─────────────────────────────────────────────────────────────
print("\nEncoding categorical features...")
encoders = {}

for col in CAT_FEATURES:
    le = LabelEncoder()
    df[col + "_enc"] = le.fit_transform(df[col].astype(str))
    encoders[col] = le
    print(f"  {col}: {len(le.classes_)} unique values")

with open(ENCODER_PATH, "wb") as f:
    pickle.dump(encoders, f)
print(f"Encoders saved → encoders.pkl")

# ─────────────────────────────────────────────────────────────
# 4. BUILD FEATURE MATRIX
# ─────────────────────────────────────────────────────────────
ENCODED_CATS = [col + "_enc" for col in CAT_FEATURES]
FEATURE_COLS = NUMERIC_FEATURES + ENCODED_CATS

X = df[FEATURE_COLS]
y = df[TARGET]

print(f"\nFeature matrix: {X.shape}")
print(f"Features: {FEATURE_COLS}")

# ─────────────────────────────────────────────────────────────
# 5. TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTrain: {X_train.shape[0]:,} rows")
print(f"Test:  {X_test.shape[0]:,} rows")

# ─────────────────────────────────────────────────────────────
# 6. TRAIN XGBOOST
# ─────────────────────────────────────────────────────────────
print("\nTraining XGBoost model...")

model = xgb.XGBRegressor(
    n_estimators     = 500,
    max_depth        = 6,
    learning_rate    = 0.05,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    random_state     = 42,
    n_jobs           = -1,
    verbosity        = 0,
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=100,
)
print("Training complete.")

# ─────────────────────────────────────────────────────────────
# 7. EVALUATE
# ─────────────────────────────────────────────────────────────
y_pred_log = model.predict(X_test)
y_pred_aed = np.exp(y_pred_log)
y_test_aed = np.exp(y_test)

r2       = r2_score(y_test, y_pred_log)
rmse_aed = np.sqrt(mean_squared_error(y_test_aed, y_pred_aed))
mape     = np.mean(np.abs((y_test_aed - y_pred_aed) / y_test_aed)) * 100

print("\n=== MODEL PERFORMANCE ===")
print(f"R²:   {r2:.4f}")
print(f"RMSE: AED {rmse_aed:,.0f}")
print(f"MAPE: {mape:.1f}%")

if r2 >= 0.85:
    print("Excellent — ready for the app.")
elif r2 >= 0.75:
    print("Good — acceptable for a portfolio project.")
else:
    print("Below target — check feature encoding or outliers.")

# Actual vs Predicted plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].scatter(y_test, y_pred_log, alpha=0.2, s=8, color="steelblue")
lims = [y_test.min(), y_test.max()]
axes[0].plot(lims, lims, "r--", lw=1.5, label="Perfect prediction")
axes[0].set_xlabel("Actual log(Rent)")
axes[0].set_ylabel("Predicted log(Rent)")
axes[0].set_title(f"Actual vs Predicted (log)   R²={r2:.3f}")
axes[0].legend()

axes[1].scatter(y_test_aed, y_pred_aed, alpha=0.2, s=8, color="teal")
lims2 = [y_test_aed.min(), y_test_aed.max()]
axes[1].plot(lims2, lims2, "r--", lw=1.5, label="Perfect prediction")
axes[1].set_xlabel("Actual Rent (AED)")
axes[1].set_ylabel("Predicted Rent (AED)")
axes[1].set_title(f"Actual vs Predicted (AED)   MAPE={mape:.1f}%")
axes[1].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(BASE, "model_actual_vs_predicted.png"), bbox_inches="tight")
print("Plot saved → model_actual_vs_predicted.png")
plt.show()

# ─────────────────────────────────────────────────────────────
# 8. SHAP EXPLAINABILITY
# ─────────────────────────────────────────────────────────────
print("\nComputing SHAP values — takes ~30 seconds...")
explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
print("Done.")

# Global importance bar chart
plt.figure(figsize=(10, 5))
shap.summary_plot(shap_values, X_test, plot_type="bar",
                  feature_names=FEATURE_COLS, show=False)
plt.title("SHAP — Global Feature Importance")
plt.tight_layout()
plt.savefig(os.path.join(BASE, "shap_importance.png"), bbox_inches="tight")
print("Plot saved → shap_importance.png")
plt.show()

# Beeswarm — direction of impact
plt.figure(figsize=(10, 5))
shap.summary_plot(shap_values, X_test,
                  feature_names=FEATURE_COLS, show=False)
plt.title("SHAP — Feature Impact Direction")
plt.tight_layout()
plt.savefig(os.path.join(BASE, "shap_beeswarm.png"), bbox_inches="tight")
print("Plot saved → shap_beeswarm.png")
plt.show()

# ─────────────────────────────────────────────────────────────
# 9. SAVE ALL FILES
# ─────────────────────────────────────────────────────────────
print("\nSaving model files...")

with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

with open(SHAP_PATH, "wb") as f:
    pickle.dump(explainer, f)

with open(FEATURE_META_PATH, "wb") as f:
    pickle.dump({
        "feature_cols"     : FEATURE_COLS,
        "numeric_features" : NUMERIC_FEATURES,
        "cat_features"     : CAT_FEATURES,
    }, f)

print("\n=== ALL FILES SAVED ===")
print("  model.pkl")
print("  encoders.pkl")
print("  shap_explainer.pkl")
print("  feature_cols.pkl")
print("  model_actual_vs_predicted.png")
print("  shap_importance.png")
print("  shap_beeswarm.png")
print(f"\nFinal R²:   {r2:.4f}")
print(f"Final MAPE: {mape:.1f}%")
print("\nNext step → build app.py")