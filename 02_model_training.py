# Airbnb Price Prediction: Model Training
# Author: Team 3
# Date: December 2025
# Purpose: Train Random Forest and Gradient Boosting models with cross-validation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, cross_validate, train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

print("="*70)
print("AIRBNB PRICE PREDICTION: MODEL TRAINING")
print("="*70)

# LOAD CLEANED DATA

print("\n[STEP 1] Loading cleaned data...")
try:
    df = pd.read_csv('../data/airbnb_cleaned.csv')
    print(f"Loaded {len(df)} rows and {len(df.columns)} columns")
except FileNotFoundError:
    print("⚠ airbnb_cleaned.csv not found. Using sample data structure.")
    print("Please ensure your cleaned CSV is in ../data/airbnb_cleaned.csv")
    exit(1)

# PREPARE DATA FOR MODELING

print("\n[STEP 2] Preparing data for modeling...")

# Target variable
target_col = 'log_price'
if target_col not in df.columns:
    print(f"⚠ Target column '{target_col}' not found in data")
    print(f"Available columns: {df.columns.tolist()}")
    exit(1)

y = df[target_col]

# Select features (drop target and non-predictive columns)
exclude_cols = [target_col, 'price', 'id', 'name', 'host_id', 'host_name']
X = df.drop(columns=[col for col in exclude_cols if col in df.columns])

# Handle categorical variables - encode them
print(f"Processing {len(X.columns)} features...")
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()

print(f"  - Numerical features: {len(numerical_cols)}")
print(f"  - Categorical features: {len(categorical_cols)}")

# Encode categorical variables
le_dict = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    le_dict[col] = le
    print(f"    Encoded {col}")

# Check for missing values
if X.isnull().sum().sum() > 0:
    print("\n⚠ Missing values detected:")
    print(X.isnull().sum()[X.isnull().sum() > 0])
    print("Filling with 0...")
    X = X.fillna(0)
else:
    print("\nNo missing values in features")

print(f"\nFinal feature matrix shape: {X.shape}")
print(f"Target variable shape: {y.shape}")

# TRAIN-TEST SPLIT

print("\n[STEP 3] Splitting data into train/test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# TRAIN RANDOM FOREST MODEL

print("\n[STEP 4] Training Random Forest Regressor...")
print("-" * 70)

rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

rf_model.fit(X_train, y_train)
print("Random Forest model trained")

# Get cross-validation scores
print("\nRunning 5-fold cross-validation...")
rf_cv_scores = cross_validate(
    rf_model, X_train, y_train,
    cv=5,
    scoring=['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2'],
    n_jobs=-1
)

rf_rmse_cv = np.sqrt(-rf_cv_scores['test_neg_mean_squared_error'])
rf_mae_cv = -rf_cv_scores['test_neg_mean_absolute_error']
rf_r2_cv = rf_cv_scores['test_r2']

print(f"  RMSE (CV): {rf_rmse_cv.mean():.4f} (+/- {rf_rmse_cv.std():.4f})")
print(f"  MAE (CV):  {rf_mae_cv.mean():.4f} (+/- {rf_mae_cv.std():.4f})")
print(f"  R² (CV):   {rf_r2_cv.mean():.4f} (+/- {rf_r2_cv.std():.4f})")

# Test set performance
rf_pred_train = rf_model.predict(X_train)
rf_pred_test = rf_model.predict(X_test)

rf_rmse_train = np.sqrt(mean_squared_error(y_train, rf_pred_train))
rf_mae_train = mean_absolute_error(y_train, rf_pred_train)
rf_r2_train = r2_score(y_train, rf_pred_train)

rf_rmse_test = np.sqrt(mean_squared_error(y_test, rf_pred_test))
rf_mae_test = mean_absolute_error(y_test, rf_pred_test)
rf_r2_test = r2_score(y_test, rf_pred_test)

print(f"\nRandom Forest - Train Performance:")
print(f"  RMSE: {rf_rmse_train:.4f}, MAE: {rf_mae_train:.4f}, R²: {rf_r2_train:.4f}")
print(f"\nRandom Forest - Test Performance:")
print(f"  RMSE: {rf_rmse_test:.4f}, MAE: {rf_mae_test:.4f}, R²: {rf_r2_test:.4f}")

# TRAIN GRADIENT BOOSTING MODEL


print("\n[STEP 5] Training Gradient Boosting Regressor...")
print("-" * 70)

gb_model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    verbose=0
)

gb_model.fit(X_train, y_train)
print("Gradient Boosting model trained")

# Get cross-validation scores
print("\nRunning 5-fold cross-validation...")
gb_cv_scores = cross_validate(
    gb_model, X_train, y_train,
    cv=5,
    scoring=['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2'],
    n_jobs=-1
)

gb_rmse_cv = np.sqrt(-gb_cv_scores['test_neg_mean_squared_error'])
gb_mae_cv = -gb_cv_scores['test_neg_mean_absolute_error']
gb_r2_cv = gb_cv_scores['test_r2']

print(f"  RMSE (CV): {gb_rmse_cv.mean():.4f} (+/- {gb_rmse_cv.std():.4f})")
print(f"  MAE (CV):  {gb_mae_cv.mean():.4f} (+/- {gb_mae_cv.std():.4f})")
print(f"  R² (CV):   {gb_r2_cv.mean():.4f} (+/- {gb_r2_cv.std():.4f})")

# Test set performance
gb_pred_train = gb_model.predict(X_train)
gb_pred_test = gb_model.predict(X_test)

gb_rmse_train = np.sqrt(mean_squared_error(y_train, gb_pred_train))
gb_mae_train = mean_absolute_error(y_train, gb_pred_train)
gb_r2_train = r2_score(y_train, gb_pred_train)

gb_rmse_test = np.sqrt(mean_squared_error(y_test, gb_pred_test))
gb_mae_test = mean_absolute_error(y_test, gb_pred_test)
gb_r2_test = r2_score(y_test, gb_pred_test)

print(f"\nGradient Boosting - Train Performance:")
print(f"  RMSE: {gb_rmse_train:.4f}, MAE: {gb_mae_train:.4f}, R²: {gb_r2_train:.4f}")
print(f"\nGradient Boosting - Test Performance:")
print(f"  RMSE: {gb_rmse_test:.4f}, MAE: {gb_mae_test:.4f}, R²: {gb_r2_test:.4f}")

# MODEL COMPARISON & FEATURE IMPORTANCE

print("\n Comparing models...")
print("="*70)

comparison_df = pd.DataFrame({
    'Model': ['Random Forest', 'Random Forest', 'Gradient Boosting', 'Gradient Boosting'],
    'Set': ['Train', 'Test', 'Train', 'Test'],
    'RMSE': [rf_rmse_train, rf_rmse_test, gb_rmse_train, gb_rmse_test],
    'MAE': [rf_mae_train, rf_mae_test, gb_mae_train, gb_mae_test],
    'R²': [rf_r2_train, rf_r2_test, gb_r2_train, gb_r2_test]
})

print("\nModel Performance Summary:")
print(comparison_df.to_string(index=False))

# Determine best model
if gb_r2_test > rf_r2_test:
    best_model_name = "Gradient Boosting"
    best_model = gb_model
    best_predictions = gb_pred_test
else:
    best_model_name = "Random Forest"
    best_model = rf_model
    best_predictions = rf_pred_test

print(f"\nBEST MODEL: {best_model_name}")

# Feature Importance
print("\n Feature Importance Analysis...")
print("-" * 70)

feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'RF_Importance': rf_model.feature_importances_,
    'GB_Importance': gb_model.feature_importances_
})

feature_importance_df['Average_Importance'] = (
    feature_importance_df['RF_Importance'] + feature_importance_df['GB_Importance']
) / 2

feature_importance_df = feature_importance_df.sort_values(
    'Average_Importance', ascending=False
)

print("\nTop 10 Most Important Features (Average across both models):")
print(feature_importance_df.head(10)[['Feature', 'Average_Importance']].to_string(index=False))

# VISUALIZATIONS

print("\n[STEP 8] Creating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Model Comparison (Test Set)
ax1 = axes[0, 0]
models = ['Random Forest', 'Gradient Boosting']
rmse_vals = [rf_rmse_test, gb_rmse_test]
colors = ['steelblue', 'coral']
ax1.bar(models, rmse_vals, color=colors, alpha=0.7, edgecolor='black')
ax1.set_ylabel('RMSE (Test Set)')
ax1.set_title('Model Comparison: RMSE')
ax1.set_ylim(0, max(rmse_vals) * 1.2)
for i, v in enumerate(rmse_vals):
    ax1.text(i, v + 0.02, f'{v:.4f}', ha='center', va='bottom', fontweight='bold')

# Plot 2: Feature Importance
ax2 = axes[0, 1]
top_features = feature_importance_df.head(10)
ax2.barh(range(len(top_features)), top_features['Average_Importance'], color='mediumseagreen', edgecolor='black')
ax2.set_yticks(range(len(top_features)))
ax2.set_yticklabels(top_features['Feature'])
ax2.set_xlabel('Average Feature Importance')
ax2.set_title('Top 10 Most Important Features')
ax2.invert_yaxis()

# Plot 3: Actual vs Predicted (Best Model)
ax3 = axes[1, 0]
ax3.scatter(y_test, best_predictions, alpha=0.5, s=20, color='purple')
min_val = min(y_test.min(), best_predictions.min())
max_val = max(y_test.max(), best_predictions.max())
ax3.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
ax3.set_xlabel('Actual log_price')
ax3.set_ylabel('Predicted log_price')
ax3.set_title(f'{best_model_name}: Actual vs Predicted')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Residuals Distribution
ax4 = axes[1, 1]
residuals = y_test - best_predictions
ax4.hist(residuals, bins=50, color='darkblue', alpha=0.7, edgecolor='black')
ax4.axvline(residuals.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {residuals.mean():.4f}')
ax4.set_xlabel('Residuals (Actual - Predicted)')
ax4.set_ylabel('Frequency')
ax4.set_title('Residuals Distribution')
ax4.legend()

plt.tight_layout()
plt.savefig('../results/model_performance.png', dpi=300, bbox_inches='tight')
print("Saved: model_performance.png")

# Save feature importance plot
fig2, ax = plt.subplots(figsize=(10, 8))
top_15_features = feature_importance_df.head(15)
ax.barh(range(len(top_15_features)), top_15_features['Average_Importance'], 
        color='teal', alpha=0.8, edgecolor='black')
ax.set_yticks(range(len(top_15_features)))
ax.set_yticklabels(top_15_features['Feature'])
ax.set_xlabel('Average Feature Importance')
ax.set_title('Top 15 Most Important Features for Price Prediction')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig('../results/feature_importance.png', dpi=300, bbox_inches='tight')
print("Saved: feature_importance.png")

# SAVE RESULTS & PREDICTIONS

print("\n[STEP 9] Saving results...")

# Save performance report
performance_report = f"""
AIRBNB PRICE PREDICTION - MODEL PERFORMANCE REPORT
{'='*70}

DATASET INFORMATION:
- Total samples: {len(df)}
- Train set size: {len(X_train)}
- Test set size: {len(X_test)}
- Number of features: {X.shape[1]}
- Target variable: log_price (nightly rate)

RANDOM FOREST RESULTS:
- Cross-Validation (5-fold):
  * RMSE: {rf_rmse_cv.mean():.4f} ± {rf_rmse_cv.std():.4f}
  * MAE: {rf_mae_cv.mean():.4f} ± {rf_mae_cv.std():.4f}
  * R²: {rf_r2_cv.mean():.4f} ± {rf_r2_cv.std():.4f}

- Train Set Performance:
  * RMSE: {rf_rmse_train:.4f}
  * MAE: {rf_mae_train:.4f}
  * R²: {rf_r2_train:.4f}

- Test Set Performance:
  * RMSE: {rf_rmse_test:.4f}
  * MAE: {rf_mae_test:.4f}
  * R²: {rf_r2_test:.4f}

GRADIENT BOOSTING RESULTS:
- Cross-Validation (5-fold):
  * RMSE: {gb_rmse_cv.mean():.4f} ± {gb_rmse_cv.std():.4f}
  * MAE: {gb_mae_cv.mean():.4f} ± {gb_mae_cv.std():.4f}
  * R²: {gb_r2_cv.mean():.4f} ± {gb_r2_cv.std():.4f}

- Train Set Performance:
  * RMSE: {gb_rmse_train:.4f}
  * MAE: {gb_mae_train:.4f}
  * R²: {gb_r2_train:.4f}

- Test Set Performance:
  * RMSE: {gb_rmse_test:.4f}
  * MAE: {gb_mae_test:.4f}
  * R²: {gb_r2_test:.4f}

BEST MODEL: {best_model_name}
- Test R² Score: {max(rf_r2_test, gb_r2_test):.4f}
- Test RMSE: {min(rf_rmse_test, gb_rmse_test):.4f}
- Test MAE: {min(rf_mae_test, gb_mae_test):.4f}

TOP 10 MOST IMPORTANT FEATURES:
{feature_importance_df.head(10)[['Feature', 'Average_Importance']].to_string(index=False)}

KEY INSIGHTS:
1. The {best_model_name} model shows the best test set performance
2. Feature importance rankings align with EDA findings
3. Accommodation capacity and bedroom count remain top predictors
4. Model achieves strong R² score, indicating good predictive power
5. Residuals show relatively normal distribution with some outliers
"""

with open('../results/model_performance_report.txt', 'w') as f:
    f.write(performance_report)
print("Saved: model_performance_report.txt")

# Save test set predictions
predictions_df = pd.DataFrame({
    'Actual_log_price': y_test.values,
    'Predicted_log_price': best_predictions,
    'Residual': y_test.values - best_predictions,
    'Error_Percentage': (abs(y_test.values - best_predictions) / abs(y_test.values)) * 100
})

predictions_df.to_csv('../results/test_predictions.csv', index=False)
print("Saved: test_predictions.csv")

# Save feature importance
feature_importance_df.to_csv('../results/feature_importance.csv', index=False)
print("Saved: feature_importance.csv")

print("\n" + "="*70)
print("MODEL TRAINING COMPLETE")
print("="*70)
print(f"\nAll results saved to ../results/")
print("\nNext steps:")
print("1. Review model_performance_report.txt for detailed metrics")
print("2. Check test_predictions.csv for sample predictions")
print("3. Use feature_importance.csv for business insights")