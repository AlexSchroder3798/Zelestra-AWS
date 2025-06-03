import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import numpy as np

from sklearn.model_selection import RandomizedSearchCV

from sklearn.linear_model import RidgeCV
# Load the data
train = pd.read_csv("trainhs.csv")
test = pd.read_csv("test.csv")



# Or check skewness numerically
skew_val = train['efficiency'].skew()
print(f"Skewness: {skew_val:.2f}")


# Convert columns that should be numeric
numeric_cols = ['humidity', 'wind_speed', 'pressure']
for col in numeric_cols:
    train[col] = pd.to_numeric(train[col], errors='coerce')
    test[col] = pd.to_numeric(test[col], errors='coerce')

# Identify numerical and categorical columns
numerical_features = train.select_dtypes(include=['float64', 'int64']).columns.drop(['id', 'efficiency'])
categorical_features = train.select_dtypes(include='object').columns

# Fill missing numerical values with median
num_imputer = SimpleImputer(strategy='median')
train[numerical_features] = num_imputer.fit_transform(train[numerical_features])
test[numerical_features] = num_imputer.transform(test[numerical_features])

# Fill missing categorical values with 'Unknown'
train[categorical_features] = train[categorical_features].fillna('Unknown')
test[categorical_features] = test[categorical_features].fillna('Unknown')

# Save original installation_type for segmented modeling
train["install_type_raw"] = train["installation_type"]
test["install_type_raw"] = test["installation_type"]

# Encode categorical features
label_encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])  # Make sure test only transforms existing classes
    label_encoders[col] = le  # Save in case you need to decode later

# Get original target
y_raw = train["efficiency"]



# Final training and testing matrices
X_train = train.drop(columns=["id", "efficiency"])
# Reflect, then log1p
y_train = np.log1p(y_raw.max() - y_raw)
X_test = test.drop(columns=["id"])

# --- Feature Engineering ---
X_train["power"] = X_train["voltage"] * X_train["current"]
X_test["power"] = X_test["voltage"] * X_test["current"]

X_train["temp_diff"] = X_train["module_temperature"] - X_train["temperature"]
X_test["temp_diff"] = X_test["module_temperature"] - X_test["temperature"]

X_train["irradiance_per_cloud"] = X_train["irradiance"] / (X_train["cloud_coverage"] + 1)
X_test["irradiance_per_cloud"] = X_test["irradiance"] / (X_test["cloud_coverage"] + 1)

X_train["soiling_adjusted_irradiance"] = X_train["irradiance"] * X_train["soiling_ratio"]
X_test["soiling_adjusted_irradiance"] = X_test["irradiance"] * X_test["soiling_ratio"]

X_train["maintenance_per_year"] = X_train["maintenance_count"] / (X_train["panel_age"] + 1)
X_test["maintenance_per_year"] = X_test["maintenance_count"] / (X_test["panel_age"] + 1)

#1. ---------------------------------------
X_train["power_temp_diff"] = X_train["power"] * X_train["temp_diff"]
X_test["power_temp_diff"] = X_test["power"] * X_test["temp_diff"]

X_train["irradiance_power"] = X_train["irradiance"] * X_train["power"]
X_test["irradiance_power"] = X_test["irradiance"] * X_test["power"]

X_train["voltage_squared"] = X_train["voltage"] ** 2
X_test["voltage_squared"] = X_test["voltage"] ** 2
#-----------------------------------------

# Re-assign X_train and X_test to include engineered features and exclude object columns
X_train = X_train.drop(columns=["install_type_raw"])
X_test = X_test.drop(columns=["install_type_raw"])



param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt']
}

import lightgbm as lgb
from sklearn.model_selection import cross_val_score

# Initialize LightGBM regressor
lgb_model = lgb.LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=-1,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

# Cross-validation
lgb_scores = cross_val_score(lgb_model, X_train, y_train, cv=5, scoring='neg_root_mean_squared_error')
lgb_rmse = -lgb_scores.mean()
lgb_score = 100 * (1 - lgb_rmse)

print(f"LightGBM CV RMSE: {lgb_rmse:.4f}")
print(f"Estimated Score: {lgb_score:.2f}")


rf = RandomForestRegressor(random_state=42, n_jobs=-1)

random_search = RandomizedSearchCV(
    rf, param_distributions=param_grid,
    n_iter=20, cv=3, scoring='neg_root_mean_squared_error', verbose=2, random_state=42
)

random_search.fit(X_train, y_train)


# Use the best model from tuning
rf_best = random_search.best_estimator_

# 🌿 Option 2: Feature Importance Pruning
importances = rf_best.feature_importances_
features = X_train.columns
feature_importance_df = pd.DataFrame({'feature': features, 'importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

# Select top 20 features
top_features = feature_importance_df['feature'].head(20).tolist()

# Use only top features for all models
X_train_top = X_train[top_features]
X_test_top = X_test[top_features]


# Cross-validated RMSE
best_rmse = -random_search.best_score_
competition_score = 100 * (1 - best_rmse)

print(f"Best RMSE: {best_rmse:.4f}")
print(f"Best Params: {random_search.best_params_}")
print(f"Estimated Competition Score: {competition_score:.2f}")

# Retrain Random Forest on top features
rf_best.fit(X_train_top, y_train)
rf_preds = rf_best.predict(X_test_top)

# Fit LightGBM on top features
lgb_model.fit(X_train_top, y_train)
lgb_preds = lgb_model.predict(X_test_top)

# ⚙️ Option 3: XGBoost
import xgboost as xgb
xgb_model = xgb.XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(X_train_top, y_train)
xgb_preds = xgb_model.predict(X_test_top)


from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Prepare OOF arrays
oof_rf = np.zeros(X_train_top.shape[0])
oof_lgb = np.zeros(X_train_top.shape[0])
oof_xgb = np.zeros(X_train_top.shape[0])

# Test preds for averaging
test_preds_rf = np.zeros(X_test_top.shape[0])
test_preds_lgb = np.zeros(X_test_top.shape[0])
test_preds_xgb = np.zeros(X_test_top.shape[0])


# Generate OOF predictions
for train_idx, valid_idx in kf.split(X_train_top):
    X_tr, X_val = X_train_top.iloc[train_idx], X_train_top.iloc[valid_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[valid_idx]
    
    rf_best.fit(X_tr, y_tr)
    lgb_model.fit(X_tr, y_tr)
    xgb_model.fit(X_tr, y_tr)

    oof_rf[valid_idx] = rf_best.predict(X_val)
    oof_lgb[valid_idx] = lgb_model.predict(X_val)
    oof_xgb[valid_idx] = xgb_model.predict(X_val)

    test_preds_rf += rf_best.predict(X_test_top) / kf.n_splits
    test_preds_lgb += lgb_model.predict(X_test_top) / kf.n_splits
    test_preds_xgb += xgb_model.predict(X_test_top) / kf.n_splits

# Stack training features from OOF preds
stack_X_train = np.vstack([oof_rf, oof_lgb, oof_xgb]).T
stack_y_train = y_train

# Stack test features from averaged predictions
stack_X_test = np.vstack([test_preds_rf, test_preds_lgb, test_preds_xgb]).T

meta_model = RidgeCV()
meta_model.fit(stack_X_train, stack_y_train)
stacked_preds = meta_model.predict(stack_X_test)
# Undo reflected log1p
stacked_preds = y_raw.max() - np.expm1(stacked_preds)




submission_stacked = pd.DataFrame({
    "id": test["id"],
    "efficiency": stacked_preds
})
submission_stacked.to_csv("final_ridge_stacked_submission.csv", index=False)
print("Final submission saved as final_ridge_stacked_submission.csv")






