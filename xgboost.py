import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif

# Load datasets
df1 = pd.read_csv('train_set.csv')
df2 = pd.read_csv('test_set.csv')

# Drop 'row_id' if it's just an index and not a useful feature
df1.drop(columns=['row_id'], inplace=True, errors='ignore')
df2.drop(columns=['row_id'], inplace=True, errors='ignore')

# Identify categorical and numerical columns
categorical_cols = ['x84', 'x85', 'x86']
numerical_cols = [col for col in df1.columns if col not in categorical_cols + ['target']]

# Step 1: Handle Missing Values
# - Categorical: Fill with 'Unknown'
# - Numerical: Use Median Imputation

df1[categorical_cols] = df1[categorical_cols].fillna('Unknown')
df2[categorical_cols] = df2[categorical_cols].fillna('Unknown')

num_imputer = SimpleImputer(strategy='median')
df1[numerical_cols] = num_imputer.fit_transform(df1[numerical_cols])
df2[numerical_cols] = num_imputer.transform(df2[numerical_cols])

# Step 2: Encode Categorical Features
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
df1[categorical_cols] = encoder.fit_transform(df1[categorical_cols])
df2[categorical_cols] = encoder.transform(df2[categorical_cols])

# Step 3: Scale Features
scaler = StandardScaler()
df1[numerical_cols] = scaler.fit_transform(df1[numerical_cols])
df2[numerical_cols] = scaler.transform(df2[numerical_cols])

# drop target variable
X = df1.drop(columns=['target'])
y = df1['target']

# Step 5: Train XGBoost Model
# parameters found using optuna

model = xgb.XGBClassifier(
    learning_rate=0.008264802216051508, n_estimators=5323, max_depth=9,
    min_child_weight=2, subsample=0.805752593270096, colsample_bytree=0.9727775809160264,
    gamma=0.37720233821532273, reg_alpha=0.034822794674595654, reg_lambda=0.013746758992078839,
    scale_pos_weight=1, objective='binary:logistic', eval_metric='auc', seed=27 )

# Train model

model.fit(X, y)

# predict final values to submission.csv file

pred = model.predict_proba(df2)[:, 1]
print("Predictions on Test Set:", pred)
df_sample = pd.read_csv('sample_solution.csv')
df_sample['target'] = pred
df_sample.to_csv('sample_solution.csv', index=False)
