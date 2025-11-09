import pandas as pd
import numpy as np
import glob
import pickle
import json
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# sklearn imports
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    average_precision_score, precision_recall_curve, roc_auc_score,
    PrecisionRecallDisplay
)

# Imbalanced learning
from imblearn.over_sampling import SMOTE

# XGBoost
from xgboost import XGBClassifier

# Visualization
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

# Set random seeds for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("="*70)
print("FRAUD DETECTION MODEL TRAINING")
print("="*70)

# STEP 1: LOAD DATA FROM .PKL FILES
print("\n[1/9] Loading data from .pkl files...")

data_folder = "data/"
pkl_files = sorted(glob.glob(os.path.join(data_folder, "*.pkl")))

if not pkl_files:
    raise FileNotFoundError(f"No .pkl files found in {data_folder}")

print(f"Found {len(pkl_files)} .pkl files")

# Load and concatenate all dataframes
dfs = []
for f in pkl_files:
    try:
        df_temp = pd.read_pickle(f)
        dfs.append(df_temp)
        print(f"  ✓ Loaded {os.path.basename(f)}: {len(df_temp)} rows")
    except Exception as e:
        print(f"  ✗ Error loading {f}: {e}")

df = pd.concat(dfs, ignore_index=True)
print(f"\nTotal records loaded: {len(df):,}")

# Basic validation
print("\n" + "="*70)
print("DATASET OVERVIEW")
print("="*70)
print(df.info())
print("\nFirst few rows:")
print(df.head())
print("\nBasic statistics:")
print(df.describe())

# Check for duplicates
if 'TRANSACTION_ID' in df.columns:
    duplicates = df.duplicated(subset='TRANSACTION_ID').sum()
    if duplicates > 0:
        print(f"\n⚠ Found {duplicates} duplicate TRANSACTION_IDs - removing...")
        df = df.drop_duplicates(subset='TRANSACTION_ID', keep='first')

# STEP 2: PREPROCESS DATETIME AND SORT
print("\n[2/9] Processing datetime and sorting...")

df['TX_DATETIME'] = pd.to_datetime(df['TX_DATETIME'], errors='coerce')
df = df.dropna(subset=['TX_DATETIME'])  # Remove rows with invalid dates
df = df.sort_values('TX_DATETIME').reset_index(drop=True)

print(f"Date range: {df['TX_DATETIME'].min()} to {df['TX_DATETIME'].max()}")

# Check class distribution
fraud_count = df['TX_FRAUD'].sum()
fraud_rate = fraud_count / len(df) * 100
print(f"\nClass distribution:")
print(f"  Legitimate: {len(df) - fraud_count:,} ({100-fraud_rate:.2f}%)")
print(f"  Fraudulent: {fraud_count:,} ({fraud_rate:.2f}%)")
print(f"  ⚠ Class imbalance ratio: 1:{(len(df) - fraud_count) / fraud_count:.0f}")

# STEP 3: FEATURE ENGINEERING
print("\n[3/9] Engineering features...")

#Time-based features
df['hour'] = df['TX_DATETIME'].dt.hour
df['day_of_week'] = df['TX_DATETIME'].dt.dayofweek
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
df['day_of_month'] = df['TX_DATETIME'].dt.day
df['month'] = df['TX_DATETIME'].dt.month

#Amount-based features
df['amount_log'] = np.log1p(df['TX_AMOUNT'])

# Customer aggregation features
print("  Computing customer aggregations...")
customer_agg = df.groupby('CUSTOMER_ID').agg({
    'TX_AMOUNT': ['mean', 'std', 'count'],
    'TX_FRAUD': 'sum'
}).reset_index()
customer_agg.columns = ['CUSTOMER_ID', 'cust_avg_amount', 'cust_std_amount', 
                        'cust_tx_count', 'cust_fraud_count']
customer_agg['cust_std_amount'] = customer_agg['cust_std_amount'].fillna(0)

df = df.merge(customer_agg, on='CUSTOMER_ID', how='left')

#Terminal aggregation features
print("  Computing terminal aggregations...")
terminal_agg = df.groupby('TERMINAL_ID').agg({
    'TX_AMOUNT': ['mean', 'count'],
    'TX_FRAUD': 'sum'
}).reset_index()
terminal_agg.columns = ['TERMINAL_ID', 'terminal_avg_amount', 
                        'terminal_tx_count', 'terminal_fraud_count']

df = df.merge(terminal_agg, on='TERMINAL_ID', how='left')

#Ratio features
df['amount_to_cust_avg_ratio'] = df['TX_AMOUNT'] / (df['cust_avg_amount'] + 1e-5)
df['amount_to_terminal_avg_ratio'] = df['TX_AMOUNT'] / (df['terminal_avg_amount'] + 1e-5)

#Transaction velocity features (simplified rolling window)
print("  Computing transaction velocity features...")
df = df.sort_values(['CUSTOMER_ID', 'TX_DATETIME'])
df['cust_tx_count_24h'] = df.groupby('CUSTOMER_ID').cumcount()

df = df.sort_values(['TERMINAL_ID', 'TX_DATETIME'])
df['terminal_tx_count_24h'] = df.groupby('TERMINAL_ID').cumcount()

# Sort back by time
df = df.sort_values('TX_DATETIME').reset_index(drop=True)

print(f"  ✓ Created {len(df.columns) - len(dfs[0].columns)} new features")

# STEP 4: PREPARE FEATURES AND TARGET
print("\n[4/9] Preparing features and target...")

# Define feature columns (exclude target, IDs, and datetime)
feature_cols = [
    'TX_AMOUNT', 'amount_log',
    'hour', 'day_of_week', 'is_weekend', 'day_of_month', 'month',
    'cust_avg_amount', 'cust_std_amount', 'cust_tx_count', 'cust_fraud_count',
    'terminal_avg_amount', 'terminal_tx_count', 'terminal_fraud_count',
    'amount_to_cust_avg_ratio', 'amount_to_terminal_avg_ratio',
    'cust_tx_count_24h', 'terminal_tx_count_24h'
]

X = df[feature_cols].copy()
y = df['TX_FRAUD'].copy()

# Handle any remaining NaNs
X = X.fillna(0)

print(f"Feature matrix shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"\nFeatures used:\n  " + "\n  ".join(feature_cols))

# STEP 5: TRAIN-TEST SPLIT
print("\n[5/9] Splitting data (80% train, 20% test)...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

print(f"Training set: {X_train.shape[0]:,} samples")
print(f"Test set: {X_test.shape[0]:,} samples")
print(f"Train fraud rate: {y_train.sum() / len(y_train) * 100:.2f}%")
print(f"Test fraud rate: {y_test.sum() / len(y_test) * 100:.2f}%")

# STEP 6: SCALING
print("\n[6/9] Scaling features...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# STEP 7: HANDLE CLASS IMBALANCE WITH SMOTE
print("\n[7/9] Applying SMOTE to training data...")

print(f"Before SMOTE - Training samples: {len(X_train_scaled):,}")
print(f"  Legitimate: {(y_train == 0).sum():,}")
print(f"  Fraud: {(y_train == 1).sum():,}")

smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

print(f"\nAfter SMOTE - Training samples: {len(X_train_resampled):,}")
print(f"  Legitimate: {(y_train_resampled == 0).sum():,}")
print(f"  Fraud: {(y_train_resampled == 1).sum():,}")

# STEP 8: TRAIN XGBOOST MODEL
print("\n[8/9] Training XGBoost classifier...")

model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=1,  # Already balanced with SMOTE
    random_state=RANDOM_STATE,
    eval_metric='aucpr',
    early_stopping_rounds=10,
    n_jobs=-1
)

# Train with validation set for early stopping
eval_set = [(X_test_scaled, y_test)]
model.fit(
    X_train_resampled, y_train_resampled,
    eval_set=eval_set,
    verbose=False
)

print("  ✓ Model training complete")

# STEP 9: EVALUATE MODEL
print("\n[9/9] Evaluating model on test set...")

# Predictions
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

# Metrics
print("\n" + "="*70)
print("CLASSIFICATION REPORT")
print("="*70)
print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud']))

# Important metrics
auprc = average_precision_score(y_test, y_pred_proba)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print("\n" + "="*70)
print("KEY METRICS")
print("="*70)
print(f"Area Under Precision-Recall Curve (AUPRC): {auprc:.4f}")
print(f"ROC-AUC Score: {roc_auc:.4f}")

# SAVE ARTIFACTS
print("\nSaving model and artifacts...")

os.makedirs("artifacts", exist_ok=True)

# 1. Save model and scaler as pipeline
model_pipeline = {
    'model': model,
    'scaler': scaler,
    'feature_names': feature_cols,
    'training_date': datetime.now().isoformat(),
    'random_state': RANDOM_STATE
}

with open('fraud_detection_model.pkl', 'wb') as f:
    pickle.dump(model_pipeline, f)
print("  ✓ Saved fraud_detection_model.pkl")

# 2. Save feature list
with open('artifacts/feature_list.json', 'w') as f:
    json.dump(feature_cols, f, indent=2)
print("  ✓ Saved artifacts/feature_list.json")

# 3. Save evaluation metrics
metrics = {
    'auprc': float(auprc),
    'roc_auc': float(roc_auc),
    'test_samples': int(len(y_test)),
    'fraud_count': int(y_test.sum()),
    'feature_count': len(feature_cols)
}

with open('artifacts/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)
print("  ✓ Saved artifacts/metrics.json")

# 4. Confusion Matrix
plt.figure(figsize=(8, 6))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap='Blues')
plt.title('Confusion Matrix - Test Set')
plt.tight_layout()
plt.savefig('artifacts/confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved artifacts/confusion_matrix.png")

# 5. Precision-Recall Curve
plt.figure(figsize=(8, 6))
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
plt.plot(recall, precision, linewidth=2, label=f'AUPRC = {auprc:.3f}')
plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Precision-Recall Curve', fontsize=14)
plt.grid(alpha=0.3)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig('artifacts/precision_recall_curve.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved artifacts/precision_recall_curve.png")

# 6. Feature Importance
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 8))
plt.barh(range(len(feature_importance)), feature_importance['importance'])
plt.yticks(range(len(feature_importance)), feature_importance['feature'])
plt.xlabel('Importance', fontsize=12)
plt.title('Feature Importance (XGBoost Gain)', fontsize=14)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('artifacts/feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved artifacts/feature_importance.png")

feature_importance.to_csv('artifacts/feature_importance.csv', index=False)

print("\n" + "="*70)
print("✓ TRAINING COMPLETE!")
print("="*70)
print(f"Model saved to: fraud_detection_model.pkl")
print(f"Artifacts saved to: artifacts/")
print(f"\nTop 5 Most Important Features:")
for idx, row in feature_importance.head(5).iterrows():
    print(f"  {row['feature']}: {row['importance']:.4f}")