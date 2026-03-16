#!/usr/bin/env python
"""
Step 3: Model Training
=======================
Trains LightGBM + XGBoost ensemble on the engineered features.
Includes label noise detection (red herring avoidance) and
probability calibration.

Input:  output/features.parquet
Output: output/predictions.parquet (probabilities for ALL accounts)
        output/feature_importance.csv
        output/cv_results.txt

Run: python 03_train_model.py
"""
import pandas as pd
import numpy as np
import os
import json
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
import lightgbm as lgb
import xgboost as xgb

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(DATA_DIR)
OUTPUT_DIR = os.path.join(DATA_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_FOLDS = 5
SEED = 42


# ============================================================
# LOAD DATA
# ============================================================

def load_features():
    """Load the feature matrix and split into train/test."""
    print("Loading features from output/features.parquet...")
    features = pd.read_parquet(os.path.join(OUTPUT_DIR, 'features.parquet'))
    print(f"  Feature matrix shape: {features.shape}")

    # Load train labels and test accounts for splitting
    train_labels = pd.read_parquet('train_labels.parquet')
    test_accounts = pd.read_parquet('test_accounts.parquet')

    train_ids = set(train_labels['account_id'])
    test_ids = set(test_accounts['account_id'])

    # Split
    train_mask = features.index.isin(train_ids)
    test_mask = features.index.isin(test_ids)

    # Target column
    y_col = 'is_mule'

    # Feature columns (everything except target)
    feature_cols = [c for c in features.columns if c != y_col]

    X_train = features.loc[train_mask, feature_cols].copy()
    y_train = features.loc[train_mask, y_col].copy()
    X_test = features.loc[test_mask, feature_cols].copy()

    print(f"  Train: {X_train.shape}, positive rate: {y_train.mean():.4f}")
    print(f"  Test:  {X_test.shape}")
    print(f"  Features: {len(feature_cols)}")

    # Check for any remaining non-numeric columns
    non_numeric = X_train.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        print(f"  WARNING: Dropping non-numeric columns: {non_numeric}")
        X_train = X_train.drop(columns=non_numeric)
        X_test = X_test.drop(columns=non_numeric)
        feature_cols = [c for c in feature_cols if c not in non_numeric]

    return X_train, y_train, X_test, feature_cols


# ============================================================
# LABEL NOISE DETECTION (Red Herring Avoidance)
# ============================================================

def detect_noisy_labels(X_train, y_train, feature_cols):
    """
    Detect potentially mislabeled training examples using
    cross-validation confidence analysis.

    Strategy: Train a quick LightGBM model in CV, and for each sample,
    measure how confidently the model predicts the OPPOSITE of its label.
    High-confidence disagreements suggest noisy labels.

    Returns: sample_weights (lower weight for suspected noisy labels)
    """
    print("\n" + "="*60)
    print("LABEL NOISE DETECTION (Red Herring Avoidance)")
    print("="*60)

    # Out-of-fold predictions for label noise detection
    oof_preds = np.zeros(len(X_train))
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    quick_params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.1,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'n_jobs': -1,
        'seed': SEED,
    }

    print("  Running quick CV for label noise detection...")
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr = X_train.iloc[train_idx]
        y_tr = y_train.iloc[train_idx]
        X_val = X_train.iloc[val_idx]

        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dval = lgb.Dataset(X_val, label=y_train.iloc[val_idx])

        model = lgb.train(
            quick_params,
            dtrain,
            num_boost_round=200,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(0)]
        )

        oof_preds[val_idx] = model.predict(X_val)
        print(f"    Fold {fold+1}: AUC={roc_auc_score(y_train.iloc[val_idx], oof_preds[val_idx]):.4f}")

    # Identify suspicious labels
    # For mule labels (is_mule=1): if model very confidently predicts 0 → suspicious
    # For legitimate labels (is_mule=0): if model very confidently predicts 1 → suspicious
    confidence = np.abs(oof_preds - y_train.values)

    # Threshold for suspicion
    suspicious_mask = confidence > 0.8  # Model disagrees strongly with label
    n_suspicious = suspicious_mask.sum()
    n_suspicious_mule = ((y_train == 1) & suspicious_mask).sum()
    n_suspicious_legit = ((y_train == 0) & suspicious_mask).sum()

    print(f"\n  Suspicious labels detected: {n_suspicious:,} ({n_suspicious/len(y_train)*100:.2f}%)")
    print(f"    Mule labeled but predicted legitimate: {n_suspicious_mule:,}")
    print(f"    Legitimate labeled but predicted mule: {n_suspicious_legit:,}")

    # Create sample weights: lower weight for suspicious samples
    sample_weights = np.ones(len(y_train))
    sample_weights[suspicious_mask] = 0.3  # Reduce weight of suspected noisy labels

    print(f"  Sample weights: {(sample_weights < 1.0).sum():,} samples downweighted")

    # Save noise analysis
    noise_df = pd.DataFrame({
        'account_id': X_train.index,
        'true_label': y_train.values,
        'oof_pred': oof_preds,
        'confidence_error': confidence,
        'is_suspicious': suspicious_mask,
        'sample_weight': sample_weights
    })
    noise_df.to_csv(os.path.join(OUTPUT_DIR, 'label_noise_analysis.csv'), index=False)
    print(f"  Saved noise analysis to output/label_noise_analysis.csv")

    return sample_weights, oof_preds


# ============================================================
# MODEL TRAINING
# ============================================================

def train_lightgbm(X_train, y_train, X_test, feature_cols, sample_weights):
    """Train LightGBM with 5-fold stratified CV and early stopping."""
    print("\n" + "="*60)
    print("TRAINING LIGHTGBM")
    print("="*60)

    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 63,
        'learning_rate': 0.05,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.7,
        'bagging_freq': 5,
        'min_child_samples': 50,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'verbose': -1,
        'n_jobs': -1,
        'seed': SEED,
        'is_unbalance': True,
    }

    # Try GPU if available
    try:
        test_params = params.copy()
        test_params['device'] = 'gpu'
        test_params['gpu_platform_id'] = 0
        test_params['gpu_device_id'] = 0
        test_ds = lgb.Dataset(X_train.head(100), label=y_train.head(100))
        lgb.train(test_params, test_ds, num_boost_round=2, verbose_eval=False)
        params['device'] = 'gpu'
        params['gpu_platform_id'] = 0
        params['gpu_device_id'] = 0
        print("  Using GPU acceleration!")
    except Exception:
        print("  GPU not available for LightGBM, using CPU")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_preds_lgb = np.zeros(len(X_train))
    test_preds_lgb = np.zeros(len(X_test))
    feature_importance = np.zeros(len(feature_cols))
    models_lgb = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        print(f"\n  Fold {fold+1}/{N_FOLDS}")

        X_tr = X_train.iloc[train_idx]
        y_tr = y_train.iloc[train_idx]
        w_tr = sample_weights[train_idx]
        X_val = X_train.iloc[val_idx]
        y_val = y_train.iloc[val_idx]

        dtrain = lgb.Dataset(X_tr, label=y_tr, weight=w_tr)
        dval = lgb.Dataset(X_val, label=y_val)

        model = lgb.train(
            params,
            dtrain,
            num_boost_round=2000,
            valid_sets=[dval],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(100)
            ]
        )

        oof_preds_lgb[val_idx] = model.predict(X_val)
        test_preds_lgb += model.predict(X_test) / N_FOLDS
        feature_importance += model.feature_importance(importance_type='gain')
        models_lgb.append(model)

        val_auc = roc_auc_score(y_val, oof_preds_lgb[val_idx])
        # Best F1 across thresholds
        best_f1 = 0
        for thresh in np.arange(0.1, 0.9, 0.01):
            f1 = f1_score(y_val, (oof_preds_lgb[val_idx] > thresh).astype(int))
            best_f1 = max(best_f1, f1)
        print(f"    AUC: {val_auc:.4f} | Best F1: {best_f1:.4f} | Trees: {model.best_iteration}")

    # Overall OOF metrics
    overall_auc = roc_auc_score(y_train, oof_preds_lgb)
    best_f1_overall = 0
    best_thresh = 0.5
    for thresh in np.arange(0.1, 0.9, 0.01):
        f1 = f1_score(y_train, (oof_preds_lgb > thresh).astype(int))
        if f1 > best_f1_overall:
            best_f1_overall = f1
            best_thresh = thresh

    print(f"\n  LightGBM Overall OOF AUC:  {overall_auc:.4f}")
    print(f"  LightGBM Overall Best F1:  {best_f1_overall:.4f} (threshold={best_thresh:.2f})")

    # Feature importance
    feat_imp = pd.DataFrame({
        'feature': feature_cols,
        'importance': feature_importance / N_FOLDS
    }).sort_values('importance', ascending=False)
    feat_imp.to_csv(os.path.join(OUTPUT_DIR, 'feature_importance_lgb.csv'), index=False)
    print(f"\n  Top 20 features:")
    print(feat_imp.head(20).to_string(index=False))

    return oof_preds_lgb, test_preds_lgb, models_lgb, feat_imp


def train_xgboost(X_train, y_train, X_test, feature_cols, sample_weights):
    """Train XGBoost with 5-fold stratified CV."""
    print("\n" + "="*60)
    print("TRAINING XGBOOST")
    print("="*60)

    # Compute scale_pos_weight for imbalanced data
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / max(n_pos, 1)

    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'min_child_weight': 50,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'scale_pos_weight': scale_pos_weight,
        'seed': SEED,
        'nthread': -1,
    }

    # Try GPU
    try:
        test_params = params.copy()
        test_params['tree_method'] = 'gpu_hist'
        test_params['gpu_id'] = 0
        dtest = xgb.DMatrix(X_train.head(100), label=y_train.head(100))
        xgb.train(test_params, dtest, num_boost_round=2)
        params['tree_method'] = 'gpu_hist'
        params['gpu_id'] = 0
        print("  Using GPU acceleration!")
    except Exception:
        params['tree_method'] = 'hist'
        print("  GPU not available for XGBoost, using CPU")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_preds_xgb = np.zeros(len(X_train))
    test_preds_xgb = np.zeros(len(X_test))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        print(f"\n  Fold {fold+1}/{N_FOLDS}")

        X_tr = X_train.iloc[train_idx]
        y_tr = y_train.iloc[train_idx]
        w_tr = sample_weights[train_idx]
        X_val = X_train.iloc[val_idx]
        y_val = y_train.iloc[val_idx]

        dtrain = xgb.DMatrix(X_tr, label=y_tr, weight=w_tr)
        dval = xgb.DMatrix(X_val, label=y_val)
        dtest = xgb.DMatrix(X_test)

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=2000,
            evals=[(dval, 'val')],
            early_stopping_rounds=50,
            verbose_eval=100
        )

        oof_preds_xgb[val_idx] = model.predict(dval)
        test_preds_xgb += model.predict(dtest) / N_FOLDS

        val_auc = roc_auc_score(y_val, oof_preds_xgb[val_idx])
        print(f"    AUC: {val_auc:.4f} | Trees: {model.best_iteration}")

    overall_auc = roc_auc_score(y_train, oof_preds_xgb)
    best_f1_overall = 0
    for thresh in np.arange(0.1, 0.9, 0.01):
        f1 = f1_score(y_train, (oof_preds_xgb > thresh).astype(int))
        best_f1_overall = max(best_f1_overall, f1)

    print(f"\n  XGBoost Overall OOF AUC:  {overall_auc:.4f}")
    print(f"  XGBoost Overall Best F1:  {best_f1_overall:.4f}")

    return oof_preds_xgb, test_preds_xgb


# ============================================================
# ENSEMBLE & CALIBRATION
# ============================================================

def ensemble_and_calibrate(oof_lgb, test_lgb, oof_xgb, test_xgb, y_train):
    """Ensemble LightGBM + XGBoost predictions and calibrate probabilities."""
    print("\n" + "="*60)
    print("ENSEMBLE & PROBABILITY CALIBRATION")
    print("="*60)

    # Weighted average ensemble
    # Determine weights based on OOF AUC
    auc_lgb = roc_auc_score(y_train, oof_lgb)
    auc_xgb = roc_auc_score(y_train, oof_xgb)
    total_auc = auc_lgb + auc_xgb
    w_lgb = auc_lgb / total_auc
    w_xgb = auc_xgb / total_auc

    print(f"  LightGBM AUC: {auc_lgb:.4f} → weight: {w_lgb:.3f}")
    print(f"  XGBoost AUC:  {auc_xgb:.4f} → weight: {w_xgb:.3f}")

    oof_ensemble = w_lgb * oof_lgb + w_xgb * oof_xgb
    test_ensemble = w_lgb * test_lgb + w_xgb * test_xgb

    ensemble_auc = roc_auc_score(y_train, oof_ensemble)
    best_f1 = 0
    best_thresh = 0.5
    for thresh in np.arange(0.1, 0.9, 0.01):
        f1 = f1_score(y_train, (oof_ensemble > thresh).astype(int))
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    print(f"\n  Ensemble OOF AUC:  {ensemble_auc:.4f}")
    print(f"  Ensemble Best F1:  {best_f1:.4f} (threshold={best_thresh:.2f})")

    # Isotonic calibration
    print("  Applying isotonic calibration...")
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(oof_ensemble, y_train)
    test_calibrated = calibrator.predict(test_ensemble)

    # Clip to [0, 1]
    test_calibrated = np.clip(test_calibrated, 0, 1)

    print(f"  Calibrated test predictions: min={test_calibrated.min():.4f}, "
          f"max={test_calibrated.max():.4f}, mean={test_calibrated.mean():.4f}")

    return test_calibrated, oof_ensemble, best_thresh


# ============================================================
# MAIN
# ============================================================

def main():
    print("="*60)
    print("AML MULE ACCOUNT DETECTION — MODEL TRAINING")
    print("="*60)
    start_time = time.time()

    # Load features
    X_train, y_train, X_test, feature_cols = load_features()

    # Label noise detection
    sample_weights, _ = detect_noisy_labels(X_train, y_train, feature_cols)

    # Train LightGBM
    oof_lgb, test_lgb, models_lgb, feat_imp = train_lightgbm(
        X_train, y_train, X_test, feature_cols, sample_weights
    )

    # Train XGBoost
    oof_xgb, test_xgb = train_xgboost(
        X_train, y_train, X_test, feature_cols, sample_weights
    )

    # Ensemble & calibrate
    test_preds, oof_ensemble, best_thresh = ensemble_and_calibrate(
        oof_lgb, test_lgb, oof_xgb, test_xgb, y_train
    )

    # Save predictions
    predictions = pd.DataFrame({
        'account_id': X_test.index,
        'is_mule': test_preds
    })
    predictions.to_parquet(os.path.join(OUTPUT_DIR, 'predictions.parquet'), index=False)
    print(f"\n  Saved predictions to output/predictions.parquet")

    # Save OOF predictions for analysis
    oof_df = pd.DataFrame({
        'account_id': X_train.index,
        'true_label': y_train.values,
        'oof_lgb': oof_lgb,
        'oof_xgb': oof_xgb,
        'oof_ensemble': oof_ensemble
    })
    oof_df.to_csv(os.path.join(OUTPUT_DIR, 'oof_predictions.csv'), index=False)

    # Save CV results summary
    cv_results = {
        'lgb_auc': roc_auc_score(y_train, oof_lgb),
        'xgb_auc': roc_auc_score(y_train, oof_xgb),
        'ensemble_auc': roc_auc_score(y_train, oof_ensemble),
        'best_f1_threshold': best_thresh,
        'n_features': len(feature_cols),
        'n_train': len(X_train),
        'n_test': len(X_test),
        'mule_rate': y_train.mean(),
    }
    with open(os.path.join(OUTPUT_DIR, 'cv_results.json'), 'w') as f:
        json.dump(cv_results, f, indent=2, default=str)

    total_time = time.time() - start_time
    print(f"\nMODEL TRAINING COMPLETE in {total_time/60:.1f} minutes")
    print(f"Next step: python 04_generate_submission.py")


if __name__ == '__main__':
    main()
