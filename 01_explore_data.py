#!/usr/bin/env python
"""
Step 1: Exploratory Data Analysis (EDA)
========================================
Quick data profiling of all files to understand shapes, types, distributions,
and class imbalance BEFORE building features.

Run: python 01_explore_data.py
"""
import pandas as pd
import numpy as np
from glob import glob
import os
import sys

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(DATA_DIR)

def separator(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")

def profile_df(name, df):
    """Print key statistics for a DataFrame."""
    print(f"\n--- {name} ---")
    print(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Dtypes:")
    for col in df.columns:
        null_count = df[col].isnull().sum()
        null_pct = null_count / len(df) * 100
        print(f"    {col:40s} {str(df[col].dtype):15s} nulls: {null_count:,} ({null_pct:.1f}%)")
    print(f"\n  First 3 rows:")
    print(df.head(3).to_string(index=False))


def main():
    separator("LOADING ALL SMALL PARQUET FILES")

    # ---- Load each file and profile it ----
    files = {
        'customers':              'customers.parquet',
        'accounts':               'accounts.parquet',
        'demographics':           'demographics.parquet',
        'branch':                 'branch.parquet',
        'product_details':        'product_details.parquet',
        'accounts_additional':    'accounts-additional.parquet',
        'customer_account_linkage': 'customer_account_linkage.parquet',
        'train_labels':           'train_labels.parquet',
        'test_accounts':          'test_accounts.parquet',
    }

    dfs = {}
    for name, path in files.items():
        full_path = os.path.join(DATA_DIR, path)
        if not os.path.exists(full_path):
            print(f"  WARNING: {path} not found!")
            continue
        dfs[name] = pd.read_parquet(full_path)
        profile_df(name, dfs[name])

    # ---- Class imbalance analysis ----
    separator("CLASS IMBALANCE (train_labels)")
    if 'train_labels' in dfs:
        labels = dfs['train_labels']
        print(f"  Total training accounts: {len(labels):,}")
        print(f"  Mule accounts (is_mule=1): {labels['is_mule'].sum():,}")
        print(f"  Legitimate accounts (is_mule=0): {(labels['is_mule'] == 0).sum():,}")
        print(f"  Mule rate: {labels['is_mule'].mean():.4f} ({labels['is_mule'].mean()*100:.2f}%)")

        if 'alert_reason' in labels.columns:
            print(f"\n  Alert reason distribution:")
            print(labels['alert_reason'].value_counts(dropna=False).to_string())

        if 'mule_flag_date' in labels.columns:
            mule_dates = pd.to_datetime(labels.loc[labels['is_mule']==1, 'mule_flag_date'], errors='coerce')
            print(f"\n  Mule flag date range: {mule_dates.min()} to {mule_dates.max()}")

        if 'flagged_by_branch' in labels.columns:
            print(f"\n  Unique flagging branches: {labels['flagged_by_branch'].nunique()}")
            print(f"  Top 10 flagging branches:")
            print(labels['flagged_by_branch'].value_counts().head(10).to_string())

    # ---- Test accounts ----
    separator("TEST ACCOUNTS")
    if 'test_accounts' in dfs:
        print(f"  Test accounts to predict: {len(dfs['test_accounts']):,}")
        print(f"  Sample: {dfs['test_accounts'].head(5).to_string(index=False)}")

    # ---- Account overlap check ----
    separator("ACCOUNT OVERLAP CHECK")
    if 'train_labels' in dfs and 'test_accounts' in dfs:
        train_ids = set(dfs['train_labels']['account_id'])
        test_ids = set(dfs['test_accounts']['account_id'])
        overlap = train_ids & test_ids
        print(f"  Train account IDs: {len(train_ids):,}")
        print(f"  Test account IDs:  {len(test_ids):,}")
        print(f"  Overlap:           {len(overlap)}")
        print(f"  Total unique:      {len(train_ids | test_ids):,}")

    # ---- Transactions sample ----
    separator("TRANSACTION FILES STRUCTURE")
    for txn_dir in ['transactions', 'transactions_additional']:
        parts = sorted(glob(os.path.join(DATA_DIR, txn_dir, 'batch-*', 'part_*.parquet')))
        print(f"\n  {txn_dir}/: {len(parts)} part files")

        # Count per batch
        for b in sorted(glob(os.path.join(DATA_DIR, txn_dir, 'batch-*/'))):
            batch_parts = glob(os.path.join(b, 'part_*.parquet'))
            print(f"    {os.path.basename(os.path.normpath(b))}: {len(batch_parts)} parts")

        if parts:
            # Read first part to see schema
            sample = pd.read_parquet(parts[0])
            profile_df(f"{txn_dir} (first part)", sample)

            # Estimate total rows
            rows_in_first = len(sample)
            estimated_total = rows_in_first * len(parts)
            print(f"\n  Estimated total rows: ~{estimated_total:,}")
            del sample

    # ---- Key numeric distributions ----
    separator("KEY DISTRIBUTIONS")
    if 'accounts' in dfs:
        accts = dfs['accounts']
        print(f"\n  Account balance stats:")
        for col in ['avg_balance', 'monthly_avg_balance', 'quarterly_avg_balance', 'daily_avg_balance']:
            if col in accts.columns:
                print(f"    {col}: mean={accts[col].mean():.2f}, median={accts[col].median():.2f}, "
                      f"min={accts[col].min():.2f}, max={accts[col].max():.2f}")

        print(f"\n  Product family distribution:")
        if 'product_family' in accts.columns:
            print(accts['product_family'].value_counts().to_string())

        print(f"\n  Account status distribution:")
        if 'account_status' in accts.columns:
            print(accts['account_status'].value_counts().to_string())

    if 'customers' in dfs:
        custs = dfs['customers']
        print(f"\n  KYC document availability:")
        for col in ['pan_available', 'aadhaar_available', 'passport_available']:
            if col in custs.columns:
                print(f"    {col}: {custs[col].value_counts().to_dict()}")

    separator("EDA COMPLETE")
    print("  Review the output above and then run: python 02_feature_engineering.py")


if __name__ == '__main__':
    main()
