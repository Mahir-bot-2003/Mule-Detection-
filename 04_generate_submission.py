#!/usr/bin/env python
"""
Step 4: Generate Submission
============================
Takes model predictions and generates the final submission CSV
with temporal suspicious activity windows.

Input:  output/predictions.parquet
        output/monthly_txn_counts.parquet
        output/monthly_txn_amounts.parquet
Output: submission.csv

Run: python 04_generate_submission.py
"""
import pandas as pd
import numpy as np
import os
import time
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(DATA_DIR)
OUTPUT_DIR = os.path.join(DATA_DIR, 'output')


# ============================================================
# TEMPORAL WINDOW DETECTION
# ============================================================

def detect_suspicious_window(account_id, monthly_counts, monthly_amounts, threshold_multiplier=2.0):
    """
    Detect the suspicious activity window for a single account.

    Strategy:
    1. Look at monthly transaction counts/amounts
    2. Compute rolling baseline (median of all months)
    3. Flag months where activity exceeds baseline * threshold
    4. The suspicious window = first flagged month to last flagged month

    Returns: (suspicious_start, suspicious_end) as ISO timestamps, or (None, None)
    """
    if account_id not in monthly_counts.index:
        return None, None

    counts = monthly_counts.loc[account_id]
    amounts = monthly_amounts.loc[account_id] if account_id in monthly_amounts.index else counts

    # Filter to non-zero months
    active_counts = counts[counts > 0]
    if len(active_counts) < 3:
        # Too few active months to detect anomaly, use full range
        if len(active_counts) > 0:
            months = sorted(active_counts.index)
            start = pd.Period(months[0], freq='M').start_time
            end = pd.Period(months[-1], freq='M').end_time
            return start.isoformat(), end.isoformat()
        return None, None

    # Compute baseline (median of active months)
    baseline_count = active_counts.median()
    baseline_amount = amounts[amounts > 0].median() if (amounts > 0).any() else 0

    # Flag anomalous months (count OR amount exceeds threshold)
    anomalous = (
        (counts > baseline_count * threshold_multiplier) |
        (amounts > baseline_amount * threshold_multiplier)
    )

    flagged_months = sorted([m for m in anomalous.index if anomalous[m]])

    if not flagged_months:
        # No clear anomaly — look for the densest cluster of activity
        # Use the period with the highest transaction volume
        if len(active_counts) > 0:
            peak_month = active_counts.idxmax()
            start = pd.Period(peak_month, freq='M').start_time
            end = pd.Period(peak_month, freq='M').end_time
            return start.isoformat(), end.isoformat()
        return None, None

    # Suspicious window = first flagged to last flagged
    start = pd.Period(flagged_months[0], freq='M').start_time
    end = pd.Period(flagged_months[-1], freq='M').end_time

    return start.isoformat(), end.isoformat()


def generate_temporal_windows(predictions, monthly_counts, monthly_amounts, mule_threshold=0.5):
    """
    Generate suspicious_start and suspicious_end for all predicted mule accounts.
    """
    print("\n" + "="*60)
    print("TEMPORAL WINDOW DETECTION")
    print("="*60)

    # Only detect windows for predicted mules
    predicted_mules = predictions[predictions['is_mule'] >= mule_threshold]
    print(f"  Accounts with is_mule >= {mule_threshold}: {len(predicted_mules):,}")

    windows = []
    for idx, row in predicted_mules.iterrows():
        account_id = row['account_id']
        start, end = detect_suspicious_window(account_id, monthly_counts, monthly_amounts)
        windows.append({
            'account_id': account_id,
            'suspicious_start': start,
            'suspicious_end': end
        })

    windows_df = pd.DataFrame(windows)
    n_with_windows = windows_df['suspicious_start'].notna().sum()
    print(f"  Accounts with detected windows: {n_with_windows:,}")

    return windows_df


# ============================================================
# GENERATE SUBMISSION CSV
# ============================================================

def main():
    print("="*60)
    print("AML MULE ACCOUNT DETECTION — SUBMISSION GENERATION")
    print("="*60)

    # Load predictions
    print("\nLoading predictions...")
    predictions = pd.read_parquet(os.path.join(OUTPUT_DIR, 'predictions.parquet'))
    print(f"  Predictions: {len(predictions):,} accounts")
    print(f"  is_mule stats: min={predictions['is_mule'].min():.4f}, "
          f"max={predictions['is_mule'].max():.4f}, "
          f"mean={predictions['is_mule'].mean():.4f}")

    # Load test accounts to ensure all are present
    test_accounts = pd.read_parquet('test_accounts.parquet')
    print(f"  Test accounts: {len(test_accounts):,}")

    # Load monthly stats for temporal windows
    monthly_counts_path = os.path.join(OUTPUT_DIR, 'monthly_txn_counts.parquet')
    monthly_amounts_path = os.path.join(OUTPUT_DIR, 'monthly_txn_amounts.parquet')

    if os.path.exists(monthly_counts_path) and os.path.exists(monthly_amounts_path):
        print("  Loading monthly transaction stats...")
        monthly_counts = pd.read_parquet(monthly_counts_path)
        monthly_amounts = pd.read_parquet(monthly_amounts_path)

        # Generate temporal windows
        windows = generate_temporal_windows(predictions, monthly_counts, monthly_amounts)

        # Merge predictions with windows
        submission = predictions.merge(
            windows[['account_id', 'suspicious_start', 'suspicious_end']],
            on='account_id',
            how='left'
        )
    else:
        print("  WARNING: Monthly stats not found, skipping temporal windows")
        submission = predictions.copy()
        submission['suspicious_start'] = ''
        submission['suspicious_end'] = ''

    # Ensure all test accounts are present
    submission = test_accounts[['account_id']].merge(
        submission, on='account_id', how='left'
    )
    submission['is_mule'] = submission['is_mule'].fillna(0.5)  # Default for missing
    submission['suspicious_start'] = submission['suspicious_start'].fillna('')
    submission['suspicious_end'] = submission['suspicious_end'].fillna('')

    # For accounts predicted as legitimate, clear the time windows
    legit_mask = submission['is_mule'] < 0.5
    submission.loc[legit_mask, 'suspicious_start'] = ''
    submission.loc[legit_mask, 'suspicious_end'] = ''

    # Ensure correct column order
    submission = submission[['account_id', 'is_mule', 'suspicious_start', 'suspicious_end']]

    # Save
    output_path = os.path.join(DATA_DIR, 'submission.csv')
    submission.to_csv(output_path, index=False)

    print(f"\n{'='*60}")
    print(f"SUBMISSION SAVED: {output_path}")
    print(f"{'='*60}")
    print(f"  Total accounts: {len(submission):,}")
    print(f"  Predicted mules (>0.5): {(submission['is_mule'] > 0.5).sum():,}")
    print(f"  With time windows: {(submission['suspicious_start'] != '').sum():,}")
    print(f"  is_mule range: [{submission['is_mule'].min():.4f}, {submission['is_mule'].max():.4f}]")
    print(f"\n  Preview:")
    print(submission.head(10).to_string(index=False))

    # Validation checks
    print(f"\n  VALIDATION:")
    print(f"  ✓ Row count matches test_accounts: {len(submission) == len(test_accounts)}")
    print(f"  ✓ All account_ids present: {set(test_accounts['account_id']).issubset(set(submission['account_id']))}")
    print(f"  ✓ is_mule in [0,1]: {submission['is_mule'].between(0, 1).all()}")
    print(f"  ✓ No null account_ids: {submission['account_id'].notna().all()}")


if __name__ == '__main__':
    main()
