#!/usr/bin/env python
"""
Step 2: Feature Engineering Pipeline
=====================================
Processes ALL data (static files + 400M transactions) and creates
per-account features for the ML model.

Memory strategy: processes transaction part files ONE AT A TIME,
aggregates per-account statistics, and accumulates across parts.

Output: output/features.parquet (160K accounts × N features)
        output/monthly_txn_stats.parquet (for temporal window prediction)

Run: python 02_feature_engineering.py
"""
import pandas as pd
import numpy as np
from glob import glob
import os
import gc
import time
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(DATA_DIR)
OUTPUT_DIR = os.path.join(DATA_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Reference date = end of data window
REF_DATE = pd.Timestamp('2025-06-30')

# All known transaction channels (from README)
ALL_CHANNELS = [
    'UPC','UPD','END','IPM','STD','P2A','FTD','NTD','MCR','FTC',
    'MAC','TPD','APD','CHQ','ATW','TPC','STC','OCD','RCD','IFD',
    'ETD','NWD','CSD','IFC','PCA','MAD','CHD','RTD','CCL','OPI',
    'CTC','SID','ASD','IAD','SCW'
]


# ============================================================
# PHASE 1: STATIC FEATURES (from small parquet files)
# ============================================================

def load_static_features():
    """Load all small files, merge by account_id, create static features."""
    print("\n" + "="*60)
    print("PHASE 1: Loading static data and creating static features")
    print("="*60)

    # --- Load all small files ---
    print("  Loading customers.parquet...")
    customers = pd.read_parquet('customers.parquet')
    print(f"    → {len(customers):,} rows, columns: {list(customers.columns)}")

    print("  Loading accounts.parquet...")
    accounts = pd.read_parquet('accounts.parquet')
    print(f"    → {len(accounts):,} rows, columns: {list(accounts.columns)}")

    print("  Loading demographics.parquet...")
    demographics = pd.read_parquet('demographics.parquet')
    print(f"    → {len(demographics):,} rows, columns: {list(demographics.columns)}")

    print("  Loading branch.parquet...")
    branch = pd.read_parquet('branch.parquet')
    print(f"    → {len(branch):,} rows, columns: {list(branch.columns)}")

    print("  Loading product_details.parquet...")
    product_details = pd.read_parquet('product_details.parquet')
    print(f"    → {len(product_details):,} rows, columns: {list(product_details.columns)}")

    print("  Loading accounts-additional.parquet...")
    accounts_add = pd.read_parquet('accounts-additional.parquet')
    print(f"    → {len(accounts_add):,} rows, columns: {list(accounts_add.columns)}")

    print("  Loading customer_account_linkage.parquet...")
    linkage = pd.read_parquet('customer_account_linkage.parquet')
    print(f"    → {len(linkage):,} rows, columns: {list(linkage.columns)}")

    print("  Loading train_labels.parquet...")
    train_labels = pd.read_parquet('train_labels.parquet')
    print(f"    → {len(train_labels):,} rows, columns: {list(train_labels.columns)}")

    print("  Loading test_accounts.parquet...")
    test_accounts = pd.read_parquet('test_accounts.parquet')
    print(f"    → {len(test_accounts):,} rows, columns: {list(test_accounts.columns)}")

    # --- Build base DataFrame with ALL account_ids ---
    all_account_ids = pd.concat([
        train_labels[['account_id']],
        test_accounts[['account_id']]
    ]).drop_duplicates().reset_index(drop=True)
    print(f"\n  Total unique accounts (train + test): {len(all_account_ids):,}")

    # --- Merge account data ---
    print("  Merging accounts...")
    df = all_account_ids.merge(accounts, on='account_id', how='left')

    # --- Merge customer via linkage ---
    print("  Merging customer_account_linkage → customers...")
    df = df.merge(linkage, on='account_id', how='left')
    df = df.merge(customers, on='customer_id', how='left')

    # --- Merge demographics ---
    print("  Merging demographics...")
    df = df.merge(demographics, on='customer_id', how='left')

    # --- Merge product details ---
    print("  Merging product_details...")
    df = df.merge(product_details, on='customer_id', how='left')

    # --- Merge branch ---
    print("  Merging branch...")
    df = df.merge(branch, on='branch_code', how='left')

    # --- Merge accounts additional ---
    print("  Merging accounts-additional...")
    df = df.merge(accounts_add, on='account_id', how='left')

    print(f"  Merged DataFrame shape: {df.shape}")

    # ---- ENGINEER STATIC FEATURES ----
    print("\n  Engineering static features...")

    # --- Account features ---
    # Account age in days
    df['account_opening_date'] = pd.to_datetime(df['account_opening_date'], errors='coerce')
    df['account_age_days'] = (REF_DATE - df['account_opening_date']).dt.days

    # Account status
    df['is_frozen'] = (df['account_status'] == 'frozen').astype(int)

    # Product family encoding
    for val in ['S', 'K', 'O']:
        df[f'product_family_{val}'] = (df['product_family'] == val).astype(int)

    # Balance features
    for col in ['avg_balance', 'monthly_avg_balance', 'quarterly_avg_balance', 'daily_avg_balance']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df['balance_monthly_daily_diff'] = df.get('monthly_avg_balance', 0) - df.get('daily_avg_balance', 0)
    if 'monthly_avg_balance' in df.columns and 'daily_avg_balance' in df.columns:
        df['balance_monthly_daily_ratio'] = df['monthly_avg_balance'] / df['daily_avg_balance'].replace(0, np.nan)

    # Binary flag conversions (Y/N → 1/0)
    yn_cols = [
        'nomination_flag', 'cheque_allowed', 'cheque_availed',
        'kyc_compliant', 'rural_branch',
        'pan_available', 'aadhaar_available', 'passport_available',
        'mobile_banking_flag', 'internet_banking_flag', 'atm_card_flag',
        'demat_flag', 'credit_card_flag', 'fastag_flag',
        'joint_account_flag', 'nri_flag'
    ]
    for col in yn_cols:
        if col in df.columns:
            df[f'{col}_num'] = (df[col] == 'Y').astype(int)

    # Number of cheque books
    if 'num_chequebooks' in df.columns:
        df['num_chequebooks'] = pd.to_numeric(df['num_chequebooks'], errors='coerce').fillna(0)

    # Freeze/unfreeze features
    df['freeze_date'] = pd.to_datetime(df.get('freeze_date'), errors='coerce')
    df['unfreeze_date'] = pd.to_datetime(df.get('unfreeze_date'), errors='coerce')
    df['was_frozen'] = df['freeze_date'].notna().astype(int)
    df['was_unfrozen'] = df['unfreeze_date'].notna().astype(int)
    df['freeze_duration_days'] = (df['unfreeze_date'] - df['freeze_date']).dt.days

    # Mobile update recency (Pattern 8: Post-Mobile-Change Spike)
    df['last_mobile_update_date'] = pd.to_datetime(df.get('last_mobile_update_date'), errors='coerce')
    df['days_since_mobile_update'] = (REF_DATE - df['last_mobile_update_date']).dt.days
    df['mobile_updated_recently'] = (df['days_since_mobile_update'] < 90).astype(int)

    # KYC recency
    df['last_kyc_date'] = pd.to_datetime(df.get('last_kyc_date'), errors='coerce')
    df['days_since_kyc'] = (REF_DATE - df['last_kyc_date']).dt.days

    # --- Customer features ---
    df['date_of_birth'] = pd.to_datetime(df.get('date_of_birth'), errors='coerce')
    df['customer_age_years'] = (REF_DATE - df['date_of_birth']).dt.days / 365.25

    df['relationship_start_date'] = pd.to_datetime(df.get('relationship_start_date'), errors='coerce')
    df['relationship_duration_days'] = (REF_DATE - df['relationship_start_date']).dt.days

    # KYC document count
    kyc_num_cols = [c for c in ['pan_available_num', 'aadhaar_available_num', 'passport_available_num'] if c in df.columns]
    if kyc_num_cols:
        df['kyc_doc_count'] = df[kyc_num_cols].sum(axis=1)

    # Digital banking adoption score
    digital_num_cols = [c for c in ['mobile_banking_flag_num', 'internet_banking_flag_num',
                                     'atm_card_flag_num', 'demat_flag_num',
                                     'credit_card_flag_num', 'fastag_flag_num'] if c in df.columns]
    if digital_num_cols:
        df['digital_adoption_score'] = df[digital_num_cols].sum(axis=1)

    # PIN code features
    if 'customer_pin' in df.columns and 'branch_pin' in df.columns:
        df['customer_pin'] = pd.to_numeric(df['customer_pin'], errors='coerce')
        df['branch_pin'] = pd.to_numeric(df['branch_pin'], errors='coerce')
        df['pin_match_branch'] = (df['customer_pin'] == df['branch_pin']).astype(int)
    if 'customer_pin' in df.columns and 'permanent_pin' in df.columns:
        df['permanent_pin'] = pd.to_numeric(df['permanent_pin'], errors='coerce')
        df['pin_match_permanent'] = (df['customer_pin'] == df['permanent_pin']).astype(int)

    # --- Demographics features ---
    if 'gender' in df.columns:
        df['is_male'] = (df['gender'] == 'M').astype(int)

    if 'address_last_update_date' in df.columns:
        df['address_last_update_date'] = pd.to_datetime(df['address_last_update_date'], errors='coerce')
        df['days_since_address_update'] = (REF_DATE - df['address_last_update_date']).dt.days

    if 'passbook_last_update_date' in df.columns:
        df['passbook_last_update_date'] = pd.to_datetime(df['passbook_last_update_date'], errors='coerce')
        df['days_since_passbook_update'] = (REF_DATE - df['passbook_last_update_date']).dt.days

    # --- Branch features ---
    if 'branch_type' in df.columns:
        for btype in ['urban', 'semi-urban', 'rural']:
            df[f'branch_type_{btype.replace("-","_")}'] = (df['branch_type'] == btype).astype(int)

    for col in ['branch_employee_count', 'branch_turnover', 'branch_asset_size']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # --- Product features ---
    product_count_cols = [c for c in ['loan_count', 'cc_count', 'od_count', 'ka_count', 'sa_count'] if c in df.columns]
    if product_count_cols:
        df['total_product_count'] = df[product_count_cols].sum(axis=1)

    product_sum_cols = [c for c in ['loan_sum', 'cc_sum', 'od_sum'] if c in df.columns]
    if product_sum_cols:
        df['total_outstanding'] = df[product_sum_cols].sum(axis=1)

    for col in product_count_cols + product_sum_cols + ['ka_sum', 'sa_sum']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if 'loan_count' in df.columns:
        df['has_loans'] = (df['loan_count'] > 0).astype(int)
    if 'cc_count' in df.columns:
        df['has_credit_cards'] = (df['cc_count'] > 0).astype(int)

    # --- Scheme features ---
    if 'scheme_code' in df.columns:
        scheme_dummies = pd.get_dummies(df['scheme_code'], prefix='scheme', dtype=int)
        df = pd.concat([df, scheme_dummies], axis=1)

    # --- Drop raw string/date columns (keep only numeric features) ---
    # We'll select numeric features at the end
    print(f"  Static features created. Total columns: {df.shape[1]}")

    return df, train_labels, test_accounts


# ============================================================
# PHASE 2: TRANSACTION FEATURES (batch-by-batch processing)
# ============================================================

def process_transactions():
    """Process all transaction part files and aggregate per-account features."""
    print("\n" + "="*60)
    print("PHASE 2: Processing transaction files (batch-by-batch)")
    print("="*60)

    all_parts = sorted(glob(os.path.join(DATA_DIR, 'transactions', 'batch-*', 'part_*.parquet')))
    print(f"  Found {len(all_parts)} transaction part files")

    if not all_parts:
        print("  WARNING: No transaction files found!")
        return pd.DataFrame()

    # ---- Accumulators ----
    # These DataFrames are indexed by account_id and accumulate stats across parts
    count_agg = None       # Counts, sums, min/max
    channel_counts = None  # Per-channel transaction counts
    monthly_counts = None  # Per year-month transaction counts (for temporal window)
    monthly_amounts = None # Per year-month total amounts

    total_rows = 0
    start_time = time.time()

    for i, part_file in enumerate(all_parts):
        t0 = time.time()
        try:
            df = pd.read_parquet(part_file, columns=[
                'account_id', 'transaction_timestamp', 'amount',
                'txn_type', 'channel', 'counterparty_id', 'mcc_code'
            ])
        except Exception as e:
            print(f"  ERROR reading {part_file}: {e}")
            continue

        rows = len(df)
        total_rows += rows

        # --- Parse timestamp and derive helper columns ---
        df['timestamp'] = pd.to_datetime(df['transaction_timestamp'], errors='coerce')
        df['abs_amount'] = df['amount'].abs()
        df['amount_sq'] = df['amount'] ** 2
        df['year_month'] = df['timestamp'].dt.to_period('M').astype(str)
        df['day_of_month'] = df['timestamp'].dt.day
        df['hour'] = df['timestamp'].dt.hour
        df['dayofweek'] = df['timestamp'].dt.dayofweek

        # --- Pattern detection flags ---
        # Pattern 2: Structuring (near ₹50K threshold)
        df['is_near_50k'] = ((df['abs_amount'] >= 45000) & (df['abs_amount'] <= 50000)).astype(int)

        # Pattern 9: Round amounts
        df['is_round_1k'] = (df['abs_amount'] % 1000 == 0).astype(int)
        df['is_round_5k'] = (df['abs_amount'] % 5000 == 0).astype(int)
        df['is_round_10k'] = (df['abs_amount'] % 10000 == 0).astype(int)
        df['is_round_50k'] = (df['abs_amount'] % 50000 == 0).astype(int)

        # Pattern 11: Month boundary transactions (salary cycle)
        df['is_month_boundary'] = ((df['day_of_month'] <= 5) | (df['day_of_month'] >= 26)).astype(int)

        # Night and weekend transactions
        df['is_night'] = ((df['hour'] >= 0) & (df['hour'] <= 6)).astype(int)
        df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)

        # --- Compute per-account aggregates for this part ---
        g = df.groupby('account_id')

        part_agg = pd.DataFrame({
            # Basic counts and sums
            'txn_count': g['amount'].count(),
            'total_amount': g['amount'].sum(),
            'total_abs_amount': g['abs_amount'].sum(),
            'sum_sq_amount': g['amount_sq'].sum(),
            'min_amount': g['amount'].min(),
            'max_amount': g['amount'].max(),
            'max_abs_amount': g['abs_amount'].max(),
            'min_timestamp': g['timestamp'].min(),
            'max_timestamp': g['timestamp'].max(),

            # Pattern flags
            'structuring_count': g['is_near_50k'].sum(),
            'round_1k_count': g['is_round_1k'].sum(),
            'round_5k_count': g['is_round_5k'].sum(),
            'round_10k_count': g['is_round_10k'].sum(),
            'round_50k_count': g['is_round_50k'].sum(),
            'month_boundary_count': g['is_month_boundary'].sum(),
            'night_txn_count': g['is_night'].sum(),
            'weekend_txn_count': g['is_weekend'].sum(),

            # Unique counts (approximate - will overcount across parts)
            'unique_counterparties_part': g['counterparty_id'].nunique(),
            'unique_mcc_part': g['mcc_code'].nunique(),
            'unique_channels_part': g['channel'].nunique(),
        })

        # Credit/Debit splits
        credits = df[df['txn_type'] == 'C'].groupby('account_id')
        debits = df[df['txn_type'] == 'D'].groupby('account_id')

        part_agg['credit_count'] = credits['amount'].count().reindex(part_agg.index, fill_value=0)
        part_agg['credit_total'] = credits['amount'].sum().reindex(part_agg.index, fill_value=0)
        part_agg['debit_count'] = debits['amount'].count().reindex(part_agg.index, fill_value=0)
        part_agg['debit_total'] = debits['amount'].sum().reindex(part_agg.index, fill_value=0)

        # Fan-in/Fan-out: unique counterparties per direction
        part_agg['unique_credit_cp'] = credits['counterparty_id'].nunique().reindex(part_agg.index, fill_value=0)
        part_agg['unique_debit_cp'] = debits['counterparty_id'].nunique().reindex(part_agg.index, fill_value=0)

        # --- Channel distribution ---
        part_channels = df.groupby(['account_id', 'channel']).size().unstack(fill_value=0)
        # Ensure all channels exist as columns
        for ch in ALL_CHANNELS:
            if ch not in part_channels.columns:
                part_channels[ch] = 0
        part_channels = part_channels[ALL_CHANNELS]  # Keep consistent column order
        part_channels.columns = [f'ch_{c}' for c in ALL_CHANNELS]

        # --- Monthly transaction counts (for temporal window prediction) ---
        part_monthly_count = df.groupby(['account_id', 'year_month']).size().unstack(fill_value=0)
        part_monthly_amount = df.groupby(['account_id', 'year_month'])['abs_amount'].sum().unstack(fill_value=0)

        # ---- ACCUMULATE across parts ----
        if count_agg is None:
            count_agg = part_agg
            channel_counts = part_channels
            monthly_counts = part_monthly_count
            monthly_amounts = part_monthly_amount
        else:
            # For counts/sums: add
            sum_cols = [c for c in part_agg.columns if c not in
                        ['min_amount', 'max_amount', 'max_abs_amount', 'min_timestamp', 'max_timestamp']]
            for col in sum_cols:
                count_agg[col] = count_agg.get(col, 0).add(part_agg[col], fill_value=0)

            # For numeric min: take element-wise min (handles NaN fine)
            for col in ['min_amount']:
                aligned_existing, aligned_new = count_agg[col].align(part_agg[col])
                count_agg[col] = np.fmin(aligned_existing, aligned_new)

            # For numeric max: take element-wise max (handles NaN fine)
            for col in ['max_amount', 'max_abs_amount']:
                aligned_existing, aligned_new = count_agg[col].align(part_agg[col])
                count_agg[col] = np.fmax(aligned_existing, aligned_new)

            # For timestamp min: use fillna pattern (NaT-safe)
            for col in ['min_timestamp']:
                existing = count_agg[col]
                new = part_agg[col]
                existing, new = existing.align(new)
                # Where existing is NaT, take new; where new is NaT, take existing; otherwise pick min
                combined = existing.where(existing <= new, new)
                combined = combined.fillna(existing)
                combined = combined.fillna(new)
                count_agg[col] = combined

            # For timestamp max: use fillna pattern (NaT-safe)
            for col in ['max_timestamp']:
                existing = count_agg[col]
                new = part_agg[col]
                existing, new = existing.align(new)
                combined = existing.where(existing >= new, new)
                combined = combined.fillna(existing)
                combined = combined.fillna(new)
                count_agg[col] = combined

            # Channel counts: add
            channel_counts = channel_counts.add(part_channels, fill_value=0)

            # Monthly counts: add
            monthly_counts = monthly_counts.add(part_monthly_count, fill_value=0)
            monthly_amounts = monthly_amounts.add(part_monthly_amount, fill_value=0)

        # Clean up
        del df, part_agg, part_channels, part_monthly_count, part_monthly_amount
        gc.collect()

        elapsed = time.time() - t0
        total_elapsed = time.time() - start_time
        print(f"  [{i+1}/{len(all_parts)}] {os.path.basename(part_file)}: "
              f"{rows:,} rows in {elapsed:.1f}s | "
              f"Total: {total_rows:,} rows, {total_elapsed:.0f}s elapsed")

    print(f"\n  Transaction processing complete. Total rows: {total_rows:,}")
    print(f"  Unique accounts with transactions: {len(count_agg):,}")

    # ---- DERIVE FINAL FEATURES from accumulated aggregates ----
    print("  Computing derived transaction features...")

    txn_features = count_agg.copy()

    # Mean and std
    txn_features['mean_amount'] = txn_features['total_amount'] / txn_features['txn_count']
    txn_features['mean_abs_amount'] = txn_features['total_abs_amount'] / txn_features['txn_count']
    variance = (txn_features['sum_sq_amount'] / txn_features['txn_count']) - (txn_features['mean_amount'] ** 2)
    txn_features['std_amount'] = np.sqrt(variance.clip(lower=0))

    # Credit/debit ratio
    txn_features['credit_debit_count_ratio'] = (
        txn_features['credit_count'] / txn_features['debit_count'].replace(0, np.nan)
    )
    txn_features['credit_debit_amount_ratio'] = (
        txn_features['credit_total'] / txn_features['debit_total'].replace(0, np.nan)
    )

    # Active span and transaction rate
    txn_features['active_span_days'] = (
        txn_features['max_timestamp'] - txn_features['min_timestamp']
    ).dt.total_seconds() / 86400.0
    txn_features['txn_per_day'] = (
        txn_features['txn_count'] / txn_features['active_span_days'].replace(0, np.nan)
    )

    # Pattern 2: Structuring percentage
    txn_features['structuring_pct'] = txn_features['structuring_count'] / txn_features['txn_count']

    # Pattern 4: Fan-in/Fan-out ratio
    txn_features['fan_ratio'] = (
        txn_features['unique_credit_cp'] / txn_features['unique_debit_cp'].replace(0, np.nan)
    )

    # Pattern 9: Round amount percentages
    txn_features['round_1k_pct'] = txn_features['round_1k_count'] / txn_features['txn_count']
    txn_features['round_5k_pct'] = txn_features['round_5k_count'] / txn_features['txn_count']
    txn_features['round_10k_pct'] = txn_features['round_10k_count'] / txn_features['txn_count']
    txn_features['round_50k_pct'] = txn_features['round_50k_count'] / txn_features['txn_count']

    # Pattern 11: Month boundary percentage
    txn_features['month_boundary_pct'] = txn_features['month_boundary_count'] / txn_features['txn_count']

    # Night and weekend percentages
    txn_features['night_txn_pct'] = txn_features['night_txn_count'] / txn_features['txn_count']
    txn_features['weekend_txn_pct'] = txn_features['weekend_txn_count'] / txn_features['txn_count']

    # Drop intermediate timestamp columns (not numeric features)
    txn_features['first_txn_timestamp'] = txn_features['min_timestamp']
    txn_features['last_txn_timestamp'] = txn_features['max_timestamp']
    txn_features.drop(columns=['min_timestamp', 'max_timestamp', 'sum_sq_amount'], inplace=True, errors='ignore')

    # ---- CHANNEL FEATURES ----
    channel_features = channel_counts.copy()
    # Dominant channel
    channel_features['dominant_channel_count'] = channel_features.max(axis=1)
    channel_features['channel_concentration'] = (
        channel_features['dominant_channel_count'] / channel_features.sum(axis=1).replace(0, np.nan)
    )
    # Number of active channels (non-zero)
    channel_features['active_channels'] = (channel_features[
        [f'ch_{c}' for c in ALL_CHANNELS]
    ] > 0).sum(axis=1)

    # ---- MONTHLY STATS (for temporal window prediction) ----
    monthly_stats = pd.DataFrame({
        'monthly_txn_std': monthly_counts.std(axis=1),
        'monthly_txn_max': monthly_counts.max(axis=1),
        'monthly_txn_min': monthly_counts.min(axis=1),
        'monthly_txn_mean': monthly_counts.mean(axis=1),
        'monthly_txn_cv': monthly_counts.std(axis=1) / monthly_counts.mean(axis=1).replace(0, np.nan),
        'active_months': (monthly_counts > 0).sum(axis=1),
        'monthly_amount_std': monthly_amounts.std(axis=1),
        'monthly_amount_max': monthly_amounts.max(axis=1),
        'monthly_amount_mean': monthly_amounts.mean(axis=1),
    })

    # Detect dormant activation: any month with count > mean + 3*std
    monthly_mean = monthly_counts.mean(axis=1)
    monthly_std_vals = monthly_counts.std(axis=1)
    burst_threshold = monthly_mean + 3 * monthly_std_vals.replace(0, 1)
    monthly_stats['burst_months'] = monthly_counts.gt(burst_threshold, axis=0).sum(axis=1)

    # Save monthly data for temporal window prediction later
    monthly_counts.to_parquet(os.path.join(OUTPUT_DIR, 'monthly_txn_counts.parquet'))
    monthly_amounts.to_parquet(os.path.join(OUTPUT_DIR, 'monthly_txn_amounts.parquet'))
    print(f"  Saved monthly stats for temporal window prediction")

    # ---- MERGE ALL TRANSACTION FEATURES ----
    txn_final = txn_features.join(channel_features, how='outer').join(monthly_stats, how='outer')

    # Prefix all transaction columns
    txn_final.columns = ['txn_' + c if not c.startswith(('ch_', 'txn_')) else c for c in txn_final.columns]

    print(f"  Transaction features shape: {txn_final.shape}")

    return txn_final


# ============================================================
# PHASE 3: TRANSACTION ADDITIONAL FEATURES
# ============================================================

def process_transactions_additional():
    """
    Process transaction_additional files for geo, IP, and balance features.

    MEMORY CHALLENGE: transactions_additional has NO account_id column.
    It only has transaction_id, which needs to be joined to main transactions.
    A full 396M-entry dict would need ~30GB RAM (won't fit in 16GB).

    SOLUTION: Process each additional part file individually. For each file,
    read just the transaction_ids, then scan main transaction part files
    (reading only transaction_id + account_id columns, ~16MB per part)
    to build a mapping for just those IDs. This is slower but memory-safe.

    OPTIMIZATION: Build the mapping incrementally - maintain a rolling
    cache of transaction_id → account_id from main txn parts.
    """
    print("\n" + "="*60)
    print("PHASE 3: Processing transaction_additional files")
    print("="*60)

    all_add_parts = sorted(glob(os.path.join(DATA_DIR, 'transactions_additional', 'batch-*', 'part_*.parquet')))
    main_parts = sorted(glob(os.path.join(DATA_DIR, 'transactions', 'batch-*', 'part_*.parquet')))
    print(f"  Found {len(all_add_parts)} transaction_additional part files")
    print(f"  Found {len(main_parts)} main transaction part files")

    if not all_add_parts:
        print("  WARNING: No transaction_additional files found!")
        return pd.DataFrame()

    # Check if account_id exists in transaction_additional
    sample_cols = pd.read_parquet(all_add_parts[0], columns=None).columns.tolist()
    print(f"  Available columns: {sample_cols}")
    has_account_id = 'account_id' in sample_cols

    if has_account_id:
        print("  account_id found directly! No mapping needed.")
    else:
        print("  No account_id in transaction_additional — will map via transaction_id")
        print("  Strategy: process in aligned batches (batch-1 ↔ batch-1, etc.)")

    # Determine which columns to read
    desired_cols = ['transaction_id', 'latitude', 'longitude', 'ip_address',
                    'balance_after_transaction', 'part_transaction_type',
                    'atm_deposit_channel_code', 'transaction_sub_type']
    read_cols = [c for c in desired_cols if c in sample_cols]
    if has_account_id:
        read_cols = ['account_id'] + read_cols
    print(f"  Reading columns: {read_cols}")

    # Pre-build batch-level mapping: for each batch, load all main txn parts
    # and build transaction_id → account_id mapping.
    # Each batch has ~100M rows × 2 cols × ~20 bytes = ~2-4 GB per batch mapping.
    # Instead of all at once, we'll do batch-by-batch.

    add_agg = None
    total_rows = 0
    start_time = time.time()

    # Group additional parts by batch
    add_batches = {}
    for p in all_add_parts:
        batch_name = os.path.basename(os.path.dirname(p))  # e.g., "batch-1"
        add_batches.setdefault(batch_name, []).append(p)

    # Group main parts by batch
    main_batches = {}
    for p in main_parts:
        batch_name = os.path.basename(os.path.dirname(p))
        main_batches.setdefault(batch_name, []).append(p)

    for batch_name in sorted(add_batches.keys()):
        batch_add_files = sorted(add_batches[batch_name])
        batch_main_files = sorted(main_batches.get(batch_name, []))

        print(f"\n  --- Processing {batch_name} ---")
        print(f"      Additional files: {len(batch_add_files)}, Main files: {len(batch_main_files)}")

        # Build transaction_id → account_id mapping for this batch
        batch_map = None
        if not has_account_id and batch_main_files:
            print(f"      Building transaction_id → account_id mapping for {batch_name}...")
            map_chunks = []
            for mp in batch_main_files:
                chunk = pd.read_parquet(mp, columns=['transaction_id', 'account_id'])
                map_chunks.append(chunk)
            batch_map = pd.concat(map_chunks, ignore_index=True)
            batch_map = batch_map.set_index('transaction_id')['account_id']
            del map_chunks
            gc.collect()
            print(f"      Mapping size: {len(batch_map):,} entries, "
                  f"~{batch_map.memory_usage(deep=True) / 1e6:.0f} MB")

        # Process each additional file in this batch
        for i, part_file in enumerate(batch_add_files):
            t0 = time.time()
            try:
                df = pd.read_parquet(part_file, columns=read_cols)
            except Exception as e:
                print(f"      ERROR reading {part_file}: {e}")
                continue

            rows = len(df)
            total_rows += rows

            # Map account_id if needed
            if not has_account_id:
                if batch_map is not None:
                    df['account_id'] = df['transaction_id'].map(batch_map)
                else:
                    df['account_id'] = np.nan
                df = df.dropna(subset=['account_id'])
                if len(df) == 0:
                    continue

            g = df.groupby('account_id')
            part_add = pd.DataFrame(index=g.groups.keys())

            # Geographic features (Pattern 5)
            if 'latitude' in df.columns and 'longitude' in df.columns:
                df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
                df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
                geo_df = df.dropna(subset=['latitude', 'longitude'])
                if len(geo_df) > 0:
                    geo = geo_df.groupby('account_id')
                    part_add['lat_min'] = geo['latitude'].min()
                    part_add['lat_max'] = geo['latitude'].max()
                    part_add['lon_min'] = geo['longitude'].min()
                    part_add['lon_max'] = geo['longitude'].max()
                    part_add['geo_txn_count'] = geo['latitude'].count()

            # IP address diversity
            if 'ip_address' in df.columns:
                part_add['unique_ip_part'] = g['ip_address'].nunique()

            # Balance after transaction
            if 'balance_after_transaction' in df.columns:
                df['balance_after_transaction'] = pd.to_numeric(df['balance_after_transaction'], errors='coerce')
                bal_df = df.dropna(subset=['balance_after_transaction'])
                if len(bal_df) > 0:
                    bal = bal_df.groupby('account_id')
                    part_add['bal_after_min'] = bal['balance_after_transaction'].min()
                    part_add['bal_after_max'] = bal['balance_after_transaction'].max()
                    part_add['bal_after_sum'] = bal['balance_after_transaction'].sum()
                    part_add['bal_after_count'] = bal['balance_after_transaction'].count()

            # Part transaction type distribution
            if 'part_transaction_type' in df.columns:
                ptt_counts = df.groupby(['account_id', 'part_transaction_type']).size().unstack(fill_value=0)
                for ptype in ['CI', 'BI', 'IP', 'IC']:
                    if ptype in ptt_counts.columns:
                        part_add[f'ptt_{ptype}_count'] = ptt_counts[ptype]
                    else:
                        part_add[f'ptt_{ptype}_count'] = 0

            # Transaction sub-type distribution
            if 'transaction_sub_type' in df.columns:
                sub_counts = df.groupby(['account_id', 'transaction_sub_type']).size().unstack(fill_value=0)
                for stype in ['CLT_CASH', 'LOAN', 'NORMAL', 'normal', 'clt_cash', 'loan']:
                    col_name = f'subtype_{stype.upper()}_count'
                    if stype in sub_counts.columns:
                        if col_name in part_add.columns:
                            part_add[col_name] = part_add[col_name].add(sub_counts[stype], fill_value=0)
                        else:
                            part_add[col_name] = sub_counts[stype]

            # ATM deposit count
            if 'atm_deposit_channel_code' in df.columns:
                atm_mask = df['atm_deposit_channel_code'].notna() & (df['atm_deposit_channel_code'] != '')
                if atm_mask.any():
                    part_add['atm_deposit_count'] = df[atm_mask].groupby('account_id').size()

            part_add = part_add.fillna(0)

            # ---- ACCUMULATE ----
            if add_agg is None:
                add_agg = part_add.copy()
            else:
                # Sum columns
                sum_cols = [c for c in part_add.columns if c not in
                            ['lat_min', 'lat_max', 'lon_min', 'lon_max',
                             'bal_after_min', 'bal_after_max']]
                for col in sum_cols:
                    if col in add_agg.columns:
                        add_agg[col] = add_agg[col].add(part_add.get(col, 0), fill_value=0)
                    else:
                        add_agg[col] = part_add.get(col, 0)

                # Min columns
                for col in ['lat_min', 'lon_min', 'bal_after_min']:
                    if col in part_add.columns:
                        if col in add_agg.columns:
                            add_agg[col] = add_agg[col].combine(part_add[col], min)
                        else:
                            add_agg[col] = part_add[col]

                # Max columns
                for col in ['lat_max', 'lon_max', 'bal_after_max']:
                    if col in part_add.columns:
                        if col in add_agg.columns:
                            add_agg[col] = add_agg[col].combine(part_add[col], max)
                        else:
                            add_agg[col] = part_add[col]

            del df, part_add
            gc.collect()

            elapsed = time.time() - t0
            total_elapsed = time.time() - start_time
            global_idx = sum(len(add_batches[b]) for b in sorted(add_batches.keys()) if b < batch_name) + i + 1
            print(f"      [{global_idx}/{len(all_add_parts)}] {os.path.basename(part_file)}: "
                  f"{rows:,} rows in {elapsed:.1f}s | "
                  f"Total: {total_rows:,} rows, {total_elapsed:.0f}s elapsed")

        # Free batch mapping
        del batch_map
        gc.collect()

    if add_agg is None or add_agg.empty:
        print("  No transaction_additional features generated.")
        return pd.DataFrame()

    # ---- Derived features ----
    print("  Computing derived transaction_additional features...")

    # Geographic spread
    if 'lat_min' in add_agg.columns:
        add_agg['geo_lat_spread'] = add_agg['lat_max'] - add_agg['lat_min']
        add_agg['geo_lon_spread'] = add_agg['lon_max'] - add_agg['lon_min']
        add_agg['geo_spread_km'] = np.sqrt(
            (add_agg['geo_lat_spread'] * 111) ** 2 +
            (add_agg['geo_lon_spread'] * 85) ** 2
        )

    # Balance volatility
    if 'bal_after_count' in add_agg.columns and 'bal_after_sum' in add_agg.columns:
        add_agg['bal_after_mean'] = add_agg['bal_after_sum'] / add_agg['bal_after_count'].replace(0, np.nan)
        add_agg['bal_after_range'] = add_agg['bal_after_max'] - add_agg['bal_after_min']

    # Customer-induced ratio
    ptt_cols = [c for c in ['ptt_CI_count', 'ptt_BI_count', 'ptt_IP_count', 'ptt_IC_count'] if c in add_agg.columns]
    if len(ptt_cols) >= 2:
        total_ptt = add_agg[ptt_cols].sum(axis=1)
        if 'ptt_CI_count' in add_agg.columns:
            add_agg['ci_ratio'] = add_agg['ptt_CI_count'] / total_ptt.replace(0, np.nan)

    # Prefix columns
    add_agg.columns = ['txn_add_' + c for c in add_agg.columns]

    print(f"  Transaction_additional features shape: {add_agg.shape}")
    return add_agg


# ============================================================
# PHASE 4: BRANCH-LEVEL COLLUSION FEATURES (Pattern 12)
# ============================================================

def compute_branch_features(static_df, train_labels):
    """Compute branch-level mule rate features (Pattern 12: Branch-Level Collusion)."""
    print("\n" + "="*60)
    print("PHASE 4: Computing branch-level collusion features")
    print("="*60)

    # Get branch_code per account
    acct_branch = static_df[['account_id', 'branch_code']].copy()

    # Merge with train labels to get mule rate per branch
    train_with_branch = acct_branch.merge(
        train_labels[['account_id', 'is_mule']], on='account_id', how='inner'
    )

    # Branch-level mule rate with Bayesian smoothing
    global_mule_rate = train_with_branch['is_mule'].mean()
    smoothing_factor = 10  # Bayesian prior strength

    branch_stats = train_with_branch.groupby('branch_code').agg(
        branch_mule_count=('is_mule', 'sum'),
        branch_total_count=('is_mule', 'count'),
        branch_mule_rate_raw=('is_mule', 'mean')
    )

    # Smoothed mule rate (avoids overfitting on small branches)
    branch_stats['branch_mule_rate_smoothed'] = (
        (branch_stats['branch_mule_count'] + smoothing_factor * global_mule_rate) /
        (branch_stats['branch_total_count'] + smoothing_factor)
    )

    print(f"  Global mule rate: {global_mule_rate:.4f}")
    print(f"  Branches with > 10% mule rate: {(branch_stats['branch_mule_rate_raw'] > 0.1).sum()}")

    # Merge back to all accounts
    branch_features = acct_branch.merge(branch_stats, on='branch_code', how='left')
    branch_features = branch_features.set_index('account_id')
    branch_features = branch_features[['branch_mule_count', 'branch_total_count',
                                        'branch_mule_rate_raw', 'branch_mule_rate_smoothed']]
    branch_features = branch_features.fillna(global_mule_rate)

    print(f"  Branch features shape: {branch_features.shape}")
    return branch_features


# ============================================================
# PHASE 5: MERGE ALL FEATURES
# ============================================================

def merge_all_features(static_df, txn_features, txn_add_features, branch_features, train_labels):
    """Merge all feature groups into a single feature matrix."""
    print("\n" + "="*60)
    print("PHASE 5: Merging all features")
    print("="*60)

    # --- Select only numeric columns from static features ---
    # Drop string/object/datetime columns that are raw identifiers
    drop_cols_exact = [
        'account_id', 'customer_id', 'account_status', 'product_code',
        'currency_code', 'product_family', 'branch_code', 'branch_pin',
        'account_opening_date', 'freeze_date', 'unfreeze_date',
        'last_mobile_update_date', 'last_kyc_date',
        'date_of_birth', 'relationship_start_date',
        'pan_available', 'aadhaar_available', 'passport_available',
        'mobile_banking_flag', 'internet_banking_flag', 'atm_card_flag',
        'demat_flag', 'credit_card_flag', 'fastag_flag',
        'nomination_flag', 'cheque_allowed', 'cheque_availed',
        'kyc_compliant', 'rural_branch',
        'name', 'gender', 'address', 'phone_number', 'branch_address',
        'branch_city', 'branch_state', 'branch_type', 'scheme_code',
        'address_last_update_date', 'passbook_last_update_date',
        'joint_account_flag', 'nri_flag',
        'customer_pin', 'permanent_pin', 'branch_pin_code'
    ]

    static_numeric = static_df.set_index('account_id')
    cols_to_drop = [c for c in drop_cols_exact if c in static_numeric.columns]
    static_numeric = static_numeric.drop(columns=cols_to_drop, errors='ignore')

    # Keep only numeric columns
    numeric_cols = static_numeric.select_dtypes(include=[np.number]).columns
    static_numeric = static_numeric[numeric_cols]

    print(f"  Static features: {static_numeric.shape}")

    # --- Merge with transaction features ---
    features = static_numeric.copy()

    if not txn_features.empty:
        # Drop timestamp columns from txn_features
        ts_cols = [c for c in txn_features.columns if 'timestamp' in c.lower()]
        txn_numeric = txn_features.drop(columns=ts_cols, errors='ignore')
        txn_numeric = txn_numeric.select_dtypes(include=[np.number])
        features = features.join(txn_numeric, how='left')
        print(f"  After adding txn features: {features.shape}")

    if not txn_add_features.empty:
        txn_add_numeric = txn_add_features.select_dtypes(include=[np.number])
        features = features.join(txn_add_numeric, how='left')
        print(f"  After adding txn_additional features: {features.shape}")

    if not branch_features.empty:
        features = features.join(branch_features, how='left')
        print(f"  After adding branch features: {features.shape}")

    # --- Pattern 1 (Dormant Activation): account age vs transaction span ---
    if 'account_age_days' in features.columns and 'txn_active_span_days' in features.columns:
        features['dormant_ratio'] = 1 - (features['txn_active_span_days'] / features['account_age_days'].replace(0, np.nan))
        features['dormant_ratio'] = features['dormant_ratio'].clip(lower=0)

    # --- Pattern 6 (New Account High Value): account age vs transaction volume ---
    if 'account_age_days' in features.columns and 'txn_txn_count' in features.columns:
        features['new_account_intensity'] = features['txn_txn_count'] / features['account_age_days'].replace(0, np.nan)

    # --- Pattern 7 (Income Mismatch): transaction amount vs balance ---
    if 'txn_max_abs_amount' in features.columns and 'avg_balance' in features.columns:
        features['income_mismatch_ratio'] = (
            features['txn_max_abs_amount'] / features['avg_balance'].replace(0, np.nan).abs()
        )
    if 'txn_total_abs_amount' in features.columns and 'avg_balance' in features.columns:
        features['volume_balance_ratio'] = (
            features['txn_total_abs_amount'] / features['avg_balance'].replace(0, np.nan).abs()
        )

    # --- Add train labels for later use ---
    features['is_mule'] = train_labels.set_index('account_id')['is_mule']

    # Replace infinities with NaN
    features = features.replace([np.inf, -np.inf], np.nan)

    print(f"\n  FINAL FEATURE MATRIX: {features.shape}")
    print(f"  Null percentage per column (top 20):")
    null_pct = features.isnull().mean().sort_values(ascending=False)
    print(null_pct.head(20).to_string())

    # Save
    output_path = os.path.join(OUTPUT_DIR, 'features.parquet')
    features.to_parquet(output_path)
    print(f"\n  Saved features to {output_path}")

    return features


# ============================================================
# MAIN
# ============================================================

def main():
    print("="*60)
    print("AML MULE ACCOUNT DETECTION — FEATURE ENGINEERING PIPELINE")
    print("="*60)
    overall_start = time.time()

    # Phase 1: Static features
    static_df, train_labels, test_accounts = load_static_features()

    # Phase 2: Transaction features
    txn_features = process_transactions()

    # Phase 3: Transaction additional features
    txn_add_features = process_transactions_additional()

    # Phase 4: Branch collusion features
    branch_features = compute_branch_features(static_df, train_labels)

    # Phase 5: Merge everything
    features = merge_all_features(static_df, txn_features, txn_add_features, branch_features, train_labels)

    total_time = time.time() - overall_start
    print(f"\nPIPELINE COMPLETE in {total_time/60:.1f} minutes")
    print(f"Output: {os.path.join(OUTPUT_DIR, 'features.parquet')}")
    print(f"Next step: python 03_train_model.py")


if __name__ == '__main__':
    main()
