# AML Mule Account Detection Challenge

> **Data Format**: All files are in **Apache Parquet** format (Snappy compression). Use `pandas.read_parquet()`, `pyarrow.parquet`, or any Parquet-compatible reader. Transaction files are split across `batch-N/` subdirectories.

## Primary Objective/Problem Statement

Identify mule accounts used for money laundering from banking transaction and account data. Given labelled training data and unlabelled test accounts, predict which test accounts are mules.


## Evaluation

- Exact Submission Guidelines will be provided shortly
- Evaluation Criteria is listed below:
    - 40% weightage for Model/Feature Ingenuity
        - Rewarded for creative and innovative ideas in types of models/features used, multiple models/algorithms in series/parallel may be used to generate final predictions
    - 20% weightage for model performance scores 
        - Rewarded for high accuracy/precision scores, calculated scores such as AUC-ROC and F1 scores
    - 15% weightage for avoidance of red-herrings in data
        - Rewarded for succesfully avoiding several red-herrings injected in the training data
    - 15% weightage for additional insights such as Temporal IOU scores, other insights
    - 10% weightage for report quality
        - Rewarded for clear and concise insights presented in the report with data-driven logic.

## Data Files

### Provided to You

| File | Rows | Size | Description |
|---|---|---|---|
| `customers.parquet` | 159,000 | 2.3 MB | Customer demographics and KYC information |
| `accounts.parquet` | 160,000 | 6.7 MB | Account-level attributes |
| `transactions/` | ~400,000,000 | 8.2 GB | Individual transaction records (5-year window: Jul 2020 - Jun 2025), 396 parts in 4 batches |
| `demographics.parquet` | ~159,000 | 4.6 MB | Customer demographics (name, gender, address, phone) |
| `accounts-additional.parquet` | ~160,000 | 0.9 MB | Scheme codes per account |
| `transactions_additional/` | ~400,000,000 | 8.4 GB | Extended transaction fields, 311 parts in 4 batches |
| `branch.parquet` | ~9,000 | 0.3 MB | Branch-level metadata |
| `customer_account_linkage.parquet` | 160,000 | 1.8 MB | Maps customers to their accounts |
| `product_details.parquet` | 159,000 | 2.9 MB | Aggregated product holdings per customer |
| `train_labels.parquet` | 96,000 | 0.6 MB | Training labels — `is_mule`: 1 = mule, 0 = legitimate |
| `test_accounts.parquet` | 64,000 | 0.4 MB | Account IDs you need to predict on |

**Total: 720 files, 16.2 GB**

**Note:** `transactions/` and `transactions_additional/` are split into batch subdirectories (`batch-1/`, `batch-2/`, etc.) with up to 100 part files each. To read all parts:

```python
import pandas as pd
from glob import glob

# Read all transaction parts
parts = sorted(glob("transactions/batch-*/part_*.parquet"))
df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
```

### Schema

**customers.parquet**

| Column | Description |
|---|---|
| `customer_id` | Unique customer identifier |
| `date_of_birth` | Date of birth (YYYY-MM-DD) |
| `relationship_start_date` | Date customer relationship began |
| `pan_available` | PAN card on file (Y/N) |
| `aadhaar_available` | Aadhaar on file (Y/N) |
| `passport_available` | Passport on file (Y/N) |
| `mobile_banking_flag` | Mobile banking registered (Y/N) |
| `internet_banking_flag` | Internet banking registered (Y/N) |
| `atm_card_flag` | ATM/debit card issued (Y/N) |
| `demat_flag` | Demat account linked (Y/N) |
| `credit_card_flag` | Credit card held (Y/N) |
| `fastag_flag` | FASTag linked (Y/N) |
| `customer_pin` | Residential PIN code |
| `permanent_pin` | Permanent address PIN code |

**accounts.parquet**

| Column | Description |
|---|---|
| `account_id` | Unique account identifier |
| `account_status` | `active` or `frozen` |
| `product_code` | Product code |
| `currency_code` | Currency (1 = INR) |
| `account_opening_date` | Date account was opened |
| `branch_code` | Branch identifier |
| `branch_pin` | Branch location PIN code |
| `avg_balance` | Average balance (can be negative for overdraft) |
| `product_family` | `S` (Savings), `K` (K-family), `O` (Overdraft) |
| `nomination_flag` | Nominee registered (Y/N) |
| `cheque_allowed` | Cheque facility available (Y/N) |
| `cheque_availed` | Cheque book opted (Y/N) |
| `num_chequebooks` | Number of cheque books issued |
| `last_mobile_update_date` | Date of last mobile number change |
| `kyc_compliant` | KYC compliant (Y/N) |
| `last_kyc_date` | Date of last KYC verification |
| `rural_branch` | Rural branch (Y/N) |
| `monthly_avg_balance` | Monthly average balance |
| `quarterly_avg_balance` | Quarterly average balance |
| `daily_avg_balance` | Daily average balance |
| `freeze_date` | Date the account was frozen (null if never frozen) |
| `unfreeze_date` | Date the account was unfrozen (null if never unfrozen) |

Note: `customer_id` is not in this file. Use `customer_account_linkage.parquet` to join accounts to customers.

**transactions (partitioned)**

| Column | Description |
|---|---|
| `transaction_id` | Unique transaction identifier |
| `account_id` | Account the transaction belongs to |
| `transaction_timestamp` | ISO format timestamp |
| `mcc_code` | Merchant Category Code |
| `channel` | Transaction channel (see below) |
| `amount` | Amount in INR (negative values indicate reversals) |
| `txn_type` | `D` (Debit) or `C` (Credit) |
| `counterparty_id` | Counterparty identifier |

Transaction channels: `UPC` (UPI Credit), `UPD` (UPI Debit), `END` (E-commerce/POS), `IPM` (IMPS), `STD` (Standing instruction debit), `P2A` (Pay-to-account), `FTD` (Fund transfer debit), `NTD` (NEFT debit), `MCR` (Mobile credit), `FTC` (Fund transfer credit), `MAC` (Mobile app), `TPD` (Third-party debit), `APD` (Auto-pay debit), `CHQ` (Cheque), `ATW` (ATM withdrawal), `TPC` (Third-party credit), `STC` (Standing instruction credit), `OCD` (Over-counter deposit), `RCD` (Recurring deposit credit), `IFD` (Internal fund debit), `ETD` (Electronic transfer debit), `NWD` (Network debit), `CSD` (Cash deposit), `IFC` (Internal fund credit), `PCA` (Payment card authorization), `MAD` (Mandate debit), `CHD` (Clearing house debit), `RTD` (Return debit), `CCL` (Credit card linked), `OPI` (Online payment initiation), `CTC` (Clearing transfer credit), `SID` (System-initiated debit), `ASD` (Auto-sweep debit), `IAD` (Inter-account debit), `SCW` (Smart card withdrawal).

**customer_account_linkage.parquet**

| Column | Description |
|---|---|
| `customer_id` | Customer identifier |
| `account_id` | Account identifier |

A single customer may hold multiple accounts.

**product_details.parquet**

| Column | Description |
|---|---|
| `customer_id` | Customer identifier |
| `loan_sum` | Total outstanding loan amount (can be negative) |
| `loan_count` | Number of active loans |
| `cc_sum` | Total credit card outstanding (can be negative) |
| `cc_count` | Number of credit cards |
| `od_sum` | Total overdraft facility amount (can be negative) |
| `od_count` | Number of overdraft accounts |
| `ka_sum` | Total balance across K-family accounts |
| `ka_count` | Number of K-family accounts |
| `sa_sum` | Total balance across savings accounts |
| `sa_count` | Number of savings accounts |

**train_labels.parquet**

| Column | Description |
|---|---|
| `account_id` | Account identifier (training set only) |
| `is_mule` | `1` = mule account, `0` = legitimate account |
| `mule_flag_date` | Date the account was flagged as a mule (empty for legitimate accounts) |
| `alert_reason` | Reason the account was flagged (empty for legitimate accounts) |
| `flagged_by_branch` | Branch code that reported the activity (empty for legitimate accounts) |

Note: Labels may contain noise/red-herrings. Not all labels are guaranteed to be correct.

**test_accounts.parquet**

| Column | Description |
|---|---|
| `account_id` | Account IDs to generate predictions for |

**demographics.parquet**

| Column | Description |
|---|---|
| `customer_id` | Customer identifier |
| `name` | Full customer name |
| `gender` | M or F |
| `address_last_update_date` | Date of last address update |
| `address` | Street address and city |
| `phone_number` | Phone number |
| `passbook_last_update_date` | Date of last passbook update |
| `joint_account_flag` | Joint account (Y/N) |
| `nri_flag` | Non-Resident Indian (Y/N) |

**accounts-additional.parquet**

| Column | Description |
|---|---|
| `account_id` | Account identifier |
| `scheme_code` | Government scheme code (PMJDY, PMSBY, PMJJBY, APY, SCSS, SSA, REGULAR) |

**transactions_additional (partitioned)**

| Column | Description |
|---|---|
| `transaction_id` | Transaction identifier (joins to transactions) |
| `mnemonic_code` | Transaction type mnemonic (same as channel) |
| `latitude` | Transaction location latitude |
| `longitude` | Transaction location longitude |
| `ip_address` | Source IP address |
| `balance_after_transaction` | Running account balance after this transaction |
| `part_transaction_type` | CI (Customer Induced), BI (Bank Induced), IP (Interest Paid), IC (Interest Collected) |
| `atm_deposit_channel_code` | CDM or CRM for ATM deposits, empty otherwise |
| `transaction_sub_type` | CLT_CASH, LOAN, or NORMAL |

**branch.parquet**

| Column | Description |
|---|---|
| `branch_code` | Branch identifier |
| `branch_address` | Branch street address |
| `branch_pin_code` | Branch location PIN code |
| `branch_city` | City |
| `branch_state` | State code |
| `branch_employee_count` | Number of employees at the branch |
| `branch_turnover` | Branch turnover metric |
| `branch_asset_size` | Branch asset size |
| `branch_type` | urban, semi-urban, or rural |

## Known Mule Behavior Patterns

The following money laundering patterns are known to exist in real-world banking data. Mule accounts in this dataset may exhibit one or more of these behaviors:

1. **Dormant Activation** — Long-inactive accounts suddenly showing high-value transaction bursts
2. **Structuring** — Repeated transactions just below reporting thresholds (e.g., amounts near 50,000)
3. **Rapid Pass-Through** — Large credits quickly followed by matching debits (funds barely rest in the account)
4. **Fan-In / Fan-Out** — Many small inflows aggregated into one large outflow, or vice versa
5. **Geographic Anomaly** — Transactions from locations inconsistent with the account holder's profile
6. **New Account High Value** — Recently opened accounts with unusually high transaction volumes
7. **Income Mismatch** — Transaction values disproportionate to account balance or customer profile
8. **Post-Mobile-Change Spike** — Sudden transaction surge after a mobile number update (potential account takeover)
9. **Round Amount Patterns** — Disproportionate use of exact round amounts (1K, 5K, 10K, 50K)
10. **Layered/Subtle** — Weak signals from multiple patterns combined, no single strong indicator
11. **Salary Cycle Exploitation** — Laundering disguised within natural salary credit and bill payment cycles at month boundaries
12. **Branch-Level Collusion** — Clusters of suspicious accounts at the same branch with shared counterparties and coordinated timing
13. **MCC-Amount Anomaly** — Transactions with amounts that are statistical outliers for their merchant category code

## Relationships

```
customers ──(customer_id)──> customer_account_linkage ──(account_id)──> accounts
    |                                                                       |
(customer_id)                                                          (account_id)
    |                                                                       |
    v                                                                       v
demographics                                                          transactions
                                                                            |
                                                                     (transaction_id)
                                                                            |
                                                                            v
                                                                 transactions_additional

customers ──(customer_id)──> product_details
accounts  ──(account_id)──> train_labels / test_accounts
accounts  ──(account_id)──> accounts-additional
accounts  ──(branch_code)──> branch
```

## Submission Format

```csv
account_id,is_mule,suspicious_start,suspicious_end
ACCT_000000,0.02,,
ACCT_000003,0.87,2023-11-15T09:30:00,2024-02-20T16:45:00
...
```

One row per account in `test_accounts.parquet`:
- **`is_mule`**: Probability score between 0 and 1
- **`suspicious_start`**: ISO timestamp of the beginning of the suspected suspicious activity window (empty if predicted legitimate)
- **`suspicious_end`**: ISO timestamp of the end of the suspected suspicious activity window (empty if predicted legitimate)

The time window should capture when you believe the mule activity occurred. Primary scoring is on `is_mule`. Time window accuracy is scored separately as a bonus metric using temporal IoU (Intersection over Union) against the ground truth activity period.

Good luck.
