import os
import json
import numpy as np
import pandas as pd

def create_cv_splits(patient_ids, n_folds=5, train_ratio=0.7, val_ratio=0.15, random_seed=42):
    """
    Create n-fold cross-validation splits with specified train/val/test ratios.

    Args:
        patient_ids (list): List of patient IDs.
        n_folds (int): Number of folds.
        train_ratio (float): Fraction for training set.
        val_ratio (float): Fraction for validation set.
        test_ratio (float): Fraction for test set.
        random_seed (int): Seed for reproducibility.

    Returns:
        pd.DataFrame: DataFrame with patient_id and fold columns.
    """
    np.random.seed(random_seed)
    patient_ids = np.array(patient_ids)
    n_patients = len(patient_ids)
    
    # Shuffle patient IDs
    shuffled_ids = np.random.permutation(patient_ids)
    
    # Calculate split sizes
    train_size = int(np.floor(train_ratio * n_patients))
    val_size = int(np.floor(val_ratio * n_patients))
    test_size = n_patients - train_size - val_size  # Ensure all patients are included
    
    # Prepare DataFrame
    df = pd.DataFrame({"patient_id": patient_ids})
    
    for fold in range(1, n_folds + 1):
        # Rotate shuffled IDs for each fold to get different splits
        rotated_ids = np.roll(shuffled_ids, shift=fold * test_size)
        
        # Assign splits as full strings
        split = ['train'] * train_size + ['validation'] * val_size + ['test'] * test_size
        
        # Map split back to patient_id order
        fold_assignment = pd.Series(split, index=rotated_ids)
        df[f'fold_{fold}'] = df['patient_id'].map(fold_assignment)
    
    return df

def save_splits(data_root, n_folds, df):
    # Folder
    split_path = os.path.join(data_root, "Split")
    os.makedirs(split_path, exist_ok=True)

    # CSV
    csv_path = os.path.join(split_path, f"folds_{n_folds}.csv")

    # Save
    df.to_csv(csv_path, index=0)
    print(f'File successfully saved at: {csv_path}')

def get_dataset_config(dataset: str):
    """
    Return dataset-specific configuration for clinical data path and patient ID column.
    
    Args:
        dataset (str): Dataset name ('hancock' or 'tcga').
        
    Returns:
        dict: Configuration dictionary with clinical_subpath and patient_id_column.
    """
    configs = {
        "hancock": {
            "clinical_subpath": os.path.join("StructuredData", "clinical_data.json"),
            "patient_id_column": "patient_id",
        },
        "tcga": {
            "clinical_subpath": "clinical_data.json",
            "patient_id_column": "submitter_id",
        },
    }
    
    if dataset not in configs:
        raise ValueError(f"Unknown dataset: {dataset}. Supported: {list(configs.keys())}")
    
    return configs[dataset]


def _get_primary_diagnosis_tcga(patient: dict) -> dict:
    """
    Return the PRIMARY diagnosis for a TCGA patient.
    
    TCGA patients can have multiple diagnoses (primary, recurrence, metastasis).
    We identify the primary using `diagnosis_is_primary_disease == 'true'`.
    """
    for d in patient.get('diagnoses', []):
        if d.get('diagnosis_is_primary_disease') == 'true':
            return d
    diagnoses = patient.get('diagnoses', [])
    return diagnoses[0] if diagnoses else {}


def extract_survival_targets(clinical_data: list, dataset: str) -> pd.DataFrame:
    """
    Extract survival targets (patient_id, event, time) from clinical data.
    
    Args:
        clinical_data: List of patient dictionaries (raw JSON).
        dataset: Dataset name ('hancock' or 'tcga').
        
    Returns:
        DataFrame with patient_id, event, time columns.
    """
    records = []
    
    if dataset == "hancock":
        for patient in clinical_data:
            patient_id = patient.get("patient_id")
            survival_status = patient.get("survival_status")
            time = patient.get("days_to_last_information")
            event = 1 if survival_status == "deceased" else 0
            records.append({"patient_id": patient_id, "event": event, "time": time})
    
    elif dataset == "tcga":
        for patient in clinical_data:
            patient_id = patient.get("submitter_id")
            demo = patient.get("demographic", {})
            vital_status = demo.get("vital_status")
            primary = _get_primary_diagnosis_tcga(patient)
            
            if vital_status == "Dead":
                event = 1
                time = demo.get("days_to_death")
            elif vital_status == "Alive":
                event = 0
                time = primary.get("days_to_last_follow_up")
            else:
                event = None
                time = None
            
            records.append({"patient_id": patient_id, "event": event, "time": time})
    
    return pd.DataFrame(records)


def print_fold_distribution(df_splits: pd.DataFrame, df_targets: pd.DataFrame, n_folds: int):
    """
    Print target distribution (event rate) for each fold and split.
    
    Args:
        df_splits: DataFrame with patient_id and fold columns.
        df_targets: DataFrame with patient_id, event, time columns.
        n_folds: Number of folds.
    """
    # Merge splits with targets
    df = df_splits.merge(df_targets, on="patient_id")
    
    print("\n" + "=" * 70)
    print("TARGET DISTRIBUTION BY FOLD")
    print("=" * 70)
    
    # Header
    header = f"{'Fold':<8}"
    for split in ['train', 'validation', 'test']:
        header += f"| {split:^18} "
    print(header)
    print("-" * 70)
    
    for fold in range(1, n_folds + 1):
        fold_col = f"fold_{fold}"
        row = f"Fold {fold:<3}"
        
        for split in ['train', 'validation', 'test']:
            mask = df[fold_col] == split
            n_total = mask.sum()
            n_events = df.loc[mask, 'event'].sum()
            event_rate = n_events / n_total * 100 if n_total > 0 else 0
            row += f"| {n_events:>3}/{n_total:<4} ({event_rate:>5.1f}%) "
        
        print(row)
    
    print("=" * 70)
    
    # Overall stats
    total_patients = len(df)
    total_events = df['event'].sum()
    overall_rate = total_events / total_patients * 100
    print(f"Overall: {total_events}/{total_patients} events ({overall_rate:.1f}%)")
    print(f"Time range: {df['time'].min():.0f} - {df['time'].max():.0f} days")


def build_folds(args):

    # Parameters
    data_root   = args.data_root
    dataset     = args.dataset.lower()
    n_folds     = args.n_folds
    train_ratio = args.train_ratio
    val_ratio   = args.val_ratio
    random_seed = args.random_seed

    # Get dataset-specific config
    dataset_config = get_dataset_config(dataset)
    clinical_subpath = dataset_config["clinical_subpath"]
    patient_id_column = dataset_config["patient_id_column"]

    # Source Clinical Data
    clinical_path = os.path.join(data_root, clinical_subpath)
    print(f"Loading clinical data from: {clinical_path}")
    with open(clinical_path, 'r') as file:
        clinical_data = json.load(file)

    print(f"Total patients in clinical data: {len(clinical_data)}")

    # Extract survival targets
    df_targets = extract_survival_targets(clinical_data, dataset)
    
    # Filter patients with valid targets (non-null time and event)
    valid_mask = df_targets['time'].notna() & df_targets['event'].notna()
    excluded_patients = df_targets[~valid_mask]['patient_id'].tolist()
    df_targets_valid = df_targets[valid_mask].copy()
    
    if excluded_patients:
        print(f"\n⚠️  Excluding {len(excluded_patients)} patients with missing survival target:")
        for pid in excluded_patients:
            print(f"   - {pid}")
    
    # List of valid patients
    valid_patient_ids = list(df_targets_valid['patient_id'])
    print(f"\nPatients with valid survival target: {len(valid_patient_ids)}")

    # Create splits
    df_splits = create_cv_splits(
        patient_ids = valid_patient_ids,
        n_folds     = n_folds,
        train_ratio = train_ratio,
        val_ratio   = val_ratio,
        random_seed = random_seed
    )
    
    # Print distribution by fold
    print_fold_distribution(df_splits, df_targets_valid, n_folds)
    
    # Save dataframe including splits for each fold
    save_splits(data_root, n_folds, df_splits)