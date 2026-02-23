"""
Lightning DataModule for Unified HNC (HANCOCK + TCGA-HNSC).

Handles:
- Loading and merging splits from both datasets
- Harmonized clinical+pathological preprocessing via unified harmonizer
- HANCOCK-specific blood preprocessing
- Creating UnifiedHNCDataset instances for each split
"""

import lightning as L
from torch.utils.data import DataLoader
from typing import Optional, Dict, List
import os
import pandas as pd

from src.data.dataset.unified_hnc import UnifiedHNCDataset, collate_fn
from src.data.preprocessors.unified_hnc.harmonizer import (
    prepare_unified_features,
    create_unified_transformer,
    apply_unified_transformer,
)
from src.data.preprocessors.blood import (
    prepare_blood_data_features,
    create_blood_transformer,
    apply_blood_transformer,
)


class UnifiedHNCDataModule(L.LightningDataModule):
    """
    Lightning DataModule for Unified HNC dataset.

    Merges HANCOCK and TCGA-HNSC into a single dataset with harmonized features.

    Preprocessing pipeline:
    1. Load and harmonize clinical+pathological features (unified schema)
    2. Merge splits from both datasets
    3. Fit transformers on unified train data
    4. Process HANCOCK blood data separately
    5. Create UnifiedHNCDataset instances
    """

    def __init__(
        self,
        # Tokenization parameters
        max_tokens_history: int = 200,
        max_tokens_surgery: int = 512,
        max_tokens_report: int = 60,
        path_lm: str = "./data/models/Bio_ClinicalBERT",
        # Training parameters
        batch_size: int = 32,
        num_workers: int = 4,
        seed: int = 42,
        # Split parameters
        k: int = 5,
        fold: int = 1,
        # Data root: parent directory containing HANCOCK/ and TCGA-HNSC/
        data_root: str = "./data",
        data_fraction: float = 1.0,
        **kwargs
    ):
        super().__init__()
        # Data roots
        self.data_root = data_root
        self.hancock_root = os.path.join(data_root, "HANCOCK")
        self.tcga_root = os.path.join(data_root, "TCGA-HNSC")
        self.data_fraction = data_fraction

        # Tokenization config
        self.max_tokens_history = max_tokens_history
        self.max_tokens_surgery = max_tokens_surgery
        self.max_tokens_report = max_tokens_report
        self.path_lm = path_lm

        # Training config
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        print(f'self.seed: {seed}')

        # Split config
        self.k = k
        self.fold = fold

        # Initialize datasets
        self.train_dataset: Optional[UnifiedHNCDataset] = None
        self.val_dataset: Optional[UnifiedHNCDataset] = None
        self.test_dataset: Optional[UnifiedHNCDataset] = None

        # Initialize patient ID lists
        self.train_patient_ids: Optional[List[str]] = None
        self.val_patient_ids: Optional[List[str]] = None
        self.test_patient_ids: Optional[List[str]] = None

        # Initialize transformers
        self.clinical_transformer = None
        self.blood_transformer = None
        self.blood_value_columns: Optional[List[str]] = None

        self._data_is_prepared: bool = False

    def prepare_data(self) -> None:
        """
        Merge splits from HANCOCK and TCGA-HNSC. Called only once per node.
        """
        if self._data_is_prepared:
            return

        # Load HANCOCK splits
        hancock_split_path = os.path.join(self.hancock_root, "Split", f"folds_{self.k}.csv")
        df_hancock = pd.read_csv(hancock_split_path, dtype=str)

        hancock_train = df_hancock[df_hancock[f"fold_{self.fold}"] == "train"].patient_id.tolist()
        hancock_val = df_hancock[df_hancock[f"fold_{self.fold}"] == "validation"].patient_id.tolist()
        hancock_test = df_hancock[df_hancock[f"fold_{self.fold}"] == "test"].patient_id.tolist()

        # Load TCGA splits
        tcga_split_path = os.path.join(self.tcga_root, "Split", f"folds_{self.k}.csv")
        df_tcga = pd.read_csv(tcga_split_path, dtype=str)

        tcga_train = df_tcga[df_tcga[f"fold_{self.fold}"] == "train"].patient_id.tolist()
        tcga_val = df_tcga[df_tcga[f"fold_{self.fold}"] == "validation"].patient_id.tolist()
        tcga_test = df_tcga[df_tcga[f"fold_{self.fold}"] == "test"].patient_id.tolist()

        # Merge
        self.train_patient_ids = hancock_train + tcga_train
        self.val_patient_ids = hancock_val + tcga_val
        self.test_patient_ids = hancock_test + tcga_test

        print(
            f"Unified HNC splits created - "
            f"Train: {len(self.train_patient_ids)} "
            f"(H:{len(hancock_train)}, T:{len(tcga_train)}), "
            f"Val: {len(self.val_patient_ids)} "
            f"(H:{len(hancock_val)}, T:{len(tcga_val)}), "
            f"Test: {len(self.test_patient_ids)} "
            f"(H:{len(hancock_test)}, T:{len(tcga_test)})"
        )

        self._data_is_prepared = True

    def setup(self, stage: Optional[str] = None) -> None:
        """Create train/val/test datasets."""
        print(f"Datamodule setup - stage: {stage}")

        if not self._data_is_prepared:
            self.prepare_data()

        if self.train_dataset is None:
            print("\n>>> Preparing Unified HNC TRAIN dataset...")

            # ======= CLINICAL + PATHOLOGICAL (HARMONIZED) =======
            print("  - Harmonizer: Loading and harmonizing features...")
            df_all, continuous_cols = prepare_unified_features(
                hancock_root=self.hancock_root,
                tcga_root=self.tcga_root,
            )

            # Filter to train patients
            df_train = df_all[df_all.patient_id.isin(self.train_patient_ids)].copy()

            # Create and fit transformer on train data
            print(f"  - Harmonizer: Creating and fitting transformer "
                  f"for {len(continuous_cols)} continuous columns...")
            self.clinical_transformer = create_unified_transformer()
            self.clinical_transformer.fit(df_train)

            # Apply
            print("  - Harmonizer: Transforming train data...")
            df_train = apply_unified_transformer(df_train, self.clinical_transformer)

            # ======= BLOOD (HANCOCK ONLY) =======
            # Get HANCOCK train patients for blood processing
            df_hancock_train = df_train[df_train.source == 'hancock']
            if len(df_hancock_train) > 0:
                # Need patient_id and sex for blood preprocessing
                blood_clinical = df_hancock_train[['patient_id', 'sex']].copy()

                print("  - Blood: Feature engineering (HANCOCK patients)...")
                blood_train, self.blood_value_columns = prepare_blood_data_features(
                    data_root=self.hancock_root,
                    data_clinical=blood_clinical,
                )
                print(f"  - Blood: Creating transformer for {len(self.blood_value_columns)} value columns...")
                self.blood_transformer = create_blood_transformer(self.blood_value_columns)
                print("  - Blood: Fitting transformer on train data...")
                self.blood_transformer.fit(blood_train)
                print("  - Blood: Transforming train data...")
                blood_train = apply_blood_transformer(
                    blood_train, self.blood_transformer, self.blood_value_columns
                )
            else:
                blood_train = None

            # Create Dataset
            self.train_dataset = UnifiedHNCDataset(
                split="train",
                data_clinical=df_train,
                data_blood=blood_train,
                hancock_root=self.hancock_root,
                tcga_root=self.tcga_root,
                list_patient_id_sample=self.train_patient_ids,
                max_tokens_history=self.max_tokens_history,
                max_tokens_surgery=self.max_tokens_surgery,
                max_tokens_report=self.max_tokens_report,
                path_lm=self.path_lm,
            )
            print(self.train_dataset)

        if self.val_dataset is None:
            print("\n>>> Preparing Unified HNC VAL dataset...")

            # ======= CLINICAL + PATHOLOGICAL =======
            df_all, _ = prepare_unified_features(self.hancock_root, self.tcga_root)
            df_val = df_all[df_all.patient_id.isin(self.val_patient_ids)].copy()
            print("  - Harmonizer: Transforming val data with train transformer...")
            df_val = apply_unified_transformer(df_val, self.clinical_transformer)

            # ======= BLOOD =======
            df_hancock_val = df_val[df_val.source == 'hancock']
            blood_val = None
            if len(df_hancock_val) > 0 and self.blood_transformer is not None:
                blood_clinical = df_hancock_val[['patient_id', 'sex']].copy()
                print("  - Blood: Feature engineering (HANCOCK val patients)...")
                blood_val, _ = prepare_blood_data_features(
                    data_root=self.hancock_root,
                    data_clinical=blood_clinical,
                )
                print("  - Blood: Transforming val data with train transformer...")
                blood_val = apply_blood_transformer(
                    blood_val, self.blood_transformer, self.blood_value_columns
                )

            self.val_dataset = UnifiedHNCDataset(
                split="val",
                data_clinical=df_val,
                data_blood=blood_val,
                hancock_root=self.hancock_root,
                tcga_root=self.tcga_root,
                list_patient_id_sample=self.val_patient_ids,
                max_tokens_history=self.max_tokens_history,
                max_tokens_surgery=self.max_tokens_surgery,
                max_tokens_report=self.max_tokens_report,
                path_lm=self.path_lm,
            )
            print(self.val_dataset)

        if self.test_dataset is None:
            print("\n>>> Preparing Unified HNC TEST dataset...")

            # ======= CLINICAL + PATHOLOGICAL =======
            df_all, _ = prepare_unified_features(self.hancock_root, self.tcga_root)
            df_test = df_all[df_all.patient_id.isin(self.test_patient_ids)].copy()
            print("  - Harmonizer: Transforming test data with train transformer...")
            df_test = apply_unified_transformer(df_test, self.clinical_transformer)

            # ======= BLOOD =======
            df_hancock_test = df_test[df_test.source == 'hancock']
            blood_test = None
            if len(df_hancock_test) > 0 and self.blood_transformer is not None:
                blood_clinical = df_hancock_test[['patient_id', 'sex']].copy()
                print("  - Blood: Feature engineering (HANCOCK test patients)...")
                blood_test, _ = prepare_blood_data_features(
                    data_root=self.hancock_root,
                    data_clinical=blood_clinical,
                )
                print("  - Blood: Transforming test data with train transformer...")
                blood_test = apply_blood_transformer(
                    blood_test, self.blood_transformer, self.blood_value_columns
                )

            self.test_dataset = UnifiedHNCDataset(
                split="test",
                data_clinical=df_test,
                data_blood=blood_test,
                hancock_root=self.hancock_root,
                tcga_root=self.tcga_root,
                list_patient_id_sample=self.test_patient_ids,
                max_tokens_history=self.max_tokens_history,
                max_tokens_surgery=self.max_tokens_surgery,
                max_tokens_report=self.max_tokens_report,
                path_lm=self.path_lm,
            )
            print(self.test_dataset)

        print(f"\nDatasets created - Train: {len(self.train_dataset)}, "
              f"Val: {len(self.val_dataset)}, Test: {len(self.test_dataset)}\n")

    # =========================================================================
    # DataLoaders
    # =========================================================================

    def train_dataloader(self, batch_size: int = None, num_workers: int = None) -> DataLoader:
        batch_size = batch_size if batch_size is not None else self.batch_size
        num_workers = num_workers if num_workers is not None else self.num_workers
        return DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False,
        )

    def val_dataloader(self, batch_size: int = None, num_workers: int = None) -> DataLoader:
        batch_size = batch_size if batch_size is not None else self.batch_size
        num_workers = num_workers if num_workers is not None else self.num_workers
        return DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False,
        )

    def test_dataloader(self, batch_size: int = None, num_workers: int = None) -> DataLoader:
        batch_size = batch_size if batch_size is not None else self.batch_size
        num_workers = num_workers if num_workers is not None else self.num_workers
        return DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False,
        )

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()

    # =========================================================================
    # Model utilities
    # =========================================================================

    def get_input_dims(self) -> Dict[str, int]:
        """Get input dimensions for each modality."""
        available = self.train_dataset or self.val_dataset or self.test_dataset
        if available is None:
            raise ValueError("No datasets have been created yet")
        return available.get_input_dims()

    def get_tmax(self) -> int:
        """Get maximum time from the train set."""
        available = self.train_dataset or self.val_dataset or self.test_dataset
        if available is None:
            raise ValueError("No datasets have been created yet")
        return available.get_tmax()

