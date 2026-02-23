"""
Unified HNC Dataset for multimodal survival prediction.

Merges HANCOCK and TCGA-HNSC into a single dataset.
"""

import os
import re
import h5py
import torch
import numpy as np
import pandas as pd

from pathlib import Path
from functools import reduce
from typing import Tuple, Dict, Optional, List
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from src.data.preprocessors.unified_hnc.feature_columns import (
    get_unified_clinical_cols,
    get_unified_pathological_cols,
)


def collate_fn(data):
    """Collate function to batch samples together."""
    patient_id, tensor_tuples, target = zip(*data)
    clinical, blood, patho, h_ids, h_mask, s_ids, s_mask, r_ids, r_mask, lymph, tumor = zip(*tensor_tuples)
    time_to_event, survival_status = zip(*target)

    clinical = torch.stack(clinical, 0)
    blood = torch.stack(blood, 0)
    patho = torch.stack(patho, 0)
    lymph = torch.stack(lymph, 0)
    tumor = torch.stack(tumor, 0)

    h_ids = torch.vstack(h_ids)
    h_mask = torch.vstack(h_mask)
    s_ids = torch.vstack(s_ids)
    s_mask = torch.vstack(s_mask)
    r_ids = torch.vstack(r_ids)
    r_mask = torch.vstack(r_mask)

    time_to_event = torch.stack(time_to_event, 0)
    survival_status = torch.stack(survival_status, 0)

    input_tuple = (clinical, blood, patho, h_ids, h_mask, s_ids, s_mask, r_ids, r_mask, lymph, tumor)
    target = (time_to_event, survival_status)
    patient_id = list(patient_id)

    return patient_id, input_tuple, target


class UnifiedHNCDataset(Dataset):
    """
    Unified HNC Dataset for multimodal survival prediction.

    Merges HANCOCK and TCGA-HNSC with harmonized features.

    Available modalities depend on patient source:
    - All patients: clinical (step1+step2), pathological (step3), tumor WSI
    - HANCOCK only: blood, text (histories/surgery/reports), lymph node WSI

    Missing modalities are zero-padded for baseline compatibility.
    """

    # Unified tumor WSI dimension (HANCOCK 1024 → pad to 1536, TCGA native 1536)
    TUMOR_DIM = 1536

    def __init__(
        self,
        split: str,
        data_clinical: pd.DataFrame,
        data_blood: Optional[pd.DataFrame],
        hancock_root: str,
        tcga_root: str,
        list_patient_id_sample: Optional[List[str]] = None,
        max_tokens_history: int = 200,
        max_tokens_surgery: int = 512,
        max_tokens_report: int = 60,
        path_lm: str = "./data/models/Bio_ClinicalBERT",
        **kwargs,
    ):
        """
        Initialize Unified HNC Dataset.

        Args:
            split: Dataset split name (train/val/test)
            data_clinical: Harmonized clinical+pathological DataFrame (all features)
            data_blood: Blood DataFrame (HANCOCK patients only), or None
            hancock_root: Path to HANCOCK data directory
            tcga_root: Path to TCGA-HNSC data directory
            list_patient_id_sample: Optional list of patient IDs to include
            max_tokens_history: Maximum tokens for history text
            max_tokens_surgery: Maximum tokens for surgery description
            max_tokens_report: Maximum tokens for report text
            path_lm: Path to language model (Bio_ClinicalBERT)
        """
        self.device = torch.device('cpu')

        self.split = split
        self.hancock_root = hancock_root
        self.tcga_root = tcga_root
        self.path_lm = path_lm
        self.list_patient_id_sample = list_patient_id_sample

        self.max_tokens_history = max_tokens_history
        self.max_tokens_surgery = max_tokens_surgery
        self.max_tokens_report = max_tokens_report

        # Store preprocessed data
        self.data_clinical = data_clinical
        self.data_blood = data_blood

        # Build source lookup: patient_id -> 'hancock' or 'tcga'
        self._source_map = dict(zip(
            data_clinical['patient_id'].astype(str),
            data_clinical['source'].astype(str),
        ))

        # Blood dimension (from HANCOCK blood DataFrame)
        if data_blood is not None and len(data_blood) > 0:
            self._blood_dim = len(data_blood.columns) - 1  # exclude patient_id
        else:
            self._blood_dim = 0

        # Prepare samples
        self.prepare_samples()

        # Build tokenizer (needed for HANCOCK text data)
        self.build_tokenizer()

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        n_hancock = sum(1 for pid in self.samples if self._get_source(pid) == 'hancock')
        n_tcga = len(self.samples) - n_hancock
        return (
            f"\n--- UnifiedHNCDataset ---\n"
            f"  - Split: {self.split}\n"
            f"  - Total samples: {len(self.samples)} "
            f"(HANCOCK: {n_hancock}, TCGA: {n_tcga})\n"
            f"  - Blood dim: {self._blood_dim}\n"
            f"  - Input dims: {self.get_input_dims()}\n\n"
        )

    def __len__(self) -> int:
        return len(self.samples)

    # =========================================================================
    # Setup
    # =========================================================================

    def build_tokenizer(self) -> None:
        """Build tokenizer from pretrained language model."""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.path_lm, local_files_only=True, do_lower_case=True
        )

    def prepare_samples(self) -> None:
        """Prepare sample list and file paths for text/images."""
        self._prepare_hancock_text()
        self._prepare_hancock_lymph()
        self._prepare_hancock_tumor()
        self._prepare_tcga_tumor()

        # Build sample list
        self.samples = list(self.data_clinical.patient_id)
        if self.list_patient_id_sample is not None:
            valid = set(self.samples)
            self.samples = [pid for pid in self.list_patient_id_sample if pid in valid]

    def _get_source(self, patient_id: str) -> str:
        """Get the source dataset for a patient."""
        return self._source_map.get(str(patient_id), 'unknown')

    # =========================================================================
    # File path preparation
    # =========================================================================

    def _prepare_hancock_text(self) -> None:
        """Load HANCOCK text files (histories, surgery descriptions, reports)."""
        subfolder = os.path.join(self.hancock_root, "TextData")
        path_histories = os.path.join(subfolder, "histories_english")
        path_surgeries = os.path.join(subfolder, "surgery_descriptions_english")
        path_reports = os.path.join(subfolder, "reports_english")

        self.histories, self.surgeries, self.reports = {}, {}, {}

        for idx in range(1, 764):
            idx_str = str(idx).zfill(3)
            self.histories[idx_str] = self._read_txt(
                os.path.join(path_histories, f"SurgeryReport_History_{idx_str}.txt")
            )
            self.surgeries[idx_str] = self._read_txt(
                os.path.join(path_reports, f"SurgeryReport_{idx_str}.txt")
            )
            self.reports[idx_str] = self._read_txt(
                os.path.join(path_surgeries, f"SurgeryDescriptionEnglish_{idx_str}.txt")
            )

    def _prepare_hancock_lymph(self) -> None:
        """Prepare HANCOCK lymph node WSI paths."""
        subfolder = os.path.join(self.hancock_root, "WSI_LymphNode", "h5_files")
        self.hancock_lymph_paths = {}
        for i in range(1, 764):
            pid = f"{i:03d}"
            path = Path(os.path.join(subfolder, f"LymphNode_HE_{pid}.h5"))
            self.hancock_lymph_paths[pid] = str(path) if path.is_file() else None

    def _prepare_hancock_tumor(self) -> None:
        """Prepare HANCOCK primary tumor WSI paths."""
        subfolder = os.path.join(self.hancock_root, "WSI_PrimaryTumor")
        files = [str(x) for x in Path(subfolder).rglob("*.h5")]

        self.hancock_tumor_paths = {f"{i:03d}": None for i in range(1, 764)}
        pattern = re.compile(r'(\d{3})')
        for path in files:
            match = pattern.search(os.path.basename(path))
            if match:
                pid = match.group(1)
                if self.hancock_tumor_paths.get(pid) is None:
                    self.hancock_tumor_paths[pid] = []
                self.hancock_tumor_paths[pid].append(path)

    def _prepare_tcga_tumor(self) -> None:
        """Prepare TCGA-HNSC primary tumor WSI paths."""
        subfolder = os.path.join(self.tcga_root, "uni_embeddings")
        self.tcga_tumor_paths = {}

        if not os.path.exists(subfolder):
            return

        files = [str(x) for x in Path(subfolder).rglob("*.h5")]
        for path in files:
            patient_id = os.path.basename(path)[:12]  # TCGA-XX-XXXX
            if patient_id not in self.tcga_tumor_paths:
                self.tcga_tumor_paths[patient_id] = []
            self.tcga_tumor_paths[patient_id].append(path)

    # =========================================================================
    # File readers
    # =========================================================================

    @staticmethod
    def _read_txt(path: str) -> str:
        """Read text file, return placeholder if not found."""
        try:
            with open(path, 'r') as f:
                return f.read()
        except (FileNotFoundError, IOError):
            return "No report available."

    @staticmethod
    def _read_h5_1024(path: Optional[str]) -> np.ndarray:
        """Read H5 features (HANCOCK format, 1024-dim)."""
        if path is None:
            return np.zeros((1, 1024))
        with h5py.File(path, 'r') as f:
            return f['features'][:]

    @staticmethod
    def _read_h5_1536(path: Optional[str]) -> np.ndarray:
        """Read H5 features (TCGA format, 1536-dim)."""
        if path is None:
            return np.zeros((1, 1536))
        with h5py.File(path, 'r') as f:
            data = f['features'][:]
        if data.ndim == 3 and data.shape[0] == 1:
            data = data[0]
        return data

    # =========================================================================
    # Feature getters
    # =========================================================================

    def __get_clinical__(self, idx: str) -> torch.Tensor:
        """Get clinical features (step1 + step2)"""
        df_patient = self.data_clinical.loc[self.data_clinical.patient_id == idx]
        cols = get_unified_clinical_cols(list(df_patient.columns))
        return torch.tensor(df_patient[cols].values.astype(float), dtype=torch.float32).flatten()

    def __get_pathological__(self, idx: str) -> torch.Tensor:
        """Get pathological features (step3)"""
        df_patient = self.data_clinical.loc[self.data_clinical.patient_id == idx]
        cols = get_unified_pathological_cols(list(df_patient.columns))
        return torch.tensor(df_patient[cols].values.astype(float), dtype=torch.float32).flatten()

    def __get_blood__(self, idx: str) -> torch.Tensor:
        """Get blood features. Returns real data for HANCOCK, zeros for TCGA."""
        if self.data_blood is not None and idx in self.data_blood.patient_id.values:
            df_patient = self.data_blood.loc[self.data_blood.patient_id == idx]
            df_no_id = df_patient.drop(columns=['patient_id'])
            return torch.tensor(df_no_id.values.astype(float), dtype=torch.float32).flatten()
        return torch.zeros(max(self._blood_dim, 1))

    def __tokenize__(self, text: str, max_tokens: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize text with middle truncation for 512, normal truncation otherwise."""
        if max_tokens == 512:
            return self._middle_trunc(text)
        encodings = self.tokenizer(
            text, return_tensors='pt', max_length=max_tokens,
            padding='max_length', truncation=True,
        )
        return encodings['input_ids'], encodings['attention_mask']

    def _middle_trunc(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Middle truncation strategy for 512-token texts."""
        encodings = self.tokenizer(text, return_tensors='pt')
        ids = encodings['input_ids']
        mask = encodings['attention_mask']
        n_tokens = ids.shape[-1]

        if n_tokens > 512:
            half = n_tokens // 2
            ids_trunc = torch.hstack((ids[:, half - 256:half], ids[:, half:half + 256]))
            mask_trunc = torch.ones_like(ids_trunc)
            return ids_trunc, mask_trunc

        encodings = self.tokenizer(
            text, return_tensors='pt', max_length=512,
            padding='max_length', truncation=True,
        )
        return encodings['input_ids'], encodings['attention_mask']

    def __get_text__(self, patient_id: str):
        """Get tokenized text for HANCOCK patients, zero tokens for TCGA."""
        source = self._get_source(patient_id)

        if source == 'hancock':
            h_ids, h_mask = self.__tokenize__(self.histories[patient_id], self.max_tokens_history)
            s_ids, s_mask = self.__tokenize__(self.surgeries[patient_id], self.max_tokens_surgery)
            r_ids, r_mask = self.__tokenize__(self.reports[patient_id], self.max_tokens_report)
        else:
            # Zero tokens for TCGA (padded to match HANCOCK token shapes for batching)
            h_ids = torch.zeros(1, self.max_tokens_history, dtype=torch.long)
            h_mask = torch.zeros(1, self.max_tokens_history, dtype=torch.long)
            s_ids = torch.zeros(1, self.max_tokens_surgery, dtype=torch.long)
            s_mask = torch.zeros(1, self.max_tokens_surgery, dtype=torch.long)
            r_ids = torch.zeros(1, self.max_tokens_report, dtype=torch.long)
            r_mask = torch.zeros(1, self.max_tokens_report, dtype=torch.long)

        return h_ids, h_mask, s_ids, s_mask, r_ids, r_mask

    def __get_lymphnode__(self, patient_id: str) -> torch.Tensor:
        """Get lymph node WSI features. HANCOCK: 1024→pad to 1536, TCGA: zeros."""
        source = self._get_source(patient_id)

        if source == 'hancock':
            file_path = self.hancock_lymph_paths.get(patient_id)
            features = self._read_h5_1024(file_path)
            features = features.mean(0)  # (1024,)
            # Pad from 1024 to 1536
            features = np.pad(features, (0, self.TUMOR_DIM - 1024), mode='constant')
            return torch.from_numpy(features).float()

        return torch.zeros(self.TUMOR_DIM)

    def __get_primarytumor__(self, patient_id: str) -> torch.Tensor:
        """
        Get primary tumor WSI features (averaged over patches).
        HANCOCK: 1024-dim → padded to 1536-dim
        TCGA: native 1536-dim
        """
        source = self._get_source(patient_id)

        if source == 'hancock':
            list_paths = self.hancock_tumor_paths.get(patient_id)
            if list_paths is not None:
                all_feats = [self._read_h5_1024(p) for p in list_paths]
                features = np.vstack(all_feats).mean(0)  # (1024,)
            else:
                features = np.zeros(1024)
            # Pad from 1024 to 1536
            features = np.pad(features, (0, self.TUMOR_DIM - 1024), mode='constant')
            return torch.from_numpy(features).float()

        elif source == 'tcga':
            list_paths = self.tcga_tumor_paths.get(patient_id)
            if list_paths is not None and len(list_paths) > 0:
                all_feats = [self._read_h5_1536(p) for p in list_paths]
                features = np.vstack(all_feats).mean(0)  # (1536,)
            else:
                features = np.zeros(self.TUMOR_DIM)
            return torch.from_numpy(features).float()

        return torch.zeros(self.TUMOR_DIM)

    # =========================================================================
    # Main interface
    # =========================================================================

    def __getitem__(self, index: int) -> Tuple[str, Tuple, Tuple]:
        patient_id = self.samples[index]

        # Tabular
        clinical = self.__get_clinical__(patient_id)
        blood = self.__get_blood__(patient_id)
        patho = self.__get_pathological__(patient_id)

        # Text
        h_ids, h_mask, s_ids, s_mask, r_ids, r_mask = self.__get_text__(patient_id)

        # Images
        lymph = self.__get_lymphnode__(patient_id)
        tumor = self.__get_primarytumor__(patient_id)

        # Survival target
        row = self.data_clinical.loc[self.data_clinical.patient_id == patient_id]
        event = row.event.values[0]
        event = torch.tensor(int(event) if pd.notna(event) else 0, dtype=torch.int8)
        time = row.time.values[0]
        time = torch.tensor(int(time) if pd.notna(time) else 0, dtype=torch.int32)

        input_tuple = (clinical, blood, patho, h_ids, h_mask, s_ids, s_mask, r_ids, r_mask, lymph, tumor)
        target = (time, event)

        return patient_id, input_tuple, target

    def get_covariates(self) -> pd.DataFrame:
        """Return all tabular covariates (harmonized clinical data)."""
        return self.data_clinical.copy()

    def get_input_dims(self) -> Dict[str, int]:
        """Get input dimensions for each modality."""
        if len(self.samples) == 0:
            raise ValueError("Dataset is empty, cannot determine input dimensions")

        first_pid = self.samples[0]
        clinical_dim = self.__get_clinical__(first_pid).shape[0]
        patho_dim = self.__get_pathological__(first_pid).shape[0]

        return {
            "clinical": clinical_dim,
            "blood": self._blood_dim if self._blood_dim > 0 else 0,
            "pathological": patho_dim,
            "text": 768,  # BioClinicalBERT (HANCOCK patients have text)
            "lymphnode": self.TUMOR_DIM,
            "primarytumor": self.TUMOR_DIM,
        }

    def get_tmax(self) -> int:
        """Get maximum time from the dataset."""
        if len(self.samples) == 0:
            raise ValueError("Dataset is empty")
        t_max = self.data_clinical.time.max()
        return int(t_max) if pd.notna(t_max) else 0

