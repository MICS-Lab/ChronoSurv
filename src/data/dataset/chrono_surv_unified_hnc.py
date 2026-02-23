"""
ChronoSurv dataset for Unified HNC (HANCOCK + TCGA-HNSC).

Creates source-aware heterogeneous directed graphs:
- HANCOCK patients: full graph (all nodes including blood, text, lymph)
- TCGA patients: partial graph (no blood, text, lymph nodes)

This allows ChronoSurv to naturally handle missing modalities by dropping graph nodes.
"""

import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from typing import Dict
from torch_geometric.data import HeteroData

from src.data.dataset.unified_hnc import UnifiedHNCDataset
from src.data.preprocessors.unified_hnc.feature_columns import (
    get_unified_step1_cols,
    get_unified_step2_cols,
    get_unified_step3_cols,
)


class ChronoSurvUnifiedHNCDataset(UnifiedHNCDataset):
    """
    ChronoSurv dataset for Unified HNC: creates source-aware heterogeneous directed graphs.

    HANCOCK patients get full graph:
    - clinical_step1 -> step1
    - clinical (step2) + blood + history -> step2
    - pathological + surgery_report + surgery_desc + lymph + tumor -> step3

    TCGA patients get partial graph (missing modalities = missing nodes):
    - clinical_step1 -> step1
    - clinical (step2) -> step2
    - pathological + tumor -> step3
    """

    def __init__(self, aggregate_images: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.aggregate_images = aggregate_images

    # =========================================================================
    # STRUCTURED GETTERS (by timeline step)
    # =========================================================================

    def __get_clinical_step1__(self, idx: str) -> torch.Tensor:
        """step1: demographics + exposures."""
        df_patient = self.data_clinical.loc[self.data_clinical.patient_id == idx]
        cols = get_unified_step1_cols(list(df_patient.columns))
        return torch.tensor(df_patient[cols].values.astype(float), dtype=torch.float32).flatten()

    def __get_clinical__(self, idx: str) -> torch.Tensor:
        """step2: anatomy, clinical staging."""
        df_patient = self.data_clinical.loc[self.data_clinical.patient_id == idx]
        cols = get_unified_step2_cols(list(df_patient.columns))
        return torch.tensor(df_patient[cols].values.astype(float), dtype=torch.float32).flatten()

    def __get_pathological__(self, idx: str) -> torch.Tensor:
        """step3: pathologic staging, grade, invasion markers."""
        df_patient = self.data_clinical.loc[self.data_clinical.patient_id == idx]
        cols = get_unified_step3_cols(list(df_patient.columns))
        return torch.tensor(df_patient[cols].values.astype(float), dtype=torch.float32).flatten()

    # =========================================================================
    # IMAGE GETTERS (override for ChronoSurv-specific behavior)
    # =========================================================================

    def __get_lymphnode__(self, patient_id: str):
        """Get lymph node WSI features for ChronoSurv (None if not available)."""
        source = self._get_source(patient_id)
        if source != 'hancock':
            return None

        file_path = self.hancock_lymph_paths.get(patient_id)
        features = self._read_h5_1024(file_path)

        if self.aggregate_images:
            features = features.mean(0)  # (1024,)
            features = np.pad(features, (0, self.TUMOR_DIM - 1024), mode='constant')
            return torch.from_numpy(features).float()
        else:
            if features.ndim == 1:
                features = features.reshape(1, -1)
            # Pad each patch from 1024 to 1536
            features = np.pad(features, ((0, 0), (0, self.TUMOR_DIM - 1024)), mode='constant')
            return torch.from_numpy(features).float()

    def __get_primarytumor__(self, patient_id: str):
        """Get primary tumor WSI features for ChronoSurv (None if not available)."""
        source = self._get_source(patient_id)

        if source == 'hancock':
            list_paths = self.hancock_tumor_paths.get(patient_id)
            if list_paths is not None:
                all_feats = [self._read_h5_1024(p) for p in list_paths]
                features = np.vstack(all_feats)
            else:
                return None  # No tumor WSI
            # Pad from 1024 to 1536
            if self.aggregate_images:
                features = features.mean(0)
                features = np.pad(features, (0, self.TUMOR_DIM - 1024), mode='constant')
                return torch.from_numpy(features).float()
            else:
                features = np.pad(features, ((0, 0), (0, self.TUMOR_DIM - 1024)), mode='constant')
                return torch.from_numpy(features).float()

        elif source == 'tcga':
            list_paths = self.tcga_tumor_paths.get(patient_id)
            if list_paths is None or len(list_paths) == 0:
                return None
            all_feats = [self._read_h5_1536(p) for p in list_paths]
            features = np.vstack(all_feats)
            if self.aggregate_images:
                features = features.mean(0)
                return torch.from_numpy(features).float()
            else:
                return torch.from_numpy(features).float()

        return None

    # =========================================================================
    # INPUT DIMS
    # =========================================================================

    def get_input_dims(self) -> Dict[str, int]:
        if len(self.samples) == 0:
            raise ValueError("Dataset is empty, cannot determine input dimensions")

        first_pid = self.samples[0]

        return {
            # Structured
            "clinical_step1": self.__get_clinical_step1__(first_pid).shape[0],
            "clinical": self.__get_clinical__(first_pid).shape[0],
            "blood": self._blood_dim if self._blood_dim > 0 else 0,
            "pathological": self.__get_pathological__(first_pid).shape[0],
            # Text (BioClinicalBERT 768-dim, from HANCOCK patients)
            "text": 768,
            # Images (unified to 1536-dim)
            "images": self.TUMOR_DIM,
        }

    # =========================================================================
    # GRAPH CONSTRUCTION
    # =========================================================================

    def _create_hetero_graph(
        self,
        patient_id: str,
        source: str,
        clinical_step1: torch.Tensor,
        clinical: torch.Tensor,
        blood: torch.Tensor = None,
        pathological: torch.Tensor = None,
        h_ids: torch.Tensor = None,
        h_mask: torch.Tensor = None,
        s_ids: torch.Tensor = None,
        s_mask: torch.Tensor = None,
        r_ids: torch.Tensor = None,
        r_mask: torch.Tensor = None,
        lymph: torch.Tensor = None,
        tumor: torch.Tensor = None,
    ) -> HeteroData:
        data = HeteroData()

        # ═══════════════════════════════════════════════════════════════
        # LEAF NODES - Structured
        # ═══════════════════════════════════════════════════════════════
        data['clinical_step1'].x = clinical_step1.unsqueeze(0)
        data['clinical_step1'].num_nodes = 1

        data['clinical'].x = clinical.unsqueeze(0)
        data['clinical'].num_nodes = 1

        # Blood node (HANCOCK only)
        if blood is not None:
            data['blood'].x = blood.unsqueeze(0)
            data['blood'].num_nodes = 1

        data['pathological'].x = pathological.unsqueeze(0)
        data['pathological'].num_nodes = 1

        # ═══════════════════════════════════════════════════════════════
        # LEAF NODES - Text (HANCOCK only)
        # ═══════════════════════════════════════════════════════════════
        if source == 'hancock' and h_ids is not None:
            max_text_len = 512
            h_ids_padded = F.pad(h_ids.squeeze(0), (0, max_text_len - h_ids.shape[1]))
            s_ids_padded = F.pad(s_ids.squeeze(0), (0, max_text_len - s_ids.shape[1]))
            r_ids_padded = F.pad(r_ids.squeeze(0), (0, max_text_len - r_ids.shape[1]))
            h_mask_padded = F.pad(h_mask.squeeze(0), (0, max_text_len - h_mask.shape[1]))
            s_mask_padded = F.pad(s_mask.squeeze(0), (0, max_text_len - s_mask.shape[1]))
            r_mask_padded = F.pad(r_mask.squeeze(0), (0, max_text_len - r_mask.shape[1]))

            data['history'].input_ids = h_ids_padded.unsqueeze(0)
            data['history'].attention_mask = h_mask_padded.unsqueeze(0)
            data['history'].num_nodes = 1

            data['surgery_report'].input_ids = s_ids_padded.unsqueeze(0)
            data['surgery_report'].attention_mask = s_mask_padded.unsqueeze(0)
            data['surgery_report'].num_nodes = 1

            data['surgery_desc'].input_ids = r_ids_padded.unsqueeze(0)
            data['surgery_desc'].attention_mask = r_mask_padded.unsqueeze(0)
            data['surgery_desc'].num_nodes = 1

        # ═══════════════════════════════════════════════════════════════
        # LEAF NODES - Images
        # ═══════════════════════════════════════════════════════════════
        # Lymph node (HANCOCK only, may be None)
        if lymph is not None:
            if self.aggregate_images:
                data['lymph'].x = lymph.unsqueeze(0)
                data['lymph'].num_nodes = 1
            else:
                if lymph.ndim == 1:
                    lymph = lymph.unsqueeze(0)
                data['lymph'].x = lymph
                data['lymph'].num_nodes = lymph.shape[0]

        # Tumor (both sources, may be None)
        if tumor is not None:
            if self.aggregate_images:
                data['tumor'].x = tumor.unsqueeze(0)
                data['tumor'].num_nodes = 1
            else:
                if tumor.ndim == 1:
                    tumor = tumor.unsqueeze(0)
                data['tumor'].x = tumor
                data['tumor'].num_nodes = tumor.shape[0]

        # ═══════════════════════════════════════════════════════════════
        # STEP AND MASTER NODES
        # ═══════════════════════════════════════════════════════════════
        data['step1'].x = torch.zeros(1, 1)
        data['step1'].num_nodes = 1
        data['step2'].x = torch.zeros(1, 1)
        data['step2'].num_nodes = 1
        data['step3'].x = torch.zeros(1, 1)
        data['step3'].num_nodes = 1
        data['master'].x = torch.zeros(1, 1)
        data['master'].num_nodes = 1

        # ═══════════════════════════════════════════════════════════════
        # EDGES - Layer 1: Leaves -> Steps (only for existing nodes)
        # ═══════════════════════════════════════════════════════════════
        data['clinical_step1', 'to_step', 'step1'].edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        data['clinical', 'to_step', 'step2'].edge_index = torch.tensor([[0], [0]], dtype=torch.long)

        if blood is not None:
            data['blood', 'to_step', 'step2'].edge_index = torch.tensor([[0], [0]], dtype=torch.long)

        data['pathological', 'to_step', 'step3'].edge_index = torch.tensor([[0], [0]], dtype=torch.long)

        # Text edges (HANCOCK only)
        if 'history' in data.node_types:
            data['history', 'to_step', 'step2'].edge_index = torch.tensor([[0], [0]], dtype=torch.long)
            data['surgery_report', 'to_step', 'step3'].edge_index = torch.tensor([[0], [0]], dtype=torch.long)
            data['surgery_desc', 'to_step', 'step3'].edge_index = torch.tensor([[0], [0]], dtype=torch.long)

        # Image edges
        if 'lymph' in data.node_types:
            num_lymph = data['lymph'].num_nodes
            data['lymph', 'to_step', 'step3'].edge_index = torch.stack([
                torch.arange(num_lymph, dtype=torch.long),
                torch.zeros(num_lymph, dtype=torch.long),
            ], dim=0)

        if 'tumor' in data.node_types:
            num_tumor = data['tumor'].num_nodes
            data['tumor', 'to_step', 'step3'].edge_index = torch.stack([
                torch.arange(num_tumor, dtype=torch.long),
                torch.zeros(num_tumor, dtype=torch.long),
            ], dim=0)

        # ═══════════════════════════════════════════════════════════════
        # EDGES - Layer 2: Temporal + Skip + Self-loops
        # ═══════════════════════════════════════════════════════════════
        data['step1', 'temporal', 'step2'].edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        data['step2', 'temporal', 'step3'].edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        data['step1', 'skip', 'step3'].edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        data['step1', 'self', 'step1'].edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        data['step2', 'self', 'step2'].edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        data['step3', 'self', 'step3'].edge_index = torch.tensor([[0], [0]], dtype=torch.long)

        # ═══════════════════════════════════════════════════════════════
        # EDGES - Layer 3: Steps -> Master
        # ═══════════════════════════════════════════════════════════════
        data['step1', 'to_master', 'master'].edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        data['step2', 'to_master', 'master'].edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        data['step3', 'to_master', 'master'].edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        data['master', 'self', 'master'].edge_index = torch.tensor([[0], [0]], dtype=torch.long)

        # ═══════════════════════════════════════════════════════════════
        # PATIENT METADATA AND TARGETS
        # ═══════════════════════════════════════════════════════════════
        data.patient_id = patient_id

        row = self.data_clinical.loc[self.data_clinical.patient_id == patient_id]
        event = row.event.values[0]
        data.event = torch.tensor(int(event) if pd.notna(event) else 0, dtype=torch.long)
        time = row.time.values[0]
        data.time = torch.tensor(float(time) if pd.notna(time) else 0, dtype=torch.float32)

        return data

    def __getitem__(self, index: int) -> HeteroData:
        patient_id = self.samples[index]
        source = self._get_source(patient_id)

        # Structured
        clinical_step1 = self.__get_clinical_step1__(patient_id)
        clinical = self.__get_clinical__(patient_id)
        pathological = self.__get_pathological__(patient_id)

        # Blood (HANCOCK only)
        blood = None
        if source == 'hancock' and self._blood_dim > 0:
            blood = self.__get_blood__(patient_id)

        # Text (HANCOCK only)
        h_ids, h_mask, s_ids, s_mask, r_ids, r_mask = None, None, None, None, None, None
        if source == 'hancock':
            h_ids, h_mask = self.__tokenize__(self.histories[patient_id], self.max_tokens_history)
            s_ids, s_mask = self.__tokenize__(self.surgeries[patient_id], self.max_tokens_surgery)
            r_ids, r_mask = self.__tokenize__(self.reports[patient_id], self.max_tokens_report)

        # Images
        lymph = self.__get_lymphnode__(patient_id)
        tumor = self.__get_primarytumor__(patient_id)

        return self._create_hetero_graph(
            patient_id=patient_id,
            source=source,
            clinical_step1=clinical_step1,
            clinical=clinical,
            blood=blood,
            pathological=pathological,
            h_ids=h_ids, h_mask=h_mask,
            s_ids=s_ids, s_mask=s_mask,
            r_ids=r_ids, r_mask=r_mask,
            lymph=lymph, tumor=tumor,
        )

