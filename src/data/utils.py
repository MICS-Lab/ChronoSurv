import torch
import tqdm
import numpy as np
import pandas as pd
import lightning as L
from typing import Literal, Tuple, Optional
from transformers import AutoModel

from src.data.datamodule.chrono_surv_unified_hnc import ChronoSurvUnifiedHNCDataModule


def process_to_array(
    datamodule: L.LightningDataModule,
    stage: Literal["train", "val", "test", "predict"]
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """
    Transform multimodal data into a flat feature matrix for classical ML models.
    
    Supports Unified HNC datasets. Automatically detects which
    modalities are present based on input_dims and only processes those.
    
    Args:
        datamodule: DataModule containing the data loaders
        stage: Data split to process (train/val/test/predict)
    
    Returns:
        Tuple containing:
            - X: Feature matrix (DataFrame) with all modalities concatenated
            - time_to_event: Time until event occurrence
            - survival_status: Censoring status
            - patient_ids: Patient identifiers
    """
    time_to_event, survival_status, patient_ids = [], [], []
    clinica_list, blood_list, patho_list, lymph_list, tumor_list = [], [], [], [], []
    h_embeddings_list, s_embeddings_list, r_embeddings_list = [], [], []

    # Get input dims to know which modalities are available
    input_dims = datamodule.get_input_dims()
    has_clinical = input_dims.get("clinical", 0) > 0
    has_blood = input_dims.get("blood", 0) > 0
    has_pathological = input_dims.get("pathological", 0) > 0
    has_text = input_dims.get("text", 0) > 0
    has_lymphnode = input_dims.get("lymphnode", 0) > 0
    has_primarytumor = input_dims.get("primarytumor", 0) > 0

    # Load text encoder only if text modality is present
    text_encoder = None
    if has_text and hasattr(datamodule, 'path_lm'):
        text_encoder = AutoModel.from_pretrained(
            datamodule.path_lm, 
            local_files_only=True
        )
        text_encoder.eval()
    
    # Handle HeteroGraph DataModules: use base datasets instead of graph-wrapped ones
    if isinstance(datamodule, ChronoSurvUnifiedHNCDataModule):
        if stage == "train":
            dataset = datamodule.base_train_dataset
        elif stage == "val":
            dataset = datamodule.base_val_dataset
        elif stage in ["test", "predict"]:
            dataset = datamodule.base_test_dataset
        
        # Create a standard DataLoader for the base dataset
        datadloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=datamodule.batch_size, 
            shuffle=False
        )
    else:
        if stage == "train":
            datadloader = datamodule.train_dataloader()
        elif stage == "val":
            datadloader = datamodule.val_dataloader()
        elif stage in ["test", "predict"]:
            datadloader = datamodule.test_dataloader()
    
    # Process batches
    with torch.no_grad():
        for p_id, input, target in tqdm.tqdm(datadloader, desc=f"Parse to Array: {stage} data", unit="batch"):
            time_to_event.append(target[0])
            survival_status.append(target[1])
            
            # Handle patient_ids (may be tensor for HANCOCK, list for TCGA)
            if isinstance(p_id, torch.Tensor):
                patient_ids.append(p_id)
            else:
                patient_ids.extend(p_id)
            
            clinical, blood, patho, h_ids, h_mask, s_ids, s_mask, r_ids, r_mask, lymph, tumor = input
            
            # Collect available modalities
            if has_clinical:
                clinica_list.append(clinical)
            if has_blood:
                blood_list.append(blood)
            if has_pathological:
                patho_list.append(patho)
            if has_lymphnode:
                lymph_list.append(lymph)
            if has_primarytumor:
                tumor_list.append(tumor)
            
            # Extract text embeddings (only if text is present)
            if has_text and text_encoder is not None:
                # Squeeze extra dimension from tokenization (shape: [batch, 1, seq] -> [batch, seq])
                h_ids, h_mask = h_ids.squeeze(1), h_mask.squeeze(1)
                s_ids, s_mask = s_ids.squeeze(1), s_mask.squeeze(1)
                r_ids, r_mask = r_ids.squeeze(1), r_mask.squeeze(1)
                
                h_embeddings = text_encoder(h_ids, h_mask).last_hidden_state[:, 0, :]
                s_embeddings = text_encoder(s_ids, s_mask).last_hidden_state[:, 0, :]
                r_embeddings = text_encoder(r_ids, r_mask).last_hidden_state[:, 0, :]
                
                h_embeddings_list.append(h_embeddings)
                s_embeddings_list.append(s_embeddings)
                r_embeddings_list.append(r_embeddings)

    # Build feature matrix from available modalities
    arrays_to_concat = []
    
    if has_clinical:
        arrays_to_concat.append(np.concatenate(clinica_list))
    if has_blood:
        arrays_to_concat.append(np.concatenate(blood_list))
    if has_pathological:
        arrays_to_concat.append(np.concatenate(patho_list))
    if has_text and text_encoder is not None:
        arrays_to_concat.append(np.concatenate(h_embeddings_list))
        arrays_to_concat.append(np.concatenate(s_embeddings_list))
        arrays_to_concat.append(np.concatenate(r_embeddings_list))
    if has_lymphnode:
        arrays_to_concat.append(np.concatenate(lymph_list))
    if has_primarytumor:
        arrays_to_concat.append(np.concatenate(tumor_list))

    # Early fusion of all features into a single flat matrix
    X = np.concatenate(arrays_to_concat, axis=1)
    X = pd.DataFrame(X)

    time_to_event = np.concatenate(time_to_event)
    survival_status = np.concatenate(survival_status)
    
    # Handle patient_ids (tensor or list)
    if isinstance(patient_ids[0], torch.Tensor):
        patient_ids = np.concatenate(patient_ids)
    else:
        patient_ids = np.array(patient_ids)
    
    return X, time_to_event, survival_status, patient_ids
