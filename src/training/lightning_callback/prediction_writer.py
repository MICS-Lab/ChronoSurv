import os
from lightning.pytorch.callbacks import BasePredictionWriter
import torch
import pandas as pd


class SurvivalPredictionWriter(BasePredictionWriter):
    """
    Lightning callback to aggregate and save survival predictions to CSV.
    
    This callback works with Lightning's predict loop for survival models:
    1. Lightning calls predict_step() for each batch and accumulates results
    2. This callback receives all accumulated predictions at epoch end
    3. Concatenates predictions from all batches (DDP-safe)
    4. Saves to CSV file (only on rank 0 for multi-GPU)
    
    Usage:
        Used in conjunction with predict_step() that returns Dict[str, Tensor]:
        - "p_id": Patient IDs [batch_size]
        - "time": Survival times [batch_size]
        - "event": Event indicators (1=event, 0=censored) [batch_size]
        - "risk_scores": Risk scores (hazard ratios) [batch_size]
        - "S_1", "S_2", ..., "S_{T_max}": Survival probabilities at each time point [batch_size]
    
    Args:
        output_dir: Directory where predictions.csv will be saved
    """
    def __init__(self, output_dir: str):
        super().__init__(write_interval="epoch")
        self.output_dir = output_dir
        self.split_names = ["val", "test"]
        
    def write_on_epoch_end(
        self, 
        trainer, 
        pl_module, 
        predictions, 
        batch_indices
    ):
        """
        Aggregate survival predictions from all batches and save to CSV.
        
        Args:
            trainer: Lightning Trainer instance
            pl_module: Lightning Module instance
            predictions: List of batch predictions from predict_step()
            batch_indices: Indices of batches (unused here)
        """
        if trainer.global_rank != 0:
            return 
        os.makedirs(self.output_dir, exist_ok=True)
        
        per_loader = predictions if (predictions and isinstance(predictions[0], list)) else [predictions]
        
        for dl_idx, preds in enumerate(per_loader):
            df = self._build_df(preds)
            df.to_csv(f"{self.output_dir}/predictions_{self.split_names[dl_idx]}.csv", index=False)
    
    def _concat_patient_ids(self, predictions) -> pd.Series:
        patient_ids = []
        tensor_patient_ids = []
        for prediction in predictions:
            p_id = prediction["p_id"]
            if isinstance(p_id, torch.Tensor):
                tensor_patient_ids.append(p_id)
            else:
                patient_ids.extend(list(p_id))

        if tensor_patient_ids:
            patient_ids.extend(torch.cat(tensor_patient_ids).cpu().numpy().tolist())

        return pd.Series(patient_ids)

    def _build_df(self, predictions) -> pd.DataFrame:
        all_times = torch.cat([p["time"] for p in predictions])
        all_events = torch.cat([p["event"] for p in predictions])
        all_risk_scores = torch.cat([p["risk_scores"] for p in predictions])
        
        df_data = {
            "patient_id": self._concat_patient_ids(predictions).values,
            "time": all_times.cpu().numpy(),
            "event": all_events.cpu().numpy(),
            "risk_score": all_risk_scores.cpu().numpy()
        }
        
        # Add survival probabilities columns (S_1, S_2, ..., S_{T_max})
        # Extract all S_k keys from the first prediction to get the time grid
        survival_keys = [k for k in predictions[0].keys() if k.startswith("S_")]
        for s_key in sorted(survival_keys, key=lambda x: int(x.split('_')[1])):
            all_s_values = torch.cat([p[s_key] for p in predictions])
            df_data[s_key] = all_s_values.cpu().numpy()
        
        return pd.DataFrame(df_data)