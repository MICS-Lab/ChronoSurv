import torch
import numpy as np
from typing import Any, List, Tuple
from src.training.lightning_module.base_module import BaseLightningModule
from sksurv.util import Surv
from sksurv.metrics import integrated_brier_score, concordance_index_ipcw


class SurvivalBaseLightningModule(BaseLightningModule):
    """
    Base Lightning Module for survival analysis models.
    
    Extends BaseLightningModule with survival-specific functionality:
    - IBS (Integrated Brier Score) tracking for train/val
    - Caching system for epoch-level metrics computation
    - Common survival metrics utilities
    
    Child classes should:
    - Call super().__init__() in their __init__
    - Return survival_probs, times, events in _shared_step outputs
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # For IBS computation
        self.y_train = None  # Surv array for training data (for IPCW)
        self.time_coordinates = None  # Time points for IBS evaluation
        
        # Training predictions cache for survival metrics computation
        self.train_predictions_cache: List[torch.Tensor] = []
        self.train_times_cache: List[torch.Tensor] = []
        self.train_events_cache: List[torch.Tensor] = []
        self.train_risk_scores_cache: List[torch.Tensor] = []
        
        # Validation predictions cache for survival metrics computation
        self.val_predictions_cache: List[torch.Tensor] = []
        self.val_times_cache: List[torch.Tensor] = []
        self.val_events_cache: List[torch.Tensor] = []
        self.val_risk_scores_cache: List[torch.Tensor] = []
    
    ########################################################
    # Public Methods (to be called by child classes)
    ########################################################
    
    def setup_survival_metrics(
        self, 
        train_times: np.ndarray, 
        train_events: np.ndarray,
        time_coordinates: np.ndarray
    ) -> None:
        """
        Setup survival metrics computation.
        
        Should be called in child's on_fit_start() after extracting train data.
        
        Args:
            train_times: Training times [N]
            train_events: Training event indicators [N]
            time_coordinates: Time points for IBS evaluation [M]
        """
        self.y_train = Surv.from_arrays(event=train_events.astype(bool), time=train_times)
        self.time_coordinates = time_coordinates
        print(f"  - Survival metrics prepared: N={len(train_times)}, M={len(time_coordinates)}")
    
    def cache_survival_predictions(
        self,
        survival_probs: torch.Tensor,
        times: torch.Tensor,
        events: torch.Tensor,
        risk_scores: torch.Tensor,
        stage: str
    ) -> None:
        """
        Cache survival predictions for epoch-level metrics computation.
        
        Should be called in child's training_step/validation_step.
        
        Args:
            survival_probs: Survival probabilities [batch_size, num_time_points]
            times: Observed times [batch_size]
            events: Event indicators [batch_size]
            risk_scores: Risk scores [batch_size] (higher = higher risk)
            stage: "train" or "val"
        """
        if stage == "train":
            self.train_predictions_cache.append(survival_probs.detach().cpu())
            self.train_times_cache.append(times.detach().cpu())
            self.train_events_cache.append(events.detach().cpu())
            self.train_risk_scores_cache.append(risk_scores.detach().cpu())
        elif stage == "val":
            self.val_predictions_cache.append(survival_probs.detach().cpu())
            self.val_times_cache.append(times.detach().cpu())
            self.val_events_cache.append(events.detach().cpu())
            self.val_risk_scores_cache.append(risk_scores.detach().cpu())
    
    ########################################################
    # Lightning Hooks (override parent)
    ########################################################
    
    def on_train_epoch_start(self) -> None:
        """Clear train cache at the start of each training epoch."""
        self.train_predictions_cache = []
        self.train_times_cache = []
        self.train_events_cache = []
        self.train_risk_scores_cache = []
    
    def on_validation_epoch_start(self) -> None:
        """Clear validation cache at the start of each validation epoch."""
        self.val_predictions_cache = []
        self.val_times_cache = []
        self.val_events_cache = []
        self.val_risk_scores_cache = []
    
    def on_train_epoch_end(self) -> None:
        """Compute survival metrics at the end of training epoch."""
        # Compute metrics BEFORE calling super() so they get added to metric_history
        if self.train_predictions_cache and self.y_train is not None:
            self._compute_and_log_survival_metrics(
                predictions_cache=self.train_predictions_cache,
                times_cache=self.train_times_cache,
                events_cache=self.train_events_cache,
                risk_scores_cache=self.train_risk_scores_cache,
                stage="train"
            )
        
        # Call parent to update metric history (will copy all logged metrics)
        super().on_train_epoch_end()
    
    def on_validation_epoch_end(self) -> None:
        """Compute survival metrics at the end of validation epoch."""
        if self.val_predictions_cache and self.y_train is not None:
            self._compute_and_log_survival_metrics(
                predictions_cache=self.val_predictions_cache,
                times_cache=self.val_times_cache,
                events_cache=self.val_events_cache,
                risk_scores_cache=self.val_risk_scores_cache,
                stage="val"
            )
    
    ########################################################
    # Private/Protected Methods
    ########################################################
    
    def _compute_and_log_survival_metrics(
        self,
        predictions_cache: List[torch.Tensor],
        times_cache: List[torch.Tensor],
        events_cache: List[torch.Tensor],
        risk_scores_cache: List[torch.Tensor],
        stage: str
    ) -> None:
        """
        Compute and log all survival metrics for a given stage.
        
        Args:
            predictions_cache: List of survival probability tensors
            times_cache: List of time tensors
            events_cache: List of event tensors
            risk_scores_cache: List of risk score tensors
            stage: "train" or "val"
        """
        # Concatenate all batches
        all_survival_probs = torch.cat(predictions_cache, dim=0)  # [N, num_time_points]
        all_times = torch.cat(times_cache, dim=0)  # [N]
        all_events = torch.cat(events_cache, dim=0)  # [N]
        all_risk_scores = torch.cat(risk_scores_cache, dim=0)  # [N]
        
        # Convert to numpy
        survival_probs_np = all_survival_probs.numpy()
        times_np = all_times.numpy()
        events_np = all_events.numpy().astype(bool)
        risk_scores_np = all_risk_scores.numpy()
        
        # Create Surv array
        y_data = Surv.from_arrays(event=events_np, time=times_np)
        
        # Compute IBS
        self._compute_and_log_ibs(
            y_data=y_data,
            survival_probs=survival_probs_np,
            times=times_np,
            stage=stage
        )
        
        # Compute C-index (Uno)
        self._compute_and_log_cindex(
            y_data=y_data,
            risk_scores=risk_scores_np,
            stage=stage
        )
        
        # Compute composite metric: c_index - ibs (higher is better)
        # This balances discrimination (c_index) and calibration (ibs)
        self._compute_and_log_composite_metric(stage=stage)
    
    def _compute_and_log_composite_metric(self, stage: str) -> None:
        """
        Compute and log composite metric: c_index_norm - ibs_norm.
        """
        c_index = self.logged_metrics_cache.get(f"{stage}/c_index")
        ibs = self.logged_metrics_cache.get(f"{stage}/ibs")
        
        if c_index is not None and ibs is not None:
            # c_index - ibs (maximize, higher is better)
            composite = c_index - ibs
            self.log(f"{stage}/c_index_minus_ibs", composite, on_epoch=True, prog_bar=False, sync_dist=True)
            self.logged_metrics_cache[f"{stage}/c_index_minus_ibs"] = float(composite)
    
    def _compute_and_log_ibs(
        self,
        y_data: Surv,
        survival_probs: np.ndarray,
        times: np.ndarray,
        stage: str
    ) -> None:
        """
        Compute and log IBS (Integrated Brier Score).
        
        Args:
            y_data: Surv array for the data
            survival_probs: Survival probabilities [N, M]
            times: Observed times [N]
            stage: "train" or "val"
        """
        if self.time_coordinates is None:
            return
        
        # Filter time points to be within data range
        min_time, max_time = times.min(), times.max()
        valid_indices = (self.time_coordinates >= min_time) & (self.time_coordinates < max_time)
        
        if valid_indices.sum() > 0:
            valid_times = self.time_coordinates[valid_indices]
            valid_survival = survival_probs[:, valid_indices]
            
            try:
                # Compute IBS
                ibs = integrated_brier_score(
                    survival_train=self.y_train,
                    survival_test=y_data,
                    estimate=valid_survival,
                    times=valid_times
                )
                
                # Log IBS
                self.log(f"{stage}/ibs", ibs, on_epoch=True, prog_bar=True, sync_dist=True)
                
                # Update cache for callbacks
                self.logged_metrics_cache[f"{stage}/ibs"] = float(ibs)
                
            except Exception as e:
                print(f"Warning: Could not compute IBS for {stage}: {e}")
    
    def _compute_and_log_cindex(
        self,
        y_data: Surv,
        risk_scores: np.ndarray,
        stage: str,
        tau: float = 5 * 365
    ) -> None:
        """
        Compute and log C-index (Uno) with IPCW.
        
        Args:
            y_data: Surv array for the data
            risk_scores: Risk scores [N] (higher = higher risk)
            stage: "train" or "val"
            tau: Truncation time for C-index (if None, use max time)
        """
        try:            
            # Compute C-index with IPCW
            cindex, _, _, _, _ = concordance_index_ipcw(
                survival_train=self.y_train,
                survival_test=y_data,
                estimate=risk_scores,
                tau=tau
            )
            
            # Log C-index
            self.log(f"{stage}/c_index", cindex, on_epoch=True, prog_bar=True, sync_dist=True)
            
            # Update cache for callbacks
            self.logged_metrics_cache[f"{stage}/c_index"] = float(cindex)
            
        except Exception as e:
            print(f"Warning: Could not compute C-index for {stage}: {e}")

