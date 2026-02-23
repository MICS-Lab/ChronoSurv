import lightning as L
import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Literal
from collections import defaultdict
from torchmetrics import MeanMetric, MetricCollection


class BaseLightningModule(L.LightningModule):
    """
    Base Lightning Module that centralizes common training logic.
    
    Handles:
    - Optimizer configuration (Adam, AdamW, SGD)
    - Scheduler configuration (ConstantLR, StepLR, ExponentialLR, CosineAnnealingLR, ReduceLROnPlateau)
    - Loss accumulation and logging (and metrics logging)
    - Checkpoint save/load for metrics history
    """
    
    def __init__(
        self,
        model: nn.Module,
        # Optimizer parameters
        optimizer_name: str = "AdamW",
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        momentum: float = 0.9,
        # Scheduler parameters
        scheduler_name: str = "ConstantLR",
        step_size: int = 30,
        gamma: float = 0.1,
        factor: float = 1.0,
        last_epoch: int = -1,
        scheduler_T_max: int = 100,
        scheduler_eta_min: float = 0.0,
        # Metrics parameters
        main_metric: str = "val_loss",
        **kwargs
    ):
        super().__init__()
        
        # Save hyperparameters for Lightning
        self.save_hyperparameters(ignore=["model"])
        
        self.model = model
        
        self.optimizer_name = optimizer_name
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.momentum = momentum
        
        self.scheduler_name = scheduler_name
        self.step_size = step_size
        self.gamma = gamma
        self.factor = factor
        self.last_epoch = last_epoch
        self.scheduler_T_max = scheduler_T_max
        self.scheduler_eta_min = scheduler_eta_min
        
        self.main_metric = main_metric or "val_loss"
        
        # Initialize metrics: initialise self.train_metrics, self.val_metrics, self.test_metrics (MetricCollection) 
        # Is empty for now, will be filled by subclasses      
        base_collection = MetricCollection([])
        self.train_metrics = base_collection.clone()
        self.val_metrics = base_collection.clone()
        self.test_metrics = base_collection.clone()        
        
        # Epoch loss accumulators
        self.train_epoch_loss_accumulator = MeanMetric()
        self.val_epoch_loss_accumulator = MeanMetric()
        self.test_epoch_loss_accumulator = MeanMetric()
        
        # For callbacks and history tracking
        # Cache stores accumulated metrics/loss values during the epoch (not just last batch)
        # Allows access to current epoch metrics without waiting for epoch end
        self.logged_metrics_cache: Dict[str, float] = defaultdict(float)
        self.metric_history: Dict[str, List[float]] = {}

    ########################################################
    # Public Static Methods
    ########################################################
    
    @staticmethod
    def get_metric_mode(metric: str) -> Literal["min", "max"]:
        """Get the mode for the main metric."""
        #TODO: bad way to do this, should be more robust
        max_metrics = ["c_index"]
        if any(m in metric.lower() for m in max_metrics):
            return "max"
        min_metrics = ["loss", "ibs", "error", "mae", "mse", "rmse"]
        if any(m in metric.lower() for m in min_metrics):
            return "min"
        return "max"

    ########################################################
    # Lightning Hooks
    ########################################################
    
    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizer and scheduler."""
        optimizer = self._create_optimizer()
        scheduler_config = self._create_scheduler(optimizer)
        
        if scheduler_config is None:
            return {"optimizer": optimizer}
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler_config
        }
    
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Save metrics history to checkpoint."""
        checkpoint["metrics_history"] = self.metric_history
        checkpoint["logged_metrics_cache"] = self.logged_metrics_cache

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Load metrics history from checkpoint."""
        self.metric_history = checkpoint.get("metrics_history", {})
        self.logged_metrics_cache = checkpoint.get("logged_metrics_cache", {})
    
    def on_train_epoch_end(self) -> None:
        """Update metric history from Lightning's callback metrics at the end of training epoch."""
        # Remark: lightning considers a training epoch end after training + validation
        for name, value in self.trainer.callback_metrics.items():
            if name.startswith("train/") or name.startswith("val/") or name.startswith("test/"):
                val = value.item() if torch.is_tensor(value) else value
                self.metric_history.setdefault(name, []).append(val)

    ########################################################
    # Public Instance Methods
    ########################################################
    
    def compute_and_log_metrics(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        phase: str = "train",
        batch_size: int = 1
    ) -> None:
        """Compute and log metrics for the given phase."""
        # Get the appropriate metrics collection
        if phase == "train":
            metrics = self.train_metrics
        elif phase == "val":
            metrics = self.val_metrics
        elif phase == "test":
            metrics = self.test_metrics
        else:
            raise ValueError(f"Unknown phase: {phase}")

        metrics.update(preds, targets)
        for name, metric in metrics.items():
            self.log(f"{phase}/{name}", metric, on_epoch=True, prog_bar=False, on_step=False, batch_size=batch_size, sync_dist=True)
            self.logged_metrics_cache[f"{phase}/{name}"] = metric.compute().item()

    def accumulate_and_log_loss(
        self,
        loss: torch.Tensor,
        phase: str = "train",
        batch_size: int = 1,
        prog_bar: bool = True
    ) -> None:
        """Accumulate and log loss for the given phase."""
        # Get the appropriate loss accumulator
        if phase == "train":
            loss_accumulator = self.train_epoch_loss_accumulator
        elif phase == "val":
            loss_accumulator = self.val_epoch_loss_accumulator
        elif phase == "test":
            loss_accumulator = self.test_epoch_loss_accumulator
        else:
            raise ValueError(f"Unknown phase: {phase}")
        
        # Update loss accumulator
        loss_accumulator.update(loss)
        
        # Log loss
        self.log(f"{phase}/loss_step", loss, on_step=True, prog_bar=True, on_epoch=False, batch_size=batch_size, sync_dist=True)
        self.log(f"{phase}/loss_epoch", loss_accumulator, on_epoch=True, prog_bar=True, on_step=False, batch_size=batch_size, sync_dist=True)
        
        # Track step-wise loss
        self.metric_history.setdefault(f"{phase}/loss_step", []).append(loss.item())
        
        # Update cache for callbacks
        self.logged_metrics_cache[f"{phase}/loss_epoch"] = loss_accumulator.compute().item()
        self.logged_metrics_cache[f"{phase}/loss_step"] = loss.item()
    
    def get_current_loss(self, phase: str = "train") -> float:
        """Get current epoch loss for the specified phase."""
        return self.logged_metrics_cache.get(f"{phase}/loss_epoch", 0.0)
    
    def get_current_metric(self, metric: str, phase: str = "train") -> Optional[float]:
        """Get current epoch metric for the specified phase."""
        return self.logged_metrics_cache.get(f"{phase}/{metric}")

    ########################################################
    # Private/Protected Instance Methods
    ########################################################
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer based on optimizer name."""
        if self.optimizer_name == "AdamW":
            return torch.optim.AdamW(
                self.parameters(), 
                lr=self.learning_rate, 
                weight_decay=self.weight_decay
            )
        elif self.optimizer_name == "Adam":
            return torch.optim.Adam(
                self.parameters(), 
                lr=self.learning_rate, 
                weight_decay=self.weight_decay
            )
        elif self.optimizer_name == "SGD":
            return torch.optim.SGD(
                self.parameters(), 
                lr=self.learning_rate, 
                weight_decay=self.weight_decay,
                momentum=self.momentum
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")
    
    def _create_scheduler(self, optimizer: torch.optim.Optimizer) -> Optional[Dict[str, Any]]:
        """Create scheduler configuration based on scheduler name."""
        if self.scheduler_name == "ConstantLR":
            scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer,
                factor=self.factor,
                last_epoch=self.last_epoch
            )
            return {
                "scheduler": scheduler,
                "frequency": 1
            }
        elif self.scheduler_name == "StepLR":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.step_size,
                gamma=self.gamma,
                last_epoch=self.last_epoch
            )
            return {
                "scheduler": scheduler,
                "frequency": 1
            }
        elif self.scheduler_name == "ExponentialLR":
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=self.gamma,
                last_epoch=self.last_epoch
            )
            return {
                "scheduler": scheduler,
                "frequency": 1
            }
        elif self.scheduler_name == "CosineAnnealingLR":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.scheduler_T_max,
                eta_min=self.scheduler_eta_min,
                last_epoch=self.last_epoch
            )
            return {
                "scheduler": scheduler,
                "frequency": 1
            }
        elif self.scheduler_name == "ReduceLROnPlateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=self.gamma,
                patience=self.step_size
            )
            return {
                "scheduler": scheduler,
                "monitor": self.main_metric,
                "frequency": 1
            }
        elif self.scheduler_name == "None" or self.scheduler_name is None:
            return None
        else:
            raise ValueError(f"Unsupported scheduler: {self.scheduler_name}")

