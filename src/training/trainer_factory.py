import lightning as L
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import EarlyStopping
from typing import Tuple, List, Optional
from dataclasses import asdict

from src.utils.config import Config, TrainingConfig
from src.model.chrono_surv import ChronoSurv

from src.training.lightning_module.base_module import BaseLightningModule
from src.training.lightning_module.chrono_surv_module import ChronoSurvModule

from src.training.lightning_callback.progress_bar import CustomProgressBarCallback
from src.training.lightning_callback.checkpoint import EnhancedModelCheckpoint
from src.training.lightning_callback.prediction_writer import SurvivalPredictionWriter


class TrainerFactory:
    """
    Factory for creating Lightning Trainer and Lightning Module.
    
    Handles all the setup including:
    - Lightning Trainer configuration
    - Callbacks (checkpointing, early stopping, LR monitoring)
    - Lightning Module creation with explicit parameters
    """
    
    @staticmethod
    def create_trainer_and_module(
        config: Config,
        model: nn.Module,
        experiment_name: Optional[str] = None
    ) -> Tuple[L.Trainer, BaseLightningModule]:
        """
        Create trainer and module from config.
        
        Args:
            config: Full configuration object
            model: PyTorch model
            experiment_name: Optional experiment name for logging
            
        Returns:
            Tuple of (trainer, lightning_module)
        """
        lightning_module = TrainerFactory._create_lightning_module(config.training, model)
        trainer = TrainerFactory._create_trainer(config, experiment_name)
        return trainer, lightning_module

    @staticmethod
    def _create_lightning_module(training_config: TrainingConfig, model: nn.Module) -> BaseLightningModule:
        """Create Lightning Module from config by extracting explicit parameters."""        
        if isinstance(model, ChronoSurv):
            return ChronoSurvModule(
                model=model,
                **asdict(training_config.optimizer),
                **asdict(training_config.scheduler),
                **asdict(training_config.learning_stat),
                **training_config.lightning_module_args
            )
        else:
            raise ValueError(f"Unsupported model type: {type(model)}")
    
    @staticmethod
    def _create_trainer(config: Config, experiment_name: Optional[str] = None) -> L.Trainer:
        """Create Lightning Trainer with callbacks and loggers."""        
        ########################
        # Accelerator and devices
        ########################
        if config.gpu_index >= 0:
            if torch.backends.mps.is_available():
                accelerator, devices = "mps", 1
            else:
                accelerator, devices = "gpu", [config.gpu_index]
        else:
            accelerator, devices = "cpu", "auto"
        
        ########################
        # Callbacks
        ########################
        callbacks = TrainerFactory._create_callbacks(config)

        
        ########################
        # Trainer
        ########################
        trainer = L.Trainer(
            max_epochs=config.training.epochs,
            accelerator=accelerator,
            devices=devices,
            callbacks=callbacks,
            check_val_every_n_epoch=1,
            num_sanity_val_steps=0,
            enable_checkpointing=True,
            enable_progress_bar=False,
            enable_model_summary=True,
            deterministic=True,
            logger=False,
            log_every_n_steps=1,
        )
        
        return trainer
    
    @staticmethod
    def _create_callbacks(config: Config) -> List[L.Callback]:
        """Create Lightning callbacks."""
        training_config = config.training
        
        callbacks = []      
        # Model Checkpoint callback
        checkpoint_config = training_config.checkpoints
        mode = BaseLightningModule.get_metric_mode(training_config.learning_stat.main_metric)
        checkpoint_callback = EnhancedModelCheckpoint(
            dirpath=checkpoint_config.checkpoint_path,
            monitor=f"val/{training_config.learning_stat.main_metric}",
            mode=mode,
            save_top_k=1,
            save_last=False,
            filename="best",
            every_n_epochs=checkpoint_config.save_interval,
            verbose=checkpoint_config.verbose,
            enable_version_counter = False,
        )
        callbacks.append(checkpoint_callback)
        
        # Custom Progress Bar callback
        progress_bar_callback = CustomProgressBarCallback()
        callbacks.append(progress_bar_callback)
        
        # Early Stopping callback
        early_stopping = EarlyStopping(
            monitor=f"val/{training_config.learning_stat.main_metric}",
            mode=mode,
            patience=checkpoint_config.patience,
            min_delta=checkpoint_config.delta,
            verbose=True
        )
        callbacks.append(early_stopping)
        
        # Survival Prediction Writer callback
        prediction_writer = SurvivalPredictionWriter(output_dir=f"{checkpoint_config.checkpoint_path}/prediction")
        callbacks.append(prediction_writer)
        
        return callbacks
