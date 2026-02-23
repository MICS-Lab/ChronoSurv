from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Literal
import yaml
import os 
import datetime

########################################################
# Training Config
########################################################

@dataclass
class OptimizerConfig:
    """Optimizer parameters (atomic)."""
    optimizer_name: str = field(default="AdamW", metadata={"help": "Optimizer type (e.g. 'AdamW', 'Adam', 'SGD')"})
    learning_rate: float = field(default=1e-3, metadata={"help": "Learning rate for optimizer"})
    weight_decay: float = field(default=1e-4, metadata={"help": "Weight decay (L2 regularization)"})

@dataclass
class SchedulerConfig:
    """Scheduler parameters (atomic)."""
    scheduler_name: str = field(default="ConstantLR", metadata={"help": "Scheduler type (e.g. 'StepLR', 'ExponentialLR', 'ReduceLROnPlateau', 'CosineAnnealingLR')"})
    step_size: int = field(default=30, metadata={"help": "Step size for StepLR scheduler"})
    gamma: float = field(default=0.1, metadata={"help": "Gamma factor for StepLR/ExponentialLR schedulers"})
    factor: float = field(default=1.0, metadata={"help": "Factor for ConstantLR scheduler"})
    last_epoch: int = field(default=-1, metadata={"help": "Last epoch for scheduler initialization"})
    # CosineAnnealingLR parameters
    scheduler_T_max: int = field(default=100, metadata={"help": "Maximum number of iterations for CosineAnnealingLR"})
    scheduler_eta_min: float = field(default=0.0, metadata={"help": "Minimum learning rate for CosineAnnealingLR"})

@dataclass
class CheckpointConfig:
    """Checkpoint parameters (atomic)."""
    checkpoint_dir: str = field(default="results/test", metadata={"help": "Checkpoint directory"})
    checkpoint_path: Optional[str] = field(default=None, metadata={"help": "Full checkpoint path"})
    
    # Other checkpoint parameters
    patience: int = field(default=10, metadata={"help": "Early stopping patience"})
    delta: float = field(default=0.0, metadata={"help": "Early stopping delta"})
    save_interval: int = field(default=1, metadata={"help": "Save checkpoint every N epochs"})
    verbose: bool = field(default=False, metadata={"help": "Verbose output for checkpointing"})

@dataclass
class LearningStatConfig:
    """Learning statistics parameters (atomic)."""
    main_metric: str = field(default="loss", metadata={"help": "Main metric to track"})
    metrics: List[str] = field(default_factory=lambda: ["loss", "accuracy", "f1_score", "precision", "recall"], metadata={"help": "Metrics to track"})


@dataclass
class TrainingConfig:
    """Training configuration (organizational)."""
    # Core training params (direct)
    batch_size: int = field(default=32, metadata={"help": "Number of samples per batch"})
    epochs: int = field(default=100, metadata={"help": "Number of training epochs"})
    num_workers: int = field(default=4, metadata={"help": "Number of workers for data loading"})
    seed: int = field(default=42, metadata={"help": "Random seed for reproducibility"})
    resume_training: bool = field(default=False, metadata={"help": "Whether to resume training from checkpoint"})
    
    # Other configs
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig, metadata={"help": "Optimizer configuration"})
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig, metadata={"help": "Scheduler configuration"})
    checkpoints: CheckpointConfig = field(default_factory=CheckpointConfig, metadata={"help": "Checkpoint configuration"})
    learning_stat: LearningStatConfig = field(default_factory=LearningStatConfig, metadata={"help": "Learning statistics configuration"})
    
    # Lightning Module parameters
    lightning_module_args: Dict[str, Any] = field(
        default_factory=dict, metadata={"help": "Arguments for the Lightning Module"}
    )
    


########################################################
# Model Config
########################################################


@dataclass
class ModelConfig:
    """Model architecture parameters (atomic)."""
    name: str = field(default="mlp", metadata={"help": "Model Name: (e.g. 'deepsurv')"})
    args: Dict[str, Any] = field(
        default_factory=dict, metadata={"help": "Arguments for the model"}
    )
    dim: int = field(default=512, metadata={"help": "Dimension of latent space"})

########################################################
# Dataset Config
########################################################
@dataclass
class DataConfig:
    """Data configuration (organizational)."""
    data_root: str = field(default="./data", metadata={"help": "Root path to data directory"})
    datamodule_type: str = field(default="UnifiedHNC_ChronoSurv", metadata={"help": "Type of datamodule: 'UnifiedHNC' or 'UnifiedHNC_ChronoSurv'"})
    data_fraction: float = field(default=1.0, metadata={"help": "Fraction of data to use"})
    
    #Model paths
    path_lm: str = field(default="./data/models/Bio_ClinicalBERT", metadata={"help": "Path to BioClinical BERT model"})
    path_optimus: str = field(default="./data/models/H-optimus-1", metadata={"help": "Path to H-Optimus-1 model"})
    path_kimianet: str = field(default="./data/models/kimia_net.pth", metadata={"help": "Path to KimiaNet model"})
    
    #Tokenization
    max_tokens_history: int = field(default=512, metadata={"help": "Max tokens for history text"})
    max_tokens_surgery: int = field(default=512, metadata={"help": "Max tokens for surgery description"})
    max_tokens_report: int = field(default=512, metadata={"help": "Max tokens for report text"})
    padding: str = field(default="max_length", metadata={"help": "Tokenizer padding strategy"})
    truncation: bool = field(default=True, metadata={"help": "Enable tokenizer truncation"})

    #Number of branches
    num_branches: int=field(default=9, metadata={"help": "Number of input modalities"})
    
    #Time discretization for logistic hazard
    num_time_bins: Optional[int] = field(default=None, metadata={"help": "Number of discrete time bins for logistic hazard. If None, uses T_max with step=1"})
    
    #Image aggregation
    aggregate_images: bool = field(default=True, metadata={"help": "If True, average images"})
    
    #Folds
    k: int = field(default=5, metadata={"help": "Number of folds"})
    fold: int = field(default=1, metadata={"help": "Fold number"})


########################################################
# Config
########################################################

@dataclass
class Config:
    """Global configuration object grouping all sub-configs."""
    mode_debug: bool = field(default=False, metadata={"help": "Enable debug mode"})
    gpu_index: int = field(default=-1, metadata={"help": "GPU index to use for training: -1 for CPU, 0 for first GPU, etc."})
    training: TrainingConfig = field(default_factory=TrainingConfig, metadata={"help": "Training hyperparameters"})
    model: ModelConfig = field(default_factory=ModelConfig, metadata={"help": "Model architecture parameters"})
    data: DataConfig = field(default_factory=DataConfig, metadata={"help": "Data configuration"})

    def __post_init__(self):
        """Call __post_init__ on all sub-configs that have one."""
        # Update Checkpoint path with fold number and date
        if self.training.checkpoints.checkpoint_path is None:
            date_subfolder = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.training.checkpoints.checkpoint_path = os.path.join(
                self.training.checkpoints.checkpoint_dir,
                f"CV_{self.data.k}",
                f"fold_{self.data.fold}", 
                date_subfolder)

    def to_dict(self) -> Dict[str, Any]:
        """Convert Config object to dictionary for serialization."""
        def _serialize_dataclass(obj):
            if hasattr(obj, '__dict__'):
                result = {}
                for key, value in obj.__dict__.items():
                    if hasattr(value, '__dict__'):
                        result[key] = _serialize_dataclass(value)
                    else:
                        result[key] = value
                return result
            return obj
        
        return _serialize_dataclass(self)

    def save_to_yaml(self, filepath: str, **kwargs) -> None:
        """
        Save configuration to YAML file with optional additional data.
        
        Args:
            filepath: Path to save the YAML file
            **kwargs: Additional data to include in the YAML file (e.g., processing_artifacts, metadata, etc.)
        """
        config_dict = self.to_dict()
        # Add any additional data provided via kwargs
        config_dict.update(kwargs)
        
        # Save to YAML file
        with open(filepath, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        print(f"Configuration saved to {filepath}")

# Loader YAML -> Config

def load_config(yaml_path: str) -> Config:
    """
    Load config from YAML file and return a Config object.
    
    Args:
        yaml_path: Path to the YAML config file
        file_handler: FileSystemHandler instance (auto-detected if None)
        
    Returns:
        Config object
    """
    # Open the YAML file
    with open(yaml_path, 'r') as file:
        raw = yaml.safe_load(file)
    
    ########################################################
    #       Training Config
    ########################################################
    training_raw = raw.get("training", {})
    
    # Build atomic configs
    atomic_configs = {
        "optimizer": OptimizerConfig(**training_raw.get("optimizer", {})),
        "scheduler": SchedulerConfig(**training_raw.get("scheduler", {})),
        "checkpoints": CheckpointConfig(**training_raw.get("checkpoints", {})),
        "learning_stat": LearningStatConfig(**training_raw.get("learning_stat", {})),
    }
    
    # Filter out atomic config keys from training params
    training_params = {k: v for k, v in training_raw.items() if k not in atomic_configs}
    
    training_config = TrainingConfig(**training_params, **atomic_configs)
    
    ########################################################
    #       Data Config
    ########################################################
    data_raw = raw.get("data", {})
    data_config = DataConfig(**data_raw)

    ########################################################
    #       Model Config
    ########################################################
    model_raw = raw.get("model", {})        
    model_config = ModelConfig(**model_raw)
    
    return Config(
        mode_debug=raw.get("mode_debug", False),
        gpu_index=raw.get("gpu_index", -1),
        training=training_config,
        model=model_config,
        data=data_config,
    ) 