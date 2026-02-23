"""
ChronoSurv - Survival Analysis Training and Evaluation Script

Usage Examples:

1. Training (default command):
   python main.py --config config/chrono_surv.yaml
   python main.py train --config config/chrono_surv.yaml --debug
   python main.py train --config config/chrono_surv.yaml --k 2
   python main.py train --config config/chrono_surv.yaml --gpu_index 1

2. Evaluation:
   python main.py eval --checkpoint-dirs results/chrono_surv/CV_5/fold_1/2025-01-01_00-00-00

3. K-Folds builder:
    python main.py folds --dataset hancock --data_root ./data/HANCOCK --random_seed 42 --n_folds 5
    python main.py folds --dataset tcga --data_root ./data/TCGA-HNSC --random_seed 42 --n_folds 5
"""

import argparse
import os
import sys
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import random
import shutil
import torch
import lightning as L
import warnings
import pandas as pd

warnings.filterwarnings("ignore", ".*GPU available but not used.*")
warnings.filterwarnings("ignore", ".*tensorboardX.*has been removed.*")

sys.path.append(str(Path(__file__).parent / "src"))

from src.utils.folds import build_folds
from src.utils.config import load_config, Config
from src.data.data_factory import DataFactory
from src.model.model_factory import ModelFactory
from src.training.trainer_factory import TrainerFactory
from src.eval.survival_eval import SurvivalEval


def parse_args():
    """Parse command line arguments with subcommands."""
    if len(sys.argv) > 1 and sys.argv[1] not in ['train', 'eval', 'folds', '-h', '--help']:
        sys.argv.insert(1, 'train')
    
    parser = argparse.ArgumentParser(description="ChronoSurv - Survival Analysis Training and Evaluation")
    subparsers = parser.add_subparsers(dest="command", help="Command to run", required=True)
    
    # Train command (default)
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("--config", type=str, default="config/chrono_surv.yaml", help="Path to the YAML config file")
    train_parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    train_parser.add_argument("--resume_training", action="store_true", help="Resume training from checkpoint")
    train_parser.add_argument("--experiment_name", type=str, default=None, help="Experiment name for logging")
    train_parser.add_argument("--k", type=int, default=None, help="Fold number for CV (overrides config)")
    train_parser.add_argument("--gpu_index", type=int, default=None, help="GPU index to use (overrides config)")
    train_parser.add_argument("--seed", type=int, default=None, help="Random seed (overrides config)")
    train_parser.add_argument("--checkpoint_dir", type=str, default=None, help="Checkpoint directory (overrides config)")
    
    # Eval command
    eval_parser = subparsers.add_parser("eval", help="Evaluate trained model(s)")
    eval_parser.add_argument("--checkpoint-dirs", nargs="+", required=True, help="List of checkpoint directories to evaluate")
    eval_parser.add_argument("--stage", type=str, default="test", choices=["train", "val", "test"], help="Stage to evaluate on")
    
    # K-fold builder
    build_folds_parser = subparsers.add_parser("folds", help="Build K-folds for Cross-Validation")
    build_folds_parser.add_argument("--dataset", type=str, default="hancock", choices=["hancock", "tcga"], help="Dataset name (hancock or tcga)")
    build_folds_parser.add_argument("--data_root", type=str, default="./data/HANCOCK", help="Path of data")
    build_folds_parser.add_argument("--random_seed", type=int, default=42, help="Random seed to make folds")
    build_folds_parser.add_argument("--n_folds", type=int, default=5, help="Number of folds")
    build_folds_parser.add_argument("--train_ratio", type=float, default=0.7, help="Ratio of train set from non-test data")
    build_folds_parser.add_argument("--val_ratio", type=float, default=0.15, help="Ratio of validation set from non-test data")

    return parser.parse_args()


def override_config(config: Config, args: argparse.Namespace) -> Config:
    """Override config from command line arguments."""
    if hasattr(args, 'k') and args.k is not None:
        config.data.fold = args.k
        print(f"Overriding fold number from config: fold = {args.k}")
        config.training.checkpoints.checkpoint_path = None
    if hasattr(args, 'gpu_index') and args.gpu_index is not None:
        config.gpu_index = args.gpu_index
        print(f"Overriding GPU index from config: gpu_index = {args.gpu_index}")
    if hasattr(args, 'seed') and args.seed is not None:
        config.training.seed = args.seed
        print(f"Overriding seed from config: seed = {args.seed}")
    if hasattr(args, 'checkpoint_dir') and args.checkpoint_dir is not None:
        config.training.checkpoints.checkpoint_dir = args.checkpoint_dir
        config.training.checkpoints.checkpoint_path = None
        print(f"Overriding checkpoint_dir from config: {args.checkpoint_dir}")
    if args.debug:
        config.mode_debug = True
    if config.mode_debug:
        print("\nDEBUG MODE ENABLED: Overriding config for fast development runs.\n")
        config.training.epochs = 2
        config.training.batch_size = 4
        config.training.checkpoints.checkpoint_dir = "results/debug"
        config.data.data_fraction = 0.1
    config.__post_init__()
    return config


def _get_source_prefix_map(config: Config) -> dict:
    """
    Return per-source patient_id prefix map for multi-cohort evaluation.
    
    For Unified HNC datasets, returns {"hancock": "<numeric>", "tcga": "TCGA-"}
    so that evaluation can be split by source.
    For single-cohort datasets, returns None (no splitting).
    """
    if config.data.datamodule_type.startswith("UnifiedHNC"):
        return {"hancock": None, "tcga": "TCGA-"}
    return None


def _run_eval_for_checkpoint(
    evaluator: SurvivalEval,
    predictions: pd.DataFrame,
    eval_dir: str,
    stage: str = "test",
    source_prefix_map: dict = None,
) -> dict:
    """
    Run evaluation on a set of predictions and save results.
    
    If source_prefix_map is set, computes per-source metrics as well.
    """
    if source_prefix_map:
        results = evaluator.evaluate_all(predictions=predictions, stage=stage)
        evaluator.save_all(results, eval_dir=eval_dir)
        return results.get("all", {})
    else:
        metrics = evaluator.evaluate(predictions=predictions, stage=stage)
        evaluator.save(
            output_path=os.path.join(eval_dir, f"metrics_{stage}.json")
        )
        return metrics


def _print_metrics(metrics: dict, title: str = "Metrics"):
    """Print metrics in a formatted table."""
    print(f"\n{'-'*60}\n{title}\n{'-'*60}")
    for metric_name, metric_value in metrics.items():
        if metric_value is None:
            print(f"  {metric_name:15s}: None")
        elif isinstance(metric_value, float):
            print(f"  {metric_name:15s}: {metric_value:.4f}")
        else:
            print(f"  {metric_name:15s}: {metric_value}")
    print("-" * 60)


def train_command(args):
    """Execute training command."""
    print("Starting ChronoSurv Training")
    
    #########################################################
    # 1. Prepare all elements
    #########################################################
    
    print("=" * 60)
    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)
    config = override_config(config, args)
    os.makedirs(config.training.checkpoints.checkpoint_path, exist_ok=True)
    config.save_to_yaml(os.path.join(config.training.checkpoints.checkpoint_path, "config.yaml"))
    
    random.seed(config.training.seed)
    L.seed_everything(config.training.seed, workers=True)
    print(f"Random seed set to: {config.training.seed}")
    
    print("Creating data module...")
    datamodule = DataFactory.create_datamodule(
        data_config=config.data,
        training_config=config.training
    )
    print("Setup data module...")
    datamodule.setup()

    print("Creating model...")
    model = ModelFactory.create_model(
        model_config=config.model,
        data_config=config.data,
        input_dims=datamodule.get_input_dims(),
        t_max=datamodule.get_tmax(),
    )
    print(model)
    
    print("Creating trainer and module...")
    trainer, lightning_module = TrainerFactory.create_trainer_and_module(
        config=config,
        model=model,
    )
    
    source_prefix_map = _get_source_prefix_map(config)
    ckpt_path_base = config.training.checkpoints.checkpoint_path
    
    #########################################################
    # 2. Training and predictions
    #########################################################    
    print("\nStarting training...")
    print(f"Checkpoint path: {ckpt_path_base}")
    ckpt_path = os.path.join(ckpt_path_base, "last.ckpt")
    if config.training.resume_training and os.path.exists(ckpt_path):
        trainer.fit(lightning_module, datamodule, ckpt_path=ckpt_path)
    else:
        trainer.fit(lightning_module, datamodule)
        
    print("\nTraining completed successfully!\n")
    
    # Predict on LAST checkpoint (current model state)
    print("\n" + "=" * 60)
    print("Running Predictions on LAST checkpoint...")
    print("=" * 60)
    trainer.predict(lightning_module, [datamodule.val_dataloader(), datamodule.test_dataloader()])
    
    pred_dir = os.path.join(ckpt_path_base, "prediction")
    pred_last_dir = os.path.join(ckpt_path_base, "prediction_last")
    os.makedirs(pred_last_dir, exist_ok=True)
    shutil.copy(os.path.join(pred_dir, "predictions_val.csv"), os.path.join(pred_last_dir, "predictions_val.csv"))
    shutil.copy(os.path.join(pred_dir, "predictions_test.csv"), os.path.join(pred_last_dir, "predictions_test.csv"))
    print("LAST predictions saved\n")
    
    # Predict on BEST checkpoint
    print("=" * 60)
    print("Running Predictions on BEST checkpoint...")
    print("=" * 60)
    best_state_dict_path = os.path.join(ckpt_path_base, "best_model_state_dict.pt")
    if os.path.exists(best_state_dict_path):
        print(f"Loading best model weights from: {best_state_dict_path}")
        best_state_dict = torch.load(best_state_dict_path, map_location=next(model.parameters()).device, weights_only=False)
        lightning_module.model.load_state_dict(best_state_dict)
        
        trainer.predict(lightning_module, [datamodule.val_dataloader(), datamodule.test_dataloader()])
        
        pred_best_dir = os.path.join(ckpt_path_base, "prediction_best")
        os.makedirs(pred_best_dir, exist_ok=True)
        shutil.copy(os.path.join(pred_dir, "predictions_val.csv"), os.path.join(pred_best_dir, "predictions_val.csv"))
        shutil.copy(os.path.join(pred_dir, "predictions_test.csv"), os.path.join(pred_best_dir, "predictions_test.csv"))
        print("BEST predictions saved\n")
    else:
        print(f"WARNING: best_model_state_dict.pt not found at {best_state_dict_path}")
        print("Skipping BEST predictions\n")
    
    #########################################################
    # 3. Evaluation on LAST and BEST
    #########################################################
    print("\n" + "=" * 60)
    print("Evaluating model performance...")
    print("=" * 60)
    
    # Evaluate LAST checkpoint
    print("\n--- Evaluating LAST checkpoint ---")
    pred_last_path = os.path.join(ckpt_path_base, "prediction_last", "predictions_test.csv")
    metrics_last = {}
    if os.path.exists(pred_last_path):
        predictions_last = pd.read_csv(pred_last_path)
        evaluator_last = SurvivalEval(
            datamodule=datamodule,
            checkpoint_dir=ckpt_path_base,
            tau=5 * 365,
            source_prefix_map=source_prefix_map,
        )
        eval_last_dir = os.path.join(ckpt_path_base, "eval_last")
        metrics_last = _run_eval_for_checkpoint(
            evaluator=evaluator_last,
            predictions=predictions_last,
            eval_dir=eval_last_dir,
            stage="test",
            source_prefix_map=source_prefix_map,
        )
        _print_metrics(metrics_last, title="LAST Checkpoint Metrics (all)")
    
    # Evaluate BEST checkpoint
    print("\n--- Evaluating BEST checkpoint ---")
    pred_best_path = os.path.join(ckpt_path_base, "prediction_best", "predictions_test.csv")
    metrics_best = {}
    if os.path.exists(pred_best_path):
        predictions_best = pd.read_csv(pred_best_path)
        evaluator_best = SurvivalEval(
            datamodule=datamodule,
            checkpoint_dir=ckpt_path_base,
            tau=5 * 365,
            source_prefix_map=source_prefix_map,
        )
        eval_best_dir = os.path.join(ckpt_path_base, "eval_best")
        metrics_best = _run_eval_for_checkpoint(
            evaluator=evaluator_best,
            predictions=predictions_best,
            eval_dir=eval_best_dir,
            stage="test",
            source_prefix_map=source_prefix_map,
        )
        _print_metrics(metrics_best, title="BEST Checkpoint Metrics (all)")
        
        # Comparison LAST vs BEST
        if metrics_last and metrics_best:
            print("\n" + "=" * 60 + "\nLAST vs BEST Comparison:\n" + "=" * 60)
            for metric_name in metrics_best.keys():
                val_last = metrics_last.get(metric_name)
                val_best = metrics_best.get(metric_name)
                if val_last is None or val_best is None:
                    print(f"  {metric_name:15s}: LAST={val_last}, BEST={val_best}")
                elif isinstance(val_last, (int, float)) and isinstance(val_best, (int, float)):
                    diff = val_best - val_last
                    symbol = "down" if diff < 0 else "up"
                    print(f"  {metric_name:15s}: LAST={val_last:.4f}, BEST={val_best:.4f}, Diff={diff:+.4f} {symbol}")
            print("=" * 60)
    
    print("\nTraining script completed!")


def eval_command(args):
    """
    Execute evaluation command on multiple checkpoint directories.
    
    For each checkpoint, evaluates both BEST and LAST predictions (if available).
    For Unified HNC datasets, per-source metrics are also computed.
    """
    print("Starting ChronoSurv Evaluation\n" + "=" * 60)
    print(f"Number of checkpoints to evaluate: {len(args.checkpoint_dirs)}")
    print(f"Stage: {args.stage}")
    print("=" * 60)
    
    all_results = {}
    for checkpoint_dir in args.checkpoint_dirs:
        print("\n" + "=" * 60 + f"\nEvaluating: {checkpoint_dir}\n" + "=" * 60)
        
        config_path = os.path.join(checkpoint_dir, "config.yaml")
        if not os.path.exists(config_path):
            print(
                f"No config file found at {config_path}. "
                f"Make sure checkpoint directory contains config.yaml"
            )
            continue
        
        print(f"\nLoading configuration from: {config_path}")
        config = load_config(config_path)
        
        print("Creating data module...")
        datamodule = DataFactory.create_datamodule(data_config=config.data, training_config=config.training)
        print("Setup data module...")
        datamodule.setup()
        
        source_prefix_map = _get_source_prefix_map(config)
        
        for ckpt_type in ["best", "last"]:
            pred_path = os.path.join(
                checkpoint_dir, f"prediction_{ckpt_type}", f"predictions_{args.stage}.csv"
            )
            
            if not os.path.exists(pred_path):
                if ckpt_type == "last":
                    pred_path_fallback = os.path.join(
                        checkpoint_dir, "prediction", f"predictions_{args.stage}.csv"
                    )
                    if os.path.exists(pred_path_fallback):
                        pred_path = pred_path_fallback
                    else:
                        print(f"  No {ckpt_type} predictions found. Skipping.")
                        continue
                else:
                    print(f"  No {ckpt_type} predictions found. Skipping.")
                    continue
            
            print(f"\n{'='*60}")
            print(f"  Evaluating {ckpt_type.upper()} checkpoint")
            print(f"  Predictions: {pred_path}")
            print(f"{'='*60}")
            
            predictions = pd.read_csv(pred_path)
            
            evaluator = SurvivalEval(
                datamodule=datamodule,
                checkpoint_dir=checkpoint_dir,
                tau=5 * 365,
                source_prefix_map=source_prefix_map,
            )
            
            eval_dir = os.path.join(checkpoint_dir, f"eval_{ckpt_type}")
            metrics = _run_eval_for_checkpoint(
                evaluator=evaluator,
                predictions=predictions,
                eval_dir=eval_dir,
                stage=args.stage,
                source_prefix_map=source_prefix_map,
            )
            
            result_key = f"{checkpoint_dir}_{ckpt_type}"
            all_results[result_key] = metrics
            _print_metrics(metrics, title=f"{ckpt_type.upper()} Metrics (all)")
    
    # Summary
    print("\n" + "=" * 60 + "\nEVALUATION SUMMARY\n" + "=" * 60)
    if len(all_results) == 0:
        print("No checkpoints were evaluated.")
    else:
        key_metrics = ["c_ipcw_5y", "ibs_5y", "td_auc", "km_calib"]
        print(f"\n{'Checkpoint':<70} | " + " | ".join([f"{m:>10s}" for m in key_metrics]))
        print("-" * (70 + 3 + (13 * len(key_metrics))))
        for result_key, metrics in all_results.items():
            values = " | ".join([
                f"{metrics.get(m, float('nan')):>10.4f}" if metrics.get(m) is not None else f"{'N/A':>10s}"
                for m in key_metrics
            ])
            print(f"{result_key:<70} | {values}")
        
        print("=" * 60)
    
    print("\nEvaluation script completed!")


def build_folds_command(args):
    """Build folds for K-folds Cross-Validation."""
    print("Starting K-folds Builder\n" + "=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Data root: {args.data_root}")
    print(f"Random seed: {args.random_seed}")
    print(f"Folds number: {args.n_folds}")
    print("=" * 60)

    build_folds(args)

    print("=" * 60)
    print("Build folds script completed!")


if __name__ == "__main__":
    args = parse_args()
    
    if args.command == "train":
        train_command(args)
    elif args.command == "eval":
        eval_command(args)
    elif args.command == "folds":
        build_folds_command(args)
    else:
        print(f"Unknown command: {args.command}")
        sys.exit(1)
