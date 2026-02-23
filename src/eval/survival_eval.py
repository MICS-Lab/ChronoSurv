import os
import json
import pandas as pd
import numpy as np
from scipy.stats import chi2 
from typing import Optional, Literal, Dict, List
from sksurv.util import Surv
from sksurv.nonparametric import SurvivalFunctionEstimator, CensoringDistributionEstimator
from sksurv.metrics import concordance_index_ipcw, cumulative_dynamic_auc, integrated_brier_score
from SurvivalEVAL.Evaluations.util import predict_median_st, predict_mean_st, predict_rmst
from SurvivalEVAL.Evaluations.MeanError import mean_error

import lightning as L
from src.data.utils import process_to_array
from src.eval.classification_eval import ClassificationEvaluator


class SurvivalEval:
    """
    Evaluation class for survival models.
    
    Computes standard survival analysis metrics:
    - C-IPCW: Concordance Index with Inverse Probability of Censoring Weighting
    - td-AUC: Time-dependent Area Under the Curve
    - IBS: Integrated Brier Score
    - KM Calibration: KL divergence between predicted and Kaplan-Meier survival curves
    - MAE: Mean Absolute Error (Margin, Hinge, Pseudo-observations, Uncensored methods)
    - MSE: Mean Squared Error (Margin, Hinge, Pseudo-observations, Uncensored methods)
    - RMSE: Root Mean Squared Error (Margin, Hinge, Pseudo-observations, Uncensored methods)
    
    Also computes binary classification metrics at a specific time horizon:
    - AUROC: Area Under ROC Curve
    - AUPRC: Area Under Precision-Recall Curve (mAP)
    - F1-score, Precision, Recall, Accuracy (using optimal threshold from validation set)
    
    Supports per-source evaluation for multi-cohort datasets (e.g. Unified HNC).
    When source_prefix_map is provided, computes metrics separately for each source
    subset in addition to the overall "all" metrics.
    
    Expected prediction format (CSV):
    ```
    patient_id,time,event,risk_score,S_1,S_2,...,S_{T_max}
    P001,365,1,38.83,1.0,0.999,...,0.489
    P002,730,0,20.88,1.0,1.0,...,0.663
    ```
    
    Required columns:
    - patient_id: Patient identifier
    - time: Observed time (ground truth)
    - event: Event indicator (1=event occurred, 0=censored)
    - risk_score: Predicted risk scores (higher = higher risk)
    - S_1, S_2, ..., S_{T_max}: Survival probabilities at times 1, 2, ..., T_max
    """
    
    # Default tau configurations for C-IPCW
    DEFAULT_FIXED_TAUS = {
        "3y": 3 * 365,
        "5y": 5 * 365,
    }
    DEFAULT_PERCENTILES = [80, 90, 95]
    
    def __init__(
        self,
        datamodule: L.LightningDataModule,
        checkpoint_dir: str,
        tau: int = 5 * 365,  # Legacy parameter, kept for compatibility
        fixed_taus: Optional[Dict[str, int]] = None,
        percentiles: Optional[List[int]] = None,
        classification_horizon: Optional[List[int]] = [3*365, 5*365],
        T_max_prediction: int = 3650,  # 10 * 365 days
        source_prefix_map: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize SurvivalEval and prepare train/val data.
        
        Args:
            datamodule: HANCOCK datamodule (used to extract train data for IPCW)
            checkpoint_dir: Path to checkpoint directory (for loading predictions and saving results)
            tau: Legacy truncation time for C-IPCW (default: 5 years). Kept for compatibility.
            fixed_taus: Dictionary of fixed tau values {name: days}, e.g., {"3y": 1095, "5y": 1825}.
                        If None, uses DEFAULT_FIXED_TAUS.
            percentiles: List of percentiles to compute from train durations, e.g., [80, 90, 95].
                         If None, uses DEFAULT_PERCENTILES.
            classification_horizon: Time horizon(s) for classification metrics in days.
                Can be an int (e.g., 365) or a list (e.g., [365, 1095, 1825]).
                If None, defaults to [1095, 1825] (3, 5 years)
            T_max_prediction: Maximum time for prediction filtering (default: 3650 days = 10 * 365 days)
            source_prefix_map: Optional mapping {source_name: patient_id_prefix} for per-source eval.
                Example: {"hancock": None, "tcga": "TCGA-"} will split predictions by patient_id 
                prefix and compute metrics separately for each source + overall.
                Use None as prefix to mean "all patients not matched by other prefixes".
                If None (the whole map), only computes overall metrics (backward compatible).
        
        Note:
            This method prepares train data (for IPCW) and validation data (for classification threshold).
        """
        self.datamodule = datamodule
        self.checkpoint_dir = checkpoint_dir
        self.tau = tau  # Legacy
        self.T_max_prediction = T_max_prediction
        self.classification_horizons = classification_horizon or [365, 3*365, 5*365]
        if isinstance(self.classification_horizons, int):
            self.classification_horizons = [self.classification_horizons]
        self.source_prefix_map = source_prefix_map
        
        # Store tau config for later (after _prepare_train_data)
        self._fixed_taus = fixed_taus
        self._percentiles = percentiles
        
        # Prepare train data (for IPCW)
        self._prepare_train_data()
        
        # Build multiple taus (fixed + percentiles based on train data)
        self.taus = self._build_taus(self._fixed_taus, self._percentiles)
        
        # Initialize classification evaluator and find best threshold from validation data
        self.classification_evaluator = ClassificationEvaluator(
            checkpoint_dir=self.checkpoint_dir,
            horizons=self.classification_horizons
        )
        self.classification_evaluator.find_val_best_threshold()
        
        # Results storage
        self.metrics = {}
        self.current_stage = None
    
    def _build_taus(
        self,
        fixed_taus: Optional[Dict[str, int]],
        percentiles: Optional[List[int]],
    ) -> Dict[str, int]:
        """
        Build dictionary of tau values from fixed values and percentiles of train durations.
        
        Args:
            fixed_taus: Dictionary of fixed tau values {name: days}
            percentiles: List of percentiles to compute from train durations
            
        Returns:
            Dictionary {tau_name: tau_days} sorted by tau value
        """
        taus = {}
        
        # Add fixed taus
        if fixed_taus is None:
            fixed_taus = self.DEFAULT_FIXED_TAUS
        taus.update(fixed_taus)
        
        # Add percentile-based taus from train durations
        if percentiles is None:
            percentiles = self.DEFAULT_PERCENTILES
        
        for p in percentiles:
            tau_value = int(np.percentile(self.time_train, p))
            taus[f"p{p}"] = tau_value
        
        # Sort by tau value
        taus = dict(sorted(taus.items(), key=lambda x: x[1]))
        
        return taus
    
    def evaluate(
        self, 
        predictions: Optional[pd.DataFrame] = None,
        stage: Literal["train", "val", "test"] = "test"
    ) -> Dict[str, float]:
        """
        Compute all survival metrics.
        
        Args:
            predictions: DataFrame with predictions (if None, load from checkpoint_dir/prediction/predictions_{stage}.csv)
            stage: Which stage to evaluate (used only if predictions is None)
        
        Returns:
            Dictionary with metrics:
                - c_ipcw_{tau}: Concordance index with IPCW at each tau (3y, 5y, p80, p90, p95)
                - td_auc: Mean time-dependent AUC
                - ibs: Integrated Brier Score (full grid, legacy)
                - ibs_{tau}: IBS at each tau horizon (3y, 5y, p80, p90, p95)
                - km_calib: KM calibration (KL divergence)
                - mae_{method}_{pred}: MAE with different methods and predictions
                - mse_{method}_{pred}: MSE with different methods and predictions
                - rmse_{method}_{pred}: RMSE with different methods and predictions
                - Classification metrics at each horizon
        """
        print(f"\n{'='*60}\nSurvivalEval - Computing Metrics\n{'='*60}")
        
        # Store current stage
        self.current_stage = stage
        
        # Load predictions
        if predictions is None:
            print(f"Loading predictions from checkpoint_dir (stage={stage})...")
            predictions = self._load_predictions(stage)
        else:
            print("Using provided predictions DataFrame...")
            print(f"Stage: {stage}")
        
        print(f"Predictions shape: {predictions.shape}")
        
        # Validate predictions format
        self._validate_predictions(predictions)
        
        # Print tau horizons
        print(f"\nEvaluation horizons (C-IPCW):")
        for tau_name, tau_days in self.taus.items():
            print(f"  {tau_name}: {tau_days} days ({tau_days/365:.1f}y)")
        
        # Compute C-IPCW at multiple taus
        print("\nComputing C-IPCW at multiple horizons...")
        self._compute_c_ipcw_multi(predictions)
        
        # Compute td-AUC
        print("\nComputing td-AUC...")
        self._compute_td_auc(predictions)
        if self.metrics.get('td_auc') is not None:
            print(f"  td-AUC: {self.metrics['td_auc']:.4f}")
        else:
            print("  td-AUC: None (computation failed)")
        
        # Compute IBS (full grid, legacy)
        print("\nComputing IBS (full grid)...")
        self._compute_ibs(predictions)
        if self.metrics.get('ibs') is not None:
            print(f"  IBS: {self.metrics['ibs']:.4f}")
        else:
            print("  IBS: None (computation failed)")
        
        # Compute IBS at multiple tau horizons
        print("\nComputing IBS at multiple horizons...")
        self._compute_ibs_multi(predictions)
        
        # Compute KM Calibration
        print("\nComputing KM Calibration...")
        self._compute_km_calibration(predictions)
        if self.metrics.get('km_calib') is not None:
            print(f"  KM Calibration: {self.metrics['km_calib']:.4f}")
        else:
            print("  KM Calibration: None (computation failed)")
        
        # Compute D-Calibration
        print("\nComputing D-Calibration...")
        self._compute_d_calibration(predictions)
        if self.metrics.get('d_calib_chi2') is not None:
            print(f"  D-Calibration chi2: {self.metrics['d_calib_chi2']:.4f}")
            print(f"  D-Calibration p-value: {self.metrics['d_calib_pvalue']:.4f}")
        else:
            print("  D-Calibration: None (computation failed)")
        
        # Compute MAE for all combinations of methods and prediction types
        print("\nComputing MAE metrics...")
        mae_methods = ["Margin", "Hinge", "Pseudo_obs", "Uncensored"]
        prediction_methods = ["median", "rmst"]
        
        for mae_method in mae_methods:
            for pred_method in prediction_methods:
                metric_key = f"mae_{mae_method.lower()}_{pred_method}"
                self._compute_mae(predictions, method=mae_method, prediction_method=pred_method)
                value = self.metrics.get(metric_key)
                if value is not None:
                    print(f"  {metric_key}: {value:.2f}")
                else:
                    print(f"  {metric_key}: None (computation failed)")
        
        # Compute MSE and RMSE for all combinations of methods and prediction types
        print("\nComputing MSE and RMSE metrics...")
        for mse_method in mae_methods:
            for pred_method in prediction_methods:
                mse_key = f"mse_{mse_method.lower()}_{pred_method}"
                rmse_key = f"rmse_{mse_method.lower()}_{pred_method}"
                self._compute_mae(predictions, method=mse_method, prediction_method=pred_method, error_type="squared")
                mse_value = self.metrics.get(mse_key)
                if mse_value is not None:
                    self.metrics[rmse_key] = np.sqrt(mse_value)
                    print(f"  {mse_key}: {mse_value:.2f}")
                    print(f"  {rmse_key}: {self.metrics[rmse_key]:.2f}")
                else:
                    self.metrics[rmse_key] = None
                    print(f"  {mse_key}: None (computation failed)")
        
        # Compute classification metrics at horizon
        print("\nComputing classification metrics...")
        try:
            classif_metrics = self.classification_evaluator.evaluate(predictions)
            self.metrics.update(classif_metrics)
        except Exception as e:
            print(f"  Warning: Classification metrics failed: {e}")
        
        print(f"\n{'='*60}\n")
        
        return self.metrics
    
    def evaluate_all(
        self,
        predictions: Optional[pd.DataFrame] = None,
        stage: Literal["train", "val", "test"] = "test",
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate on all patients + per-source subsets.
        
        If source_prefix_map was provided at init, computes metrics for each source
        in addition to the overall "all" metrics.
        
        Args:
            predictions: DataFrame with predictions (if None, load from checkpoint_dir)
            stage: Which stage to evaluate
            
        Returns:
            Dictionary {subset_name: metrics_dict}. Always contains "all".
            If source_prefix_map is set, also contains one entry per source.
            Example: {"all": {...}, "hancock": {...}, "tcga": {...}}
        """
        # Load predictions once
        if predictions is None:
            predictions = self._load_predictions(stage)
        self.current_stage = stage
        
        results = {}
        
        # 1. Evaluate on ALL patients
        print(f"\n{'#'*60}")
        print(f"# Evaluating ALL patients (n={len(predictions)})")
        print(f"{'#'*60}")
        self.metrics = {}
        results["all"] = self.evaluate(predictions=predictions, stage=stage)
        
        # 2. Evaluate per source (if configured)
        if self.source_prefix_map:
            # Collect all explicit prefixes (non-None) to identify "remainder" sources
            explicit_prefixes = {
                name: pfx for name, pfx in self.source_prefix_map.items() 
                if pfx is not None
            }
            
            for source_name, prefix in self.source_prefix_map.items():
                patient_ids = predictions["patient_id"].astype(str)
                
                if prefix is not None:
                    # Match by prefix
                    source_preds = predictions[patient_ids.str.startswith(prefix)].copy()
                else:
                    # None prefix = "everything not matched by any other explicit prefix"
                    matched_mask = pd.Series(False, index=predictions.index)
                    for pfx in explicit_prefixes.values():
                        matched_mask |= patient_ids.str.startswith(pfx)
                    source_preds = predictions[~matched_mask].copy()
                
                if len(source_preds) == 0:
                    print(f"\n  WARNING: No patients found for source '{source_name}' "
                          f"(prefix='{prefix}'). Skipping.")
                    continue
                
                n_events = int(source_preds["event"].sum())
                print(f"\n{'#'*60}")
                print(f"# Evaluating {source_name.upper()} patients "
                      f"(n={len(source_preds)}, events={n_events})")
                print(f"{'#'*60}")
                
                # Reset metrics for this subset
                self.metrics = {}
                results[source_name] = self.evaluate(
                    predictions=source_preds, stage=stage
                )
        
        return results
    
    def save(self, output_path: Optional[str] = None):
        """
        Save metrics to JSON file.
        
        Args:
            output_path: Path to save JSON file (if None, saves to checkpoint_dir/eval/metrics_{stage}.json)
        """
        if output_path is None:
            stage_suffix = f"_{self.current_stage}" if self.current_stage else ""
            output_path = os.path.join(self.checkpoint_dir, "eval", f"metrics{stage_suffix}.json")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        print(f"Metrics saved to: {output_path}")
    
    def save_all(
        self,
        results: Dict[str, Dict[str, float]],
        eval_dir: str,
    ) -> None:
        """
        Save per-source evaluation results to separate JSON files.
        
        Creates one file per subset:
            eval_dir/metrics_test.json          (all patients)
            eval_dir/metrics_test_hancock.json   (hancock only)
            eval_dir/metrics_test_tcga.json      (tcga only)
        
        Args:
            results: Output of evaluate_all(), dict {subset_name: metrics}
            eval_dir: Directory to save metrics (e.g. checkpoint_dir/eval_best/)
        """
        os.makedirs(eval_dir, exist_ok=True)
        stage_suffix = f"_{self.current_stage}" if self.current_stage else ""
        
        for subset_name, metrics in results.items():
            if subset_name == "all":
                filename = f"metrics{stage_suffix}.json"
            else:
                filename = f"metrics{stage_suffix}_{subset_name}.json"
            
            output_path = os.path.join(eval_dir, filename)
            with open(output_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            print(f"  Saved {subset_name} metrics to: {output_path}")
    
    def _prepare_train_data(self):
        """Extract y_train from datamodule using process_to_array and compute IPCW safe time."""
        _, time_train, event_train, _ = process_to_array(self.datamodule, stage="train")
        self.y_train = Surv.from_arrays(event=event_train, time=time_train)
        self.time_train = time_train
        self.event_train = event_train
        self.t_max_train = float(np.max(time_train))  # Max time for RMST
        
        # Compute the max time where the training censoring KM G(t) > 0.
        self.ipcw_max_time = self._compute_ipcw_max_time()
    
    def _compute_ipcw_max_time(self) -> float:
        """
        Compute the maximum time supported by the training censoring distribution.
        
        The IPCW weight is 1/G(t) where G(t) is the censoring survival function
        estimated from training data. 
        
        Returns:
            Maximum time where G(t) > 0. Test observations beyond this time
            must be filtered out before computing IPCW-based metrics.
        """
        cens = CensoringDistributionEstimator()
        cens.fit(self.y_train)
        
        # Evaluate G(t) at all unique training times
        unique_times = np.unique(self.time_train)
        G_values = cens.predict_proba(unique_times)
        
        # Find the last time where G(t) > 0
        positive_mask = G_values > 0
        if positive_mask.any():
            ipcw_max = float(unique_times[positive_mask][-1])
        else:
            # Fallback: all G(t) = 0, use max training time
            ipcw_max = float(np.max(self.time_train))
        
        print(f"  IPCW max safe time (train censoring support): {ipcw_max:.0f} days ({ipcw_max/365:.1f}y)")
        return ipcw_max
    
    def _clip_ipcw_safe(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        beyond_mask = df["time"].values > self.ipcw_max_time
        n_clipped = beyond_mask.sum()
        if n_clipped > 0:
            n_events_clipped = int(df.loc[beyond_mask, "event"].sum())
            print(f"  IPCW admin censoring: clipped {n_clipped} observations beyond t={self.ipcw_max_time:.0f}d "
                  f"({n_events_clipped} events, {n_clipped - n_events_clipped} censored) -> censored at t={self.ipcw_max_time:.0f}d")
            df.loc[beyond_mask, "time"] = self.ipcw_max_time
            df.loc[beyond_mask, "event"] = 0
        return df
    
    def _load_predictions(self, stage: str) -> pd.DataFrame:
        """
        Load predictions CSV from checkpoint_dir/prediction/predictions_{stage}.csv.
        Filter survival columns S_t to keep only those with t <= T_max_prediction.
        """
        pred_path = os.path.join(
            self.checkpoint_dir, 
            "prediction", 
            f"predictions_{stage}.csv"
        )
        
        if not os.path.exists(pred_path):
            raise FileNotFoundError(f"Predictions file not found: {pred_path}")
        
        df = pd.read_csv(pred_path)
        
        # Filter survival columns by T_max_prediction
        survival_cols = [c for c in df.columns if c.startswith("S_")]
        cols_to_keep = []
        cols_to_drop = []
        
        for col in survival_cols:
            time_point = int(col.split("_")[1])
            if time_point <= self.T_max_prediction:
                cols_to_keep.append(col)
            else:
                cols_to_drop.append(col)
        
        if cols_to_drop:
            print(f"  Filtering survival columns: keeping {len(cols_to_keep)} columns (t <= {self.T_max_prediction}), dropping {len(cols_to_drop)} columns")
            df = df.drop(columns=cols_to_drop)
        
        return df
    
    def _validate_predictions(self, df: pd.DataFrame):
        """Validate that predictions DataFrame has required columns and covers T_max_prediction."""
        required_cols = ["patient_id", "time", "event", "risk_score"]
        survival_cols = [c for c in df.columns if c.startswith("S_")]
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        if len(survival_cols) == 0:
            raise ValueError("No survival columns (S_1, S_2, ...) found in predictions")
        
        # Check that predictions cover at least T_max_prediction
        max_time_predicted = max([int(c.split("_")[1]) for c in survival_cols])
        if max_time_predicted < self.T_max_prediction:
            raise ValueError(
                f"Predictions do not cover T_max_prediction: "
                f"max predicted time = {max_time_predicted} days, "
                f"T_max_prediction = {self.T_max_prediction} days. "
                f"Predictions must cover at least T_max_prediction."
            )
        
        print(f"  Validated: {len(required_cols)} required columns + {len(survival_cols)} survival columns")
        print(f"  Max predicted time: {max_time_predicted} days (T_max_prediction: {self.T_max_prediction} days)")
    
    def _compute_c_ipcw_multi(self, df: pd.DataFrame) -> None:
        """
        Compute Concordance Index with IPCW at multiple tau horizons.
        
        Measures discrimination: probability that model correctly ranks pairs of patients.
        Higher is better (range: 0.5-1.0, 0.5=random, 1.0=perfect).
        
        Computes c_ipcw_{tau_name} for each tau in self.taus.
        """
        y_test = Surv.from_arrays(event=df["event"].values, time=df["time"].values)
        risk_scores = df["risk_score"].values
        
        for tau_name, tau_days in self.taus.items():
            metric_key = f"c_ipcw_{tau_name}"
            try:
                c_index_ipcw = concordance_index_ipcw(
                    survival_train=self.y_train,
                    survival_test=y_test,
                    estimate=risk_scores,
                    tau=tau_days
                )
                self.metrics[metric_key] = float(c_index_ipcw[0])
                print(f"  c_ipcw_{tau_name} (tau={tau_days}d): {self.metrics[metric_key]:.4f}")
            except Exception as e:
                self.metrics[metric_key] = None
                print(f"  c_ipcw_{tau_name} (tau={tau_days}d): None (failed: {e})")
    
    def _compute_td_auc(self, df: pd.DataFrame) -> Optional[float]:
        """
        Compute time-dependent AUC (mean over time grid).
        
        Measures discrimination at different time points.
        Higher is better (range: 0.5-1.0).
        
        """
        try:
            df_safe = self._clip_ipcw_safe(df)
            
            if len(df_safe) < 10:
                print(f"  Warning: too few samples after IPCW filtering ({len(df_safe)}), skipping td-AUC")
                self.metrics["td_auc"] = None
                return None
            
            y_train_times = np.array([x[1] for x in self.y_train])
            
            # Upper bound for time grid: use p95 of train times, capped by IPCW max
            max_time = min(np.percentile(y_train_times, 95), self.ipcw_max_time)
            
            # Ensure we have a valid time range
            if max_time <= 365:
                print(f"  Warning: max_time ({max_time:.0f}) too small for td-AUC, skipping")
                self.metrics["td_auc"] = None
                return None
            
            times = np.linspace(365, max_time, num=100)
            
            y_test = Surv.from_arrays(event=df_safe["event"].values, time=df_safe["time"].values)
            risk_scores = df_safe["risk_score"].values
            
            _, cd_auc = cumulative_dynamic_auc(
                survival_train=self.y_train,
                survival_test=y_test,
                estimate=risk_scores,
                times=times
            )
            
            self.metrics["td_auc"] = float(np.mean(cd_auc))
            return float(np.mean(cd_auc))
            
        except Exception as e:
            print(f"  Warning: td-AUC computation failed: {e}")
            self.metrics["td_auc"] = None
            return None
    
    def _compute_ibs(self, df: pd.DataFrame) -> Optional[float]:
        """
        Compute Integrated Brier Score (full grid, legacy).
        
        Measures calibration: squared difference between predicted and actual survival.
        Lower is better (range: 0-1, 0=perfect).
        """
        try:
            y_test = Surv.from_arrays(event=df["event"].values, time=df["time"].values)
            
            # Extract survival columns and times
            survival_cols = sorted(
                [c for c in df.columns if c.startswith("S_")],
                key=lambda x: int(x.split("_")[1])
            )
            all_times = [int(c.split("_")[1]) for c in survival_cols]
            
            # Filter times to be within test data range
            y_test_times = df["time"].values
            min_time, max_time = y_test_times.min(), y_test_times.max()
            
            valid_cols = [c for c, t in zip(survival_cols, all_times) if min_time <= t < max_time]
            valid_times = [t for t in all_times if min_time <= t < max_time]
            
            if len(valid_times) == 0:
                print("  Warning: No valid time points found for IBS computation")
                self.metrics["ibs"] = None
                return None
            
            survs = df[valid_cols].values
            
            ibs = integrated_brier_score(
                survival_train=self.y_train,
                survival_test=y_test,
                estimate=survs,
                times=valid_times
            )
            
            self.metrics["ibs"] = float(ibs)
            return float(ibs)
            
        except Exception as e:
            print(f"  Warning: IBS computation failed: {e}")
            self.metrics["ibs"] = None
            return None

    def _compute_ibs_multi(self, df: pd.DataFrame) -> None:
        """
        Compute Integrated Brier Score at multiple tau horizons (same taus as C-IPCW).

        For each tau, the IBS evaluation grid is capped at [min_test_time, tau).
        This avoids instability from late-time IPCW weights and gives clinically
        relevant calibration metrics at specific horizons.

        Stores metrics as ibs_{tau_name} (e.g. ibs_3y, ibs_5y, ibs_p80, ibs_p90).
        """
        # Pre-extract survival columns (shared across taus)
        survival_cols = sorted(
            [c for c in df.columns if c.startswith("S_")],
            key=lambda x: int(x.split("_")[1])
        )
        all_times = np.array([int(c.split("_")[1]) for c in survival_cols])
        y_test = Surv.from_arrays(event=df["event"].values, time=df["time"].values)
        y_test_times = df["time"].values
        min_time = max(float(y_test_times.min()), 1.0)

        for tau_name, tau_days in self.taus.items():
            metric_key = f"ibs_{tau_name}"
            try:
                # Grid: [min_test_time, tau)
                mask = (all_times >= min_time) & (all_times < tau_days)
                valid_times = all_times[mask].tolist()
                valid_cols = [survival_cols[i] for i in range(len(survival_cols)) if mask[i]]

                if len(valid_times) == 0:
                    self.metrics[metric_key] = None
                    print(f"  ibs_{tau_name} (tau={tau_days}d): None (no valid time points)")
                    continue

                survs = df[valid_cols].values

                ibs_val = integrated_brier_score(
                    survival_train=self.y_train,
                    survival_test=y_test,
                    estimate=survs,
                    times=valid_times,
                )
                self.metrics[metric_key] = float(ibs_val)
                print(f"  ibs_{tau_name} (tau={tau_days}d): {ibs_val:.4f}")

            except Exception as e:
                self.metrics[metric_key] = None
                print(f"  ibs_{tau_name} (tau={tau_days}d): None (failed: {e})")
    
    def _compute_mae(
        self, 
        df: pd.DataFrame, 
        method: Literal["Margin", "Hinge", "Pseudo_obs", "Uncensored"] = "Margin",
        prediction_method: Literal["median", "rmst"] = "median",
        error_type: Literal["absolute", "squared"] = "absolute"
    ) -> Optional[float]:
        """
        Compute Mean Absolute Error (MAE) or Mean Squared Error (MSE) using SurvivalEVAL.
        
        This metric measures the difference between predicted and observed survival times,
        accounting for censoring using various methods.
        
        Args:
            df: DataFrame with predictions
            method: Method for handling censoring:
                - "Margin": Margin-based method (recommended)
                - "Hinge": Hinge-based method
                - "Pseudo_obs": Pseudo-observations method
                - "Uncensored": Only on uncensored patients (baseline)
            prediction_method: How to derive predicted time from survival curve:
                - "median": Use median survival time (default)
                - "rmst": Use restricted mean survival time
            error_type: Type of error to compute:
                - "absolute": Mean Absolute Error (MAE)
                - "squared": Mean Squared Error (MSE)
        
        Returns:
            Error value (lower is better, MAE in days, MSE in days²), or None if failed.
        
        Reference: https://github.com/shi-ang/SurvivalEVAL
        """
        error_prefix = "mae" if error_type == "absolute" else "mse"
        metric_key = f"{error_prefix}_{method.lower()}_{prediction_method}"
        
        try:
            # Extract survival columns and times
            survival_cols = sorted(
                [c for c in df.columns if c.startswith("S_")],
                key=lambda x: int(x.split("_")[1])
            )
            times_coordinates = np.array([int(c.split("_")[1]) for c in survival_cols])
            
            # Extract survival curves [n_samples, n_times]
            survival_curves = df[survival_cols].values
            
            # Predict survival times using specified method
            if prediction_method == "median":
                predicted_times = predict_median_st(survival_curves, times_coordinates)
            elif prediction_method == "rmst":
                # For RMST, use max time from training set
                predicted_times = predict_rmst(survival_curves, times_coordinates, interpolation="None")
            else:
                raise ValueError(f"Unknown prediction_method: {prediction_method}")
            
            # Extract test event times and indicators
            event_times_test = df["time"].values
            event_indicators_test = df["event"].values.astype(int)
            
            # Extract train event times and indicators
            event_times_train = self.time_train
            event_indicators_train = self.event_train.astype(int)
            
            # Compute error using SurvivalEVAL
            error_value = mean_error(
                predicted_times=predicted_times,
                event_times=event_times_test,
                event_indicators=event_indicators_test,
                train_event_times=event_times_train,
                train_event_indicators=event_indicators_train,
                error_type=error_type,
                method=method,
                weighted=False,
                log_scale=False,
                verbose=False,
                truncated_time=None,
            )
            
            self.metrics[metric_key] = float(error_value)
            return float(error_value)
            
        except Exception as e:
            self.metrics[metric_key] = None
            return None

####################################
#   Calibration Metrics 
####################################

    def _compute_d_calibration(self, df: pd.DataFrame, bins: int = 10) -> Optional[float]:
        """
        Compute D-Calibration metric.
        
        Note: No interpolation -> assumes each observed time is exactly on the survival grid.
        """
        try:
            df = df.copy()
            df = df[df["time"] <= self.T_max_prediction]
            times = df["time"].values.astype(int)

            preds_at_obs_time = np.array([
               df.iloc[i][f"S_{int(t)}"] 
               for i, t in enumerate(times) 
               
            ])  # S_i(T_i)
            preds_at_obs_time = np.clip(preds_at_obs_time, 1e-12, 1)
            events = df["event"].values.astype(bool)

            out = self.d_calibration(event_indicators=events, predictions=preds_at_obs_time, bins=bins)

            self.metrics["d_calib_chi2"] = float(out["chi2_statistic"])
            self.metrics["d_calib_pvalue"] = float(out["p_value"])

            return float(out["chi2_statistic"])
            
        except Exception as e:
            print(f"  Warning: D-Calibration computation failed: {e}")
            self.metrics["d_calib_chi2"] = None
            self.metrics["d_calib_pvalue"] = None
            return None

    def d_calibration(
        self,
        event_indicators,
        predictions,
        bins: int = 10,
    ) -> dict:
        """
        D-Calibration by Haider et al.
        From the authors' implementation:
            https://github.com/haiderstats/survival_evaluation/blob/70e3a4d/survival_evaluation/evaluations.py#L111
        Returns the original outputs + the chi2 statistic `s`.
        """
        # include minimum to catch if probability = 1.
        bin_index = np.minimum(np.floor(predictions * bins), bins - 1).astype(int)
        censored_bin_indexes = bin_index[~event_indicators]
        uncensored_bin_indexes = bin_index[event_indicators]

        censored_predictions = predictions[~event_indicators]
        censored_contribution = 1 - (censored_bin_indexes / bins) * (
            1 / censored_predictions
        )
        censored_following_contribution = 1 / (bins * censored_predictions)

        contribution_pattern = np.tril(np.ones([bins, bins]), k=-1).astype(bool)

        following_contributions = np.matmul(
            censored_following_contribution, contribution_pattern[censored_bin_indexes]
        )
        single_contributions = np.matmul(
            censored_contribution, np.eye(bins)[censored_bin_indexes]
        )
        uncensored_contributions = np.sum(np.eye(bins)[uncensored_bin_indexes], axis=0)
        bin_count = (
            single_contributions + following_contributions + uncensored_contributions
        )
        chi2_statistic = np.sum(
            np.square(bin_count - len(predictions) / bins) / (len(predictions) / bins)
        )
        return dict(
            chi2_statistic=chi2_statistic,
            p_value=1 - chi2.cdf(chi2_statistic, bins - 1),
            bin_proportions=bin_count / len(predictions),
            censored_contributions=(single_contributions + following_contributions)
            / len(predictions),
            uncensored_contributions=uncensored_contributions / len(predictions),
        )
    
    def _compute_km_calibration(self, df: pd.DataFrame) -> Optional[float]:
        """
        Compute KM calibration using KL divergence.
        
        Measures how well predicted survival matches Kaplan-Meier estimate.
        Lower is better (0=perfect calibration).
        
        Reference: Yanagisawa et al. (ICML 2023)
        """
        try:
            y_test = Surv.from_arrays(event=df["event"].values, time=df["time"].values)
            
            # Extract survival columns
            survival_cols = sorted(
                [c for c in df.columns if c.startswith("S_")],
                key=lambda x: int(x.split("_")[1])
            )
            all_times = [int(c.split("_")[1]) for c in survival_cols]
            
            # Filter times to be within test data range
            y_test_times = df["time"].values
            min_time, max_time = y_test_times.min(), y_test_times.max()
            
            valid_cols = [c for c, t in zip(survival_cols, all_times) if min_time <= t < max_time]
            valid_times = [t for t in all_times if min_time <= t < max_time]
            
            if len(valid_times) == 0:
                print("  Warning: No valid time points found for KM calibration computation")
                self.metrics["km_calib"] = None
                return None
            
            S_pred = df[valid_cols].values
            
            # Compute Kaplan-Meier on test set
            km_estimator = SurvivalFunctionEstimator(conf_type=None)
            km_estimator.fit(y_test)
            S_km = km_estimator.predict_proba(valid_times, return_conf_int=False)
            
            # KL divergence between KM and predicted survival
            kl_div = self._km_calibration_kl(S_pred, S_km, valid_times, B=30)
            self.metrics["km_calib"] = float(kl_div)
            return float(kl_div)
            
        except Exception as e:
            print(f"  Warning: KM calibration computation failed: {e}")
            self.metrics["km_calib"] = None
            return None
    
    def _km_calibration_kl(
        self, 
        S_pred: np.ndarray, 
        S_km: np.ndarray, 
        all_times: list, 
        B: Optional[int] = None, 
        eps: float = 1e-6
    ) -> float:
        """
        KM-calibration KL divergence metric.
        
        Args:
            S_pred: Predicted survival matrix [n_samples, n_times]
            S_km: Kaplan-Meier survival curve [n_times]
            all_times: Time grid
            B: Number of bins for time discretization (if None, use all times)
            eps: Small value for numerical stability
        
        Returns:
            KL divergence (lower is better)
        """
        all_times_arr = np.asarray(all_times, float)
        
        if B is None:
            # Use all available times
            grid = all_times_arr
        else:
            # Create B evenly spaced time points
            grid_times = np.linspace(all_times[0], all_times[-1], B)
            
            # Map grid times to nearest indices in all_times
            indices = np.searchsorted(all_times_arr, grid_times, side='left')
            indices = np.clip(indices, 0, len(all_times) - 1)
            
            # Interpolate to grid using indices
            S_pred = S_pred[:, indices]
            S_km = S_km[indices]
            grid = all_times_arr[indices]
        
        # Mean predicted survival
        S_bar = S_pred.mean(axis=0)
        
        # Augment with 0 at the end to close distribution
        S_km_aug = np.append(S_km, 0.0)
        S_bar_aug = np.append(S_bar, 0.0)
        
        # Event masses per interval
        p = S_km_aug[:-1] - S_km_aug[1:]
        q = S_bar_aug[:-1] - S_bar_aug[1:]
        
        # Clip for numerical stability
        p = np.clip(p, eps, 1.0)
        q = np.clip(q, eps, 1.0)
        
        # KL divergence: KL(p || q) = sum(p * log(p / q))
        kl_div = np.sum(p * np.log(p / q))
        
        return float(kl_div)
