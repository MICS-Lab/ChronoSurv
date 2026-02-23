"""
ChronoSurv DataModule for Unified HNC (HANCOCK + TCGA-HNSC).

Inherits all preprocessing from UnifiedHNCDataModule.
Overrides setup() to wrap datasets in ChronoSurvUnifiedHNCDataset,
producing source-aware heterogeneous graphs.
"""

from torch_geometric.loader import DataLoader
from typing import Optional

from src.data.datamodule.unified_hnc import UnifiedHNCDataModule
from src.data.dataset.chrono_surv_unified_hnc import ChronoSurvUnifiedHNCDataset
from src.data.dataset.unified_hnc import UnifiedHNCDataset


class ChronoSurvUnifiedHNCDataModule(UnifiedHNCDataModule):
    """
    ChronoSurv DataModule for Unified HNC.

    Inherits all preprocessing from UnifiedHNCDataModule.
    Overrides setup() to wrap datasets in ChronoSurvUnifiedHNCDataset,
    which creates source-aware heterogeneous directed graphs.
    """

    def __init__(
        self,
        aggregate_images: bool = True,
        **kwargs,
    ):
        """
        Initialize ChronoSurv datamodule for Unified HNC.

        Args:
            aggregate_images: If True, average images into one node per modality.
                            If False, create one node per image patch.
        """
        super().__init__(**kwargs)
        self.aggregate_images = aggregate_images

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Setup datasets by reusing parent preprocessing then wrapping in ChronoSurv datasets.
        """
        # Call parent setup to do all preprocessing and create base datasets
        super().setup(stage)

        # Store base datasets before wrapping (needed for evaluation utils)
        if not hasattr(self, 'base_train_dataset'):
            self.base_train_dataset = self.train_dataset
            self.base_val_dataset = self.val_dataset
            self.base_test_dataset = self.test_dataset

        # Wrap the datasets to return HeteroData instead of tuples
        needs_wrapping = (
            (self.train_dataset is not None and not isinstance(self.train_dataset, ChronoSurvUnifiedHNCDataset)) or
            (self.val_dataset is not None and not isinstance(self.val_dataset, ChronoSurvUnifiedHNCDataset)) or
            (self.test_dataset is not None and not isinstance(self.test_dataset, ChronoSurvUnifiedHNCDataset))
        )

        if needs_wrapping:
            print("\n>>> Converting to ChronoSurv Unified HNC datasets...")

        if self.train_dataset is not None and not isinstance(self.train_dataset, ChronoSurvUnifiedHNCDataset):
            self.train_dataset = self._wrap_as_chrono_surv_dataset(self.train_dataset)
            print(f"   - Train: {len(self.train_dataset)} patient graphs")

        if self.val_dataset is not None and not isinstance(self.val_dataset, ChronoSurvUnifiedHNCDataset):
            self.val_dataset = self._wrap_as_chrono_surv_dataset(self.val_dataset)
            print(f"   - Val: {len(self.val_dataset)} patient graphs")

        if self.test_dataset is not None and not isinstance(self.test_dataset, ChronoSurvUnifiedHNCDataset):
            self.test_dataset = self._wrap_as_chrono_surv_dataset(self.test_dataset)
            print(f"   - Test: {len(self.test_dataset)} patient graphs")

    def _wrap_as_chrono_surv_dataset(self, base_dataset: UnifiedHNCDataset):
        """
        Wrap a base UnifiedHNCDataset into ChronoSurvUnifiedHNCDataset.

        This preserves all preprocessed data and changes __getitem__ behavior
        to return HeteroData graphs instead of tuples.
        """
        return ChronoSurvUnifiedHNCDataset(
            split=base_dataset.split,
            data_clinical=base_dataset.data_clinical,
            data_blood=base_dataset.data_blood,
            hancock_root=base_dataset.hancock_root,
            tcga_root=base_dataset.tcga_root,
            list_patient_id_sample=base_dataset.list_patient_id_sample,
            max_tokens_history=base_dataset.max_tokens_history,
            max_tokens_surgery=base_dataset.max_tokens_surgery,
            max_tokens_report=base_dataset.max_tokens_report,
            path_lm=base_dataset.path_lm,
            aggregate_images=self.aggregate_images,
        )

    # Override dataloaders to use PyG DataLoader

    def train_dataloader(self, batch_size: int = None, num_workers: int = None) -> DataLoader:
        batch_size = batch_size if batch_size is not None else self.batch_size
        num_workers = num_workers if num_workers is not None else self.num_workers
        return DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )

    def val_dataloader(self, batch_size: int = None, num_workers: int = None) -> DataLoader:
        batch_size = batch_size if batch_size is not None else self.batch_size
        num_workers = num_workers if num_workers is not None else self.num_workers
        return DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

    def test_dataloader(self, batch_size: int = None, num_workers: int = None) -> DataLoader:
        batch_size = batch_size if batch_size is not None else self.batch_size
        num_workers = num_workers if num_workers is not None else self.num_workers
        return DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
