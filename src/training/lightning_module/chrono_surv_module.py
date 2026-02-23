import torch
from typing import Any, Dict, Tuple
from torch_geometric.data import Batch as HeteroBatch

from src.training.lightning_module.logistic_hazard_module import LogisticHazardModule


class ChronoSurvModule(LogisticHazardModule):
    """
    Lightning Module for ChronoSurv.

    Inherits all training logic from LogisticHazardModule.
    Only overrides batch unpacking and forward to handle PyG HeteroData format.
    """

    def _unpack_batch(self, batch: HeteroBatch) -> Tuple[torch.Tensor, Tuple, torch.Tensor, torch.Tensor]:
        """
        Unpack PyG HeteroData batch into components.

        Args:
            batch: PyG Batch of HeteroData objects

        Returns:
            Tuple of (patient_id, inputs, time, event)
            where inputs is (x_dict, edge_index_dict) for the GNN
        """
        patient_id = batch.patient_id

        # PyG batches HeteroData by storing each node/edge type separately
        # We need to reconstruct x_dict and edge_index_dict from the batch
        # The batch object allows indexing by node_type: batch[node_type] returns a Storage object
        inputs = (batch, batch.edge_index_dict)

        time = batch.time
        event = batch.event
        return patient_id, inputs, time, event

    def forward(self, inputs: Tuple) -> torch.Tensor:
        """
        Forward pass with PyG HeteroData format and temperature scaling.

        Args:
            inputs: Tuple of (batch, edge_index_dict) where batch is PyG HeteroBatch

        Returns:
            Logits for discrete time bins [batch_size, num_bins]
        """
        batch, edge_index_dict = inputs
        logits = self.model(batch, edge_index_dict)
        logits = logits / self.temperature
        return logits
