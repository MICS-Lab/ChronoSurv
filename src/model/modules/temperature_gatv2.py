import torch
from torch import Tensor
from typing import Optional
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import softmax


class TemperatureGATv2Conv(GATv2Conv):
    """
    GATv2Conv with temperature-scaled attention.

    Overrides edge_update to divide attention logits by temperature
    before softmax: softmax(alpha / T).

    Args:
        temperature: Temperature for attention softmax. Default 1.0 (no change).
            < 1.0 = sharper attention (more discriminative)
            > 1.0 = softer attention (more uniform)
        All other args are passed to GATv2Conv.
    """

    def __init__(self, *args, temperature: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = temperature

    def edge_update(
        self,
        x_j: Tensor,
        x_i: Tensor,
        edge_attr: Optional[Tensor],
        index: Tensor,
        ptr: Optional[Tensor],
        dim_size: Optional[int],
    ) -> Tensor:
        x = x_i + x_j

        if edge_attr is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            assert self.lin_edge is not None
            edge_attr = self.lin_edge(edge_attr)
            edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
            x = x + edge_attr

        x = torch.nn.functional.leaky_relu(x, self.negative_slope)
        alpha = (x * self.att).sum(dim=-1)

        # Temperature scaling before softmax
        alpha = alpha / self.temperature

        alpha = softmax(alpha, index, ptr, dim_size)
        alpha = torch.nn.functional.dropout(alpha, p=self.dropout, training=self.training)
        return alpha





