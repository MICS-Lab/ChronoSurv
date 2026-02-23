import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, List
from torch_geometric.nn import HeteroConv, GATv2Conv, SAGEConv, LayerNorm
from torch_geometric.utils import scatter
from transformers import AutoModel

from src.model.modules.temperature_gatv2 import TemperatureGATv2Conv


class ChronoSurv(nn.Module):
    """
    ChronoSurv: A Clinical Pathway-Guided Graph Framework for Multimodal Survival Analysis.



    Architecture follows clinical pathway timeline:
    - 3 leaf node types: structured, text, images
    - 3 step nodes: step1, step2, step3
    - 1 master node

    Leaf node composition:
    - structured: 4 nodes (step1[0], step2[1], blood[2], step3[3])
    - text: 3 nodes (history[0], surgery_report[1], surgery_desc[2])
    - images: 2 nodes (lymph[0], tumor[1])

    Supports multiple convolution backends:
    - 'gatv2': GATv2Conv (default, attention-based)
    - 'sage': GraphSAGE (aggregation-based, no attention)
    - 'gatv2' + attention_temperature: Temperature-scaled GATv2Conv

    Compatible with LogisticHazardModule (outputs logits for discrete time bins).
    """

    def __init__(
        self,
        input_dims: Dict[str, int],
        num_bins: int,
        hidden_dim: int = 512,
        num_heads: int = 4,
        dropout: float = 0.3,
        lm_path: str = "./data/models/Bio_ClinicalBERT",
        freeze_text_encoder: bool = True,
        hidden_dims: Optional[List[int]] = None,
        hidden_reduction_factors: Optional[List[int]] = None,
        conv_type: str = "gatv2",
        attention_temperature: float = 1.0,
        **kwargs
    ):
        super().__init__()

        self.input_dims = input_dims
        self.num_bins = num_bins
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.conv_type = conv_type
        self.attention_temperature = attention_temperature

        # Text encoder (BioClinical BERT) - only if text modality is present
        self.text_model_path = lm_path
        self.has_text = input_dims.get('text', 0) > 0
        if self.has_text:
            self.text_encoder = AutoModel.from_pretrained(lm_path, local_files_only=True)
            if freeze_text_encoder:
                for param in self.text_encoder.parameters():
                    param.requires_grad = False
            self.text_dim = self.text_encoder.config.hidden_size
        else:
            self.text_encoder = None
            self.text_dim = 0

        # ============================================
        # Leaf encoders - only create for modalities with dim > 0
        # ============================================
        self.leaf_encoders = nn.ModuleDict()
        
        # Structured modalities
        if input_dims.get('clinical_step1', 0) > 0:
            self.leaf_encoders['clinical_step1'] = nn.Linear(input_dims['clinical_step1'], hidden_dim)
        if input_dims.get('clinical', 0) > 0:
            self.leaf_encoders['clinical'] = nn.Linear(input_dims['clinical'], hidden_dim)
        if input_dims.get('blood', 0) > 0:
            self.leaf_encoders['blood'] = nn.Linear(input_dims['blood'], hidden_dim)
        if input_dims.get('pathological', 0) > 0:
            self.leaf_encoders['pathological'] = nn.Linear(input_dims['pathological'], hidden_dim)
        
        # Text modalities
        if self.has_text:
            self.leaf_encoders['history'] = nn.Linear(self.text_dim, hidden_dim)
            self.leaf_encoders['surgery_report'] = nn.Linear(self.text_dim, hidden_dim)
            self.leaf_encoders['surgery_desc'] = nn.Linear(self.text_dim, hidden_dim)
        
        # Image modalities
        if input_dims.get('images', 0) > 0:
            self.leaf_encoders['lymph'] = nn.Linear(input_dims['images'], hidden_dim)
            self.leaf_encoders['tumor'] = nn.Linear(input_dims['images'], hidden_dim)

        # ============================================
        # Layer 1: FULLY HETEROGENEOUS - separate conv per edge type
        # Each leaf node type has its own learned transformation
        # ============================================
        conv1_dict = {}
        
        # Structured modalities - each gets its own conv
        if 'clinical_step1' in self.leaf_encoders:
            conv1_dict[('clinical_step1', 'to_step', 'step1')] = self._make_conv(hidden_dim, hidden_dim)
        
        if 'clinical' in self.leaf_encoders:
            conv1_dict[('clinical', 'to_step', 'step2')] = self._make_conv(hidden_dim, hidden_dim)
        
        if 'blood' in self.leaf_encoders:
            conv1_dict[('blood', 'to_step', 'step2')] = self._make_conv(hidden_dim, hidden_dim)
        
        if 'pathological' in self.leaf_encoders:
            conv1_dict[('pathological', 'to_step', 'step3')] = self._make_conv(hidden_dim, hidden_dim)
        
        # Text modalities - each gets its own conv
        if self.has_text:
            conv1_dict[('history', 'to_step', 'step2')] = self._make_conv(hidden_dim, hidden_dim)
            conv1_dict[('surgery_report', 'to_step', 'step3')] = self._make_conv(hidden_dim, hidden_dim)
            conv1_dict[('surgery_desc', 'to_step', 'step3')] = self._make_conv(hidden_dim, hidden_dim)
        
        # Image modalities - each gets its own conv
        if 'lymph' in self.leaf_encoders:
            conv1_dict[('lymph', 'to_step', 'step3')] = self._make_conv(hidden_dim, hidden_dim)
        if 'tumor' in self.leaf_encoders:
            conv1_dict[('tumor', 'to_step', 'step3')] = self._make_conv(hidden_dim, hidden_dim)
        
        self.conv1 = HeteroConv(conv1_dict, aggr='mean')

        # ============================================
        # Layer 2: FULLY HETEROGENEOUS - Temporal + Skip + Self-loops
        # Each edge type has its own conv
        # ============================================
        self.conv2 = HeteroConv({
            # Temporal edges - distinct convolutions
            ('step1', 'temporal', 'step2'): self._make_conv(hidden_dim, hidden_dim),
            ('step2', 'temporal', 'step3'): self._make_conv(hidden_dim, hidden_dim),
            # Skip connection
            ('step1', 'skip', 'step3'): self._make_conv(hidden_dim, hidden_dim),
            # Self-loops - distinct convolutions per step
            ('step1', 'self', 'step1'): self._make_conv(hidden_dim, hidden_dim),
            ('step2', 'self', 'step2'): self._make_conv(hidden_dim, hidden_dim),
            ('step3', 'self', 'step3'): self._make_conv(hidden_dim, hidden_dim),
        }, aggr='mean')

        self.layer_norms_conv2 = nn.ModuleDict({
            'step1': LayerNorm(hidden_dim),
            'step2': LayerNorm(hidden_dim),
            'step3': LayerNorm(hidden_dim),
        })

        # ============================================
        # Layer 3: Steps -> Master
        # ============================================
        self.conv3 = HeteroConv({
            ('step1', 'to_master', 'master'): self._make_conv(hidden_dim, hidden_dim),
            ('step2', 'to_master', 'master'): self._make_conv(hidden_dim, hidden_dim),
            ('step3', 'to_master', 'master'): self._make_conv(hidden_dim, hidden_dim),
            ('master', 'self', 'master'): self._make_conv(hidden_dim, hidden_dim),
        }, aggr='mean')

        # Activation and Dropout
        self.activation = nn.ReLU()
        self.dropout_layer = nn.Dropout(dropout)

        # ============================================
        # Hazard head
        # ============================================
        if hidden_dims is not None and hidden_reduction_factors is not None:
            final_hidden_dims = [hidden_dim // f for f in hidden_reduction_factors]
        elif hidden_reduction_factors is not None:
            final_hidden_dims = [hidden_dim // f for f in hidden_reduction_factors]
        elif hidden_dims is not None:
            final_hidden_dims = hidden_dims
        else:
            final_hidden_dims = [hidden_dim // 2, hidden_dim // 4]

        layers = []
        prev_dim = hidden_dim
        for hid_dim in final_hidden_dims:
            layers.extend([nn.Linear(prev_dim, hid_dim), nn.ReLU(), nn.Dropout(dropout)])
            prev_dim = hid_dim
        layers.append(nn.Linear(prev_dim, num_bins))
        self.hazard_head = nn.Sequential(*layers)

        # Count convolutions
        num_conv1 = len(conv1_dict)
        num_conv2 = len(self.conv2.convs)
        num_conv3 = len(self.conv3.convs)
        conv_label = self._conv_type_label()
        
        print(f"\n>>> ChronoSurv initialized:")
        print(f"    - FULLY HETEROGENEOUS architecture:")
        print(f"        Layer 1 (Leaves→Steps): {num_conv1} distinct {conv_label}")
        print(f"        Layer 2 (Temporal+Skip+Self): {num_conv2} distinct {conv_label}")
        print(f"        Layer 3 (Steps→Master): {num_conv3} distinct {conv_label}")
        print(f"    - Conv type: {conv_type}" + (f", temperature: {attention_temperature}" if conv_type == 'gatv2' and attention_temperature != 1.0 else ""))
        print(f"    - Hidden dim: {hidden_dim}, Heads: {num_heads}, Dropout: {dropout}")
        print(f"    - Leaf encoders: {list(self.leaf_encoders.keys())}")
        print(f"    - Text encoder: {lm_path if self.has_text else 'disabled'}")
        print(f"    - Hazard head: {hidden_dim} -> {final_hidden_dims} -> {num_bins}")

    # =========================================================================
    # Convolution factory
    # =========================================================================

    def _make_conv(self, in_channels: int, out_channels: int):
        """
        Create a graph convolution layer based on conv_type config.

        Returns:
            GATv2Conv, TemperatureGATv2Conv, or SAGEConv depending on self.conv_type.
        """
        if self.conv_type == 'sage':
            return SAGEConv(in_channels, out_channels, aggr='mean')
        elif self.conv_type == 'gatv2':
            if self.attention_temperature != 1.0:
                return TemperatureGATv2Conv(
                    in_channels, out_channels,
                    temperature=self.attention_temperature,
                    heads=self.num_heads,
                    concat=False,
                    add_self_loops=False,
                )
            return GATv2Conv(
                in_channels, out_channels,
                heads=self.num_heads,
                concat=False,
                add_self_loops=False,
            )
        else:
            raise ValueError(f"Unsupported conv_type: {self.conv_type}. Use 'gatv2' or 'sage'.")

    def _conv_type_label(self) -> str:
        """Get a human-readable label for the convolution type."""
        if self.conv_type == 'sage':
            return 'SAGEConv'
        elif self.attention_temperature != 1.0:
            return f'TemperatureGATv2Conv(T={self.attention_temperature})'
        return 'GATv2Conv'

    # =========================================================================
    # Forward
    # =========================================================================

    def _encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        with torch.set_grad_enabled(self.text_encoder.training):
            outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
            return outputs.last_hidden_state[:, 0, :]

    def forward(self, batch, edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor]) -> torch.Tensor:
        device = next(self.parameters()).device
        batch_size = batch['master'].x.size(0)

        x_dict = {}
        
        # Collect all leaf features and their batch indices for master aggregation
        # This handles patients with missing modalities
        all_leaf_features = []
        all_leaf_batch_indices = []

        # ============================================
        # Encode structured nodes (only if present)
        # ============================================
        if 'clinical_step1' in self.leaf_encoders and 'clinical_step1' in batch.node_types:
            x_dict['clinical_step1'] = self.dropout_layer(
                self.activation(self.leaf_encoders['clinical_step1'](batch['clinical_step1'].x))
            )
            all_leaf_features.append(x_dict['clinical_step1'])
            all_leaf_batch_indices.append(batch['clinical_step1'].batch)
        
        if 'clinical' in self.leaf_encoders and 'clinical' in batch.node_types:
            x_dict['clinical'] = self.dropout_layer(
                self.activation(self.leaf_encoders['clinical'](batch['clinical'].x))
            )
            all_leaf_features.append(x_dict['clinical'])
            all_leaf_batch_indices.append(batch['clinical'].batch)
        
        if 'blood' in self.leaf_encoders and 'blood' in batch.node_types:
            x_dict['blood'] = self.dropout_layer(
                self.activation(self.leaf_encoders['blood'](batch['blood'].x))
            )
            all_leaf_features.append(x_dict['blood'])
            all_leaf_batch_indices.append(batch['blood'].batch)
        
        if 'pathological' in self.leaf_encoders and 'pathological' in batch.node_types:
            x_dict['pathological'] = self.dropout_layer(
                self.activation(self.leaf_encoders['pathological'](batch['pathological'].x))
            )
            all_leaf_features.append(x_dict['pathological'])
            all_leaf_batch_indices.append(batch['pathological'].batch)

        # ============================================
        # Encode text nodes (only if text modality is present)
        # ============================================
        if self.has_text and 'history' in batch.node_types:
            history_emb = self._encode_text(batch['history'].input_ids, batch['history'].attention_mask)
            x_dict['history'] = self.dropout_layer(
                self.activation(self.leaf_encoders['history'](history_emb))
            )
            all_leaf_features.append(x_dict['history'])
            all_leaf_batch_indices.append(batch['history'].batch)

            surgery_report_emb = self._encode_text(batch['surgery_report'].input_ids, batch['surgery_report'].attention_mask)
            x_dict['surgery_report'] = self.dropout_layer(
                self.activation(self.leaf_encoders['surgery_report'](surgery_report_emb))
            )
            all_leaf_features.append(x_dict['surgery_report'])
            all_leaf_batch_indices.append(batch['surgery_report'].batch)

            surgery_desc_emb = self._encode_text(batch['surgery_desc'].input_ids, batch['surgery_desc'].attention_mask)
            x_dict['surgery_desc'] = self.dropout_layer(
                self.activation(self.leaf_encoders['surgery_desc'](surgery_desc_emb))
            )
            all_leaf_features.append(x_dict['surgery_desc'])
            all_leaf_batch_indices.append(batch['surgery_desc'].batch)

        # ============================================
        # Encode image nodes (only if present)
        # These can have multiple patches per patient, need pooling
        # Some patients may not have images at all
        # ============================================
        if 'lymph' in self.leaf_encoders and 'lymph' in batch.node_types:
            x_dict['lymph'] = self.dropout_layer(
                self.activation(self.leaf_encoders['lymph'](batch['lymph'].x))
            )
            # Pool patches to patient level, then add to master aggregation
            lymph_pooled = scatter(x_dict['lymph'], batch['lymph'].batch, dim=0, reduce='mean')
            all_leaf_features.append(lymph_pooled)
            all_leaf_batch_indices.append(batch['lymph'].batch.unique())
        
        if 'tumor' in self.leaf_encoders and 'tumor' in batch.node_types:
            x_dict['tumor'] = self.dropout_layer(
                self.activation(self.leaf_encoders['tumor'](batch['tumor'].x))
            )
            # Pool patches to patient level, then add to master aggregation
            tumor_pooled = scatter(x_dict['tumor'], batch['tumor'].batch, dim=0, reduce='mean')
            all_leaf_features.append(tumor_pooled)
            all_leaf_batch_indices.append(batch['tumor'].batch.unique())

        # ============================================
        # Initialize step and master nodes
        # ============================================
        x_dict['step1'] = torch.zeros(batch_size, self.hidden_dim, device=device)
        x_dict['step2'] = torch.zeros(batch_size, self.hidden_dim, device=device)
        x_dict['step3'] = torch.zeros(batch_size, self.hidden_dim, device=device)

        # Master: mean of all available leaf encodings per patient
        # Uses scatter to handle patients with different available modalities
        if all_leaf_features:
            all_features = torch.cat(all_leaf_features, dim=0)
            all_indices = torch.cat(all_leaf_batch_indices, dim=0)
            x_dict['master'] = scatter(all_features, all_indices, dim=0, reduce='mean', dim_size=batch_size)
        else:
            x_dict['master'] = torch.zeros(batch_size, self.hidden_dim, device=device)

        # ============================================
        # Layer 1: Leaves -> Steps
        # ============================================
        x_dict_1 = self.conv1(x_dict, edge_index_dict)
        x_dict_1 = {k: self.dropout_layer(self.activation(v)) for k, v in x_dict_1.items()}

        # ============================================
        # Layer 2: Temporal + Skip + Self-loops
        # ============================================
        x_dict_1_norm = {
            k: self.layer_norms_conv2[k](v) if k in self.layer_norms_conv2 else v
            for k, v in x_dict_1.items()
        }
        x_dict_2_raw = self.conv2(x_dict_1_norm, edge_index_dict)
        x_dict_2 = {
            k: x_dict_1[k] + x_dict_2_raw[k] if k in x_dict_1 else x_dict_2_raw[k]
            for k in x_dict_2_raw.keys()
        }
        x_dict_2 = {k: self.dropout_layer(self.activation(v)) for k, v in x_dict_2.items()}
        x_dict_2['master'] = x_dict['master']

        # ============================================
        # Layer 3: Steps -> Master
        # ============================================
        x_dict_3 = self.conv3(x_dict_2, edge_index_dict)

        return self.hazard_head(x_dict_3['master'])

    def __str__(self):
        hidden_dims_info = [f"{l.in_features}->{l.out_features}" for l in self.hazard_head if isinstance(l, nn.Linear)]
        return (
            f"\n--- ChronoSurv ---\n"
            f"  - Conv type: {self._conv_type_label()}\n"
            f"  - Leaf encoders: {list(self.leaf_encoders.keys())}\n"
            f"  - Hidden dim: {self.hidden_dim}, Heads: {self.num_heads}\n"
            f"  - Text encoder: {'enabled' if self.has_text else 'disabled'}\n"
            f"  - Hazard head: {' -> '.join(hidden_dims_info)}\n"
        )
