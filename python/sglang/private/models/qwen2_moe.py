"""
Qwen2MoE model with proper handling of auxiliary hidden states for Phoenix.

This extends the base Qwen2MoE implementation to handle aux_hidden_states
correctly for Phoenix1 and Phoenix2 speculative decoding.
"""

import logging
from typing import Optional, Union

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.models.qwen2_moe import Qwen2MoeModel as Qwen2MoeModelBase

logger = logging.getLogger(__name__)


class Qwen2MoeModel(Qwen2MoeModelBase):
    """
    Qwen2MoE model with Phoenix compatibility.
    """

    def set_eagle3_layers_to_capture(self, layers_to_capture):
        """Override to handle Phoenix2 layer capture.

        Args:
            layers_to_capture: List of layer indices.
                - Phoenix1: [num_layers - 1] (single last layer)
                - Phoenix2: Multiple layers from config.phoenix_layers
        """
        num_layers = self.config.num_hidden_layers

        # Filter in-bounds vs out-of-bounds
        in_bounds = [l for l in layers_to_capture if l < num_layers]

        self._capture_final = len(in_bounds) < len(layers_to_capture)

        super().set_eagle3_layers_to_capture(in_bounds)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Union[torch.Tensor, Optional[PPProxyTensors]]:
        """Forward pass with proper aux_hidden_states handling."""

        outputs = super().forward(
            input_ids, positions, forward_batch, input_embeds, pp_proxy_tensors
        )

        # Handle final layer on last rank after norm.
        if not self.pp_group.is_last_rank:
            return outputs
        else:
            # Extract components
            if isinstance(outputs, tuple):
                hidden_states, aux_hidden_states = outputs
            else:
                hidden_states = outputs
                aux_hidden_states = []

            # Capture final layer hidden states (Post-Norm)
            if getattr(self, "_capture_final", False):
                aux_hidden_states.append(hidden_states.detach())

            if len(aux_hidden_states) == 0:
                return hidden_states
            else:
                return hidden_states, aux_hidden_states
