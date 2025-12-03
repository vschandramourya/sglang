"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from sglang.srt.utils import add_prefix

# Adapted from
# https://github.com/SafeAILab/EAGLE/blob/main/eagle/model/cnets.py
"""Inference-only LLaMA-EAGLE model compatible with HuggingFace weights."""

import logging
from typing import Iterable, Optional, Tuple

import torch
from torch import nn
from transformers import LlamaConfig

from sglang.srt.distributed import get_pp_group
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.models.llama import LlamaDecoderLayer, LlamaForCausalLM

logger = logging.getLogger(__name__)


class LlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(
        self,
        config: LlamaConfig,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        phoenix_flag: bool = False,
    ) -> None:
        super().__init__(config, layer_id, quant_config, prefix)

        # Skip the input_layernorm
        # https://github.com/SafeAILab/EAGLE/blob/35c78f6cdc19a73e05cf5c330b4c358dad970c6a/eagle/model/cnets.py#L427

        if layer_id == 0 and not phoenix_flag:  # if phoenix is true then skip
            del self.input_layernorm
            setattr(self, "input_layernorm", lambda x: x)


class LlamaModel(nn.Module):
    """Eagle model (non-Phoenix) - simple single-layer capture."""

    def __init__(
        self,
        config: LlamaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=add_prefix("embed_tokens", prefix),
        )
        self.layers = nn.ModuleList(
            [
                LlamaDecoderLayer(
                    config,
                    i,
                    quant_config=quant_config,
                    prefix=add_prefix(f"layers.{i}", prefix),
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        self.fc = torch.nn.Linear(config.hidden_size * 2, config.hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        if input_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = input_embeds

        hidden_states = self.fc(
            torch.cat((hidden_states, forward_batch.spec_info.hidden_states), dim=-1)
        )

        residual = None
        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions,
                hidden_states,
                forward_batch,
                residual,
            )
        return hidden_states + residual


class LlamaModelPhoenix(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size

        if not hasattr(config, "speculator_type") or config.speculator_type is None:
            if hasattr(config, "phoenix_layers") and config.phoenix_layers:
                config.speculator_type = "phoenix2"
                logger.info(
                    f"[Phoenix2] Auto-detected Phoenix2 from phoenix_layers={config.phoenix_layers}"
                )
            elif hasattr(config, "target_hidden_size"):
                # Phoenix1: has target_hidden_size but no phoenix_layers
                config.speculator_type = "phoenix"
                logger.info(
                    f"[Phoenix1] Auto-detected Phoenix1 from target_hidden_size={config.target_hidden_size}"
                )
            else:
                raise AttributeError(
                    "Phoenix model config must have 'speculator_type', 'phoenix_layers', or 'target_hidden_size' attribute. "
                    "Check model config.json"
                )

        self.speculator_type = config.speculator_type.lower()

        self.phoenix_layers = getattr(config, "phoenix_layers", None)

        self.raw_target_hidden_size = config.target_hidden_size

        if self.phoenix_layers is not None:
            self.captured_layer_count = len(self.phoenix_layers)
            if self.captured_layer_count == 0:
                raise ValueError("Phoenix2 phoenix_layers cannot be empty")
        else:
            self.captured_layer_count = 1
            logger.info("[Phoenix1] Using single layer capture (last layer)")

        self.target_capture_hidden_size = (
            self.raw_target_hidden_size * self.captured_layer_count
        )

        self.h_down_proj_bias = False
        self.h_down_proj_norm_eps = config.rms_norm_eps

        self.enable_down_proj = self.captured_layer_count > 1

        self._init_layers(config, quant_config, prefix)

        logger.info(
            "[Phoenix] Draft model init - type=%s raw_target=%d captured_layers=%d processed_target=%d down_proj=%s",
            self.speculator_type.upper(),
            self.raw_target_hidden_size,
            self.captured_layer_count,
            self.processed_target_hidden_size,
            "on" if self.enable_down_proj else "off",
        )

    def _init_layers(
        self,
        config: LlamaConfig,
        quant_config: Optional[QuantizationConfig],
        prefix: str,
    ) -> None:
        """Initialize model layers and projections."""
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=add_prefix("embed_tokens", prefix),
        )
        self.layers = nn.ModuleList(
            [
                LlamaDecoderLayer(
                    config,
                    i,
                    quant_config=quant_config,
                    prefix=add_prefix(f"layers.{i}", prefix),
                    phoenix_flag=True,
                )
                for i in range(config.num_hidden_layers)
            ]
        )

        if self.enable_down_proj:
            down_proj_out_dim = self.raw_target_hidden_size
            self.h_down_proj_norm = RMSNorm(
                self.target_capture_hidden_size, eps=self.h_down_proj_norm_eps
            )
            self.h_down_proj = nn.Linear(
                self.target_capture_hidden_size,
                down_proj_out_dim,
                bias=self.h_down_proj_bias,
            )
            self.processed_target_hidden_size = down_proj_out_dim
        else:
            self.h_down_proj_norm = None
            self.h_down_proj = None
            self.processed_target_hidden_size = self.target_capture_hidden_size

        # Embedding-Hidden projection combines draft embeddings with target hidden states
        eh_proj_in_features = config.hidden_size + self.processed_target_hidden_size
        self.eh_proj = torch.nn.Linear(
            eh_proj_in_features,
            config.hidden_size,
            bias=True,
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    @torch.no_grad()
    def project_target_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Project target hidden states for Phoenix2.
        Phoenix1: Returns as-is (single layer, no projection needed)
        Phoenix2: Projects (hidden_size * num_layers) -> hidden_size
        """
        if not self.enable_down_proj:
            # Phoenix1: no projection needed - early return to avoid overhead
            return hidden_states

        target_hidden_dim = hidden_states.shape[-1]

        # Phoenix2: Apply down-projection
        if target_hidden_dim == self.target_capture_hidden_size:
            # Expected multi-layer concatenation - apply down-projection
            if self.h_down_proj_norm is not None:
                hidden_states = self.h_down_proj_norm(hidden_states)
            if self.h_down_proj is not None:
                hidden_states = self.h_down_proj(hidden_states)
        elif target_hidden_dim == self.raw_target_hidden_size:
            # Single layer case - pass through without projection
            pass
        elif target_hidden_dim == self.processed_target_hidden_size:
            # Already processed - pass through
            pass
        else:
            # Unexpected dimension
            raise ValueError(
                f"Unexpected Phoenix hidden state dim {target_hidden_dim}; "
                f"expected one of {self.target_capture_hidden_size}, "
                f"{self.raw_target_hidden_size}, or {self.processed_target_hidden_size}."
            )

        if hidden_states.shape[-1] != self.processed_target_hidden_size:
            raise ValueError(
                f"Phoenix processed hidden size mismatch: expected {self.processed_target_hidden_size}, "
                f"got {hidden_states.shape[-1]}"
            )

        return hidden_states

    def maybe_down_projection(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Always clones to ensure Phoenix can stay anchored to original target states.
        For multi-layer mode, applies normalization and down projection.
        """

        hidden_states = hidden_states.clone().to(self.config.torch_dtype)

        input_dim = hidden_states.shape[-1]

        # Apply down-projection only if we have the expected multi-layer concatenated input
        if self.enable_down_proj and input_dim == self.target_capture_hidden_size:
            if self.h_down_proj_norm is not None:
                hidden_states = self.h_down_proj_norm(hidden_states)

            if self.h_down_proj is not None:
                hidden_states = self.h_down_proj(hidden_states)
        elif (
            input_dim == self.processed_target_hidden_size
            or input_dim == self.raw_target_hidden_size
        ):
            # Already projected to hidden_size or single layer hidden_size - return as-is
            pass
        else:
            # Unexpected dimension
            logger.warning(
                "[Phoenix] maybe_down_projection received unexpected input dim %d, "
                "expected %d (multi-layer), %d (processed), or %d (raw). Returning as-is.",
                input_dim,
                self.target_capture_hidden_size,
                self.processed_target_hidden_size,
                self.raw_target_hidden_size,
            )

        return hidden_states

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        if input_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = input_embeds

        target_hidden_states = forward_batch.spec_info.hidden_states
        target_hidden_states = self.project_target_hidden_states(target_hidden_states)

        # Combine draft and target hidden states
        combined_hidden = torch.cat((hidden_states, target_hidden_states), dim=-1)

        if combined_hidden.shape[-1] != self.eh_proj.in_features:
            raise ValueError(
                f"Phoenix eh_proj in_features mismatch: expected {self.eh_proj.in_features}, "
                f"got {combined_hidden.shape[-1]}"
            )

        hidden_states = self.eh_proj(combined_hidden)

        residual = None
        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions,
                hidden_states,
                forward_batch,
                residual,
            )

        if self.norm is not None:
            hidden_states_out, _ = self.norm(hidden_states, residual)
        else:
            hidden_states_out = hidden_states + residual

        # Check if we should capture auxiliary hidden states
        capture_aux = (
            hasattr(self, "capture_aux_hidden_states")
            and self.capture_aux_hidden_states
        ) or (
            hasattr(forward_batch, "capture_aux_hidden_states")
            and forward_batch.capture_aux_hidden_states
        )
        if capture_aux:
            return hidden_states_out, [target_hidden_states]

        return hidden_states_out


class LlamaForCausalLMEagle(LlamaForCausalLM):
    def __init__(
        self,
        config: LlamaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.quant_config = quant_config
        self.pp_group = get_pp_group()
        # Llama 3.2 1B Instruct set tie_word_embeddings to True
        # Llama 3.1 8B Instruct set tie_word_embeddings to False
        # check if speculator_type key exists, and if it's eq. to 'phoenix'. If no for either, then use eagle weight loading
        if not hasattr(config, "speculator_type") or config.speculator_type is None:

            if hasattr(config, "phoenix_layers") and config.phoenix_layers is not None:
                if not isinstance(config.phoenix_layers, list):
                    raise ValueError(
                        f"phoenix_layers must be a list, got: {type(config.phoenix_layers)}"
                    )
                config.speculator_type = "phoenix2"
                logger.info(
                    f"[Phoenix2] Detected Phoenix2 model with {len(config.phoenix_layers)} capture layers: {config.phoenix_layers}"
                )
            elif (
                hasattr(config, "target_hidden_size")
                and config.target_hidden_size != config.hidden_size
            ):
                config.speculator_type = "phoenix"
                logger.info(
                    f"[Phoenix1] Detected Phoenix1 model with target_hidden_size={config.target_hidden_size}"
                )
            else:
                # Default to Eagle
                config.speculator_type = "eagle"
                logger.info("[Eagle] Detected Eagle model (no phoenix attributes)")

        self.config_speculator_type = config.speculator_type

        # Choose model architecture based on speculator type
        is_phoenix = self.config_speculator_type in {"phoenix", "phoenix2"}
        self.model = (
            LlamaModelPhoenix(
                config, quant_config=quant_config, prefix=add_prefix("model", prefix)
            )
            if is_phoenix
            else LlamaModel(
                config, quant_config=quant_config, prefix=add_prefix("model", prefix)
            )
        )
        if is_phoenix:
            self.model.speculator_type = self.config_speculator_type

        lm_head_vocab_size = getattr(config, "hot_vocab_size", config.vocab_size)
        self.lm_head = (
            self.model.embed_tokens
            if self.config.tie_word_embeddings
            else ParallelLMHead(
                lm_head_vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("lm_head", prefix),
            )
        )

        self.logits_processor = LogitsProcessor(config)

        # Enable aux hidden states capture for Phoenix models
        self.capture_aux_hidden_states = is_phoenix
        if is_phoenix:
            self.model.capture_aux_hidden_states = True

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        if self.config_speculator_type in {"phoenix", "phoenix2"}:
            self._load_phoenix_weights(weights)
        else:
            self._load_eagle_weights(weights)

    def _load_eagle_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """Load Eagle model weights - simple prefix addition."""
        for name, loaded_weight in weights:
            if "lm_head" not in name:
                name = "model." + name
                super().load_weights([(name, loaded_weight)])

    def _load_phoenix_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """Load Phoenix model weights with conditional handling for Phoenix1 vs Phoenix2."""
        weights_list = list(weights)
        if not weights_list:
            raise ValueError("No weights found in checkpoint!")

        weight_names = [name for name, _ in weights_list]

        # Configure model based on available weights
        if not any("model.norm.weight" in name for name in weight_names):
            self.model.norm = None

        if not any("model.layers.0.input_layernorm" in name for name in weight_names):
            if hasattr(self.model.layers[0], "input_layernorm"):
                self.model.layers[0].input_layernorm = lambda x: x

        # Determine if Phoenix1 or Phoenix2 based on config
        phoenix_layers = getattr(self.config, "phoenix_layers", None)

        if phoenix_layers is None:
            is_phoenix1 = True
        else:
            is_phoenix1 = len(phoenix_layers) == 1

        loaded_count = 0
        for name, loaded_weight in weights_list:
            if is_phoenix1:
                if "h_down_proj" in name or "h_down_proj_norm" in name:
                    continue

            if "lm_head" in name:
                super().load_weights([(name, loaded_weight)])
            else:
                # Prefix other weights with "model."
                prefixed_name = name if name.startswith("model.") else f"model.{name}"
                super().load_weights([(prefixed_name, loaded_weight)])
            loaded_count += 1

        if loaded_count == 0:
            raise ValueError("No weights were successfully loaded from checkpoint...")

        logger.info(
            "[Phoenix] Loaded %d tensors for %s draft model",
            loaded_count,
            (
                self.config_speculator_type.upper()
                if self.config_speculator_type
                else "PHOENIX"
            ),
        )

    def apply_eagle3_fc(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Apply Phoenix projections to target model hidden states.
        This is called by Eagle3OneModelWorker for CUDA graph compatibility.

        Args:
            hidden_states: Target model hidden states from prepare_1st_drafter_inputs
        Returns:
            Projected target hidden states (will be combined with embeddings in forward)
        """
        if isinstance(self.model, LlamaModelPhoenix):
            return self.model.maybe_down_projection(hidden_states)
        else:
            return hidden_states


EntryClass = [LlamaForCausalLMEagle]
