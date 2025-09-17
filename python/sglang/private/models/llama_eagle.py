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
        
        if layer_id == 0 and not phoenix_flag: #if phoenix is true then skip
            del self.input_layernorm
            setattr(self, "input_layernorm", lambda x: x)


class LlamaModel(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
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
        self.fc = torch.nn.Linear(config.hidden_size + config.target_hidden_size, config.hidden_size)
        # Add norm layer - will be set to None if not in checkpoint (handled in phoenixload_weights)
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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

        # assert last dim of spec_info hidden states matches target hidden size
        if forward_batch.spec_info.hidden_states.shape[-1] != self.config.target_hidden_size:
            raise ValueError(
                f"Expected spec_info hidden states last dim {forward_batch.spec_info.hidden_states.shape[-1]} "
                f"to match target hidden size {self.config.target_hidden_size}"
            )
        
        # forward_batch.spec_info.hidden_states is the target model hidden states
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

        # Apply norm if it exists, rms norm will handle residual
        if self.norm is not None: 
            hidden_states, _ = self.norm(hidden_states, residual)
            return hidden_states
        else:
            return hidden_states + residual


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
        #check if speculator_type key exists, and if it's eq. to 'phoenix'. If no for either, then use eagle weight loading
        self.config_speculator_type = getattr(config, "speculator_type", None) 
        if self.config_speculator_type == "phoenix":
            self.model = LlamaModelPhoenix(
                config, quant_config=quant_config, prefix=add_prefix("model", prefix)
            )
        else:
            self.model = LlamaModel(
                config, quant_config=quant_config, prefix=add_prefix("model", prefix)
            )

        #need to init self.model first    
        if self.config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            self.lm_head = ParallelLMHead(
                getattr(config, "hot_vocab_size", config.vocab_size),
                config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("lm_head", prefix),
            )

        self.logits_processor = LogitsProcessor(config)
        self.capture_aux_hidden_states = False


    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        if self.config_speculator_type == "phoenix":
            # phoenix specific weight loading!
            self._load_phoenix_weights(weights)
        else:
            # Default eagle weight loading logic
            self._load_eagle_weights(weights)

    def _load_eagle_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        for name, loaded_weight in weights:
            if "lm_head" not in name:
                name = "model." + name
                super().load_weights([(name, loaded_weight)])

    def _load_phoenix_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        # Convert to list to allow multiple iterations
        weights_list = list(weights)

        if not weights_list:
            raise ValueError("No weights found in checkpoint!")

        # Check what weights exist in checkpoint, and set model.norm and input_layernorm
        weight_names = [name for name, _ in weights_list]
        has_norm = any("norm.weight" in name for name in weight_names)
        has_input_layernorm = any("layers.0.input_layernorm" in name for name in weight_names)

        # Configure model based on checkpoint
        if not has_norm:
            self.model.norm = None

        if not has_input_layernorm:
            # Skip input_layernorm for layer 0 if not in checkpoint
            if hasattr(self.model.layers[0], 'input_layernorm'):
                delattr(self.model.layers[0], 'input_layernorm')
                setattr(self.model.layers[0], 'input_layernorm', lambda x: x)

        # actually load the weights
        loaded_count = 0
        for name, loaded_weight in weights_list:
            if "lm_head" in name:
                # Load lm_head weights directly if present in checkpoint
                super().load_weights([(name, loaded_weight)])
            else:
                # Prefix other weights with "model."
                prefixed_name = "model." + name
                super().load_weights([(prefixed_name, loaded_weight)])
            loaded_count += 1

        if loaded_count == 0:
            raise ValueError(f"No weights were successfully loaded from checkpoint...")


EntryClass = [LlamaForCausalLMEagle]