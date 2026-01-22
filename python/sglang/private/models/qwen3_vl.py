from typing import Iterable, Tuple

import torch

from sglang.srt.layers.utils import get_layer_id
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.qwen3_vl import (
    Qwen3VLForConditionalGeneration as SGLangQwen3VLForConditionalGeneration,
)


class Qwen3VLForConditionalGeneration(SGLangQwen3VLForConditionalGeneration):
    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            ("gate_up_proj", "up_proj", 1),
            ("gate_up_proj", "gate_proj", 0),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if "language_model" in name:
                name = name.replace(r"model.language_model.", r"model.")
            layer_id = get_layer_id(name)

            if (
                self.pp_group.world_size > 1
                and self.pp_group.is_last_rank
                and "model.embed_tokens.weight" in name
            ):
                if "lm_head.weight" in params_dict:
                    lm_head_param = params_dict["lm_head.weight"]
                    weight_loader = getattr(
                        lm_head_param, "weight_loader", default_weight_loader
                    )
                    weight_loader(lm_head_param, loaded_weight)

            is_visual = "visual" in name
            if (
                not is_visual
                and layer_id is not None
                and hasattr(self, "model")
                and hasattr(self.model, "start_layer")
                and (
                    layer_id < self.model.start_layer
                    or layer_id >= self.model.end_layer
                )
            ):
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if "visual" in name:
                    continue
                name = name.replace(weight_name, param_name)

                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Skip loading visual/language model weights
                if (
                    self.config.encoder_only or self.config.language_only
                ) and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if "visual" in name:
                    # adapt to VisionAttention
                    name = name.replace(r"attn.qkv.", r"attn.qkv_proj.")
                    name = name.replace(r"model.visual.", r"visual.")

                try:
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    if name in params_dict.keys():
                        param = params_dict[name]
                    else:
                        continue

                except KeyError:
                    print(params_dict.keys())
                    raise

                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)


EntryClass = Qwen3VLForConditionalGeneration
