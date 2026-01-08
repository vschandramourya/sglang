import concurrent.futures
import logging
import os
from typing import Iterable, Optional, Tuple

import torch
from sgl_kernel import merge_state_v2
from transformers import PretrainedConfig

from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.utils import get_layer_id
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.deepseek_v2 import AttentionBackendRegistry, AttnForwardMethod
from sglang.srt.models.deepseek_v2 import (
    DeepseekV2AttentionMLA as SGLangDeepseekV2AttentionMLA,
)
from sglang.srt.models.deepseek_v2 import (
    DeepseekV2ForCausalLM as SGLangDeepseekV2ForCausalLM,
)
from sglang.srt.models.deepseek_v2 import DeepseekV2Model as SGLangDeepseekV2Model
from sglang.srt.models.deepseek_v2 import (
    _dispatch_mla_subtype,
    _get_sum_extend_prefix_lens,
)
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import log_info_on_rank0

logger = logging.getLogger(__name__)

_HAS_FLASHINFER_K_TRANSFORM = False
if get_global_server_args().enable_mla_k_transform_kernel:
    try:
        from tgl_kernel import k_transform

        _HAS_FLASHINFER_K_TRANSFORM = True
    except ImportError:
        logger.warning("Failed to import k_transform from tgl_kernel")
else:
    logger.info("mla k_transform kernel disabled by server args")


def _is_extend_without_speculative(forward_batch):
    return (
        forward_batch.forward_mode.is_extend(include_draft_extend_v2=True)
        and not forward_batch.forward_mode.is_target_verify()
        and not forward_batch.forward_mode.is_draft_extend(include_v2=True)
    )


# the same as original, but need to do this to make _is_extend_without_speculative modification works
def handle_attention_trtllm_mla(attn, forward_batch):
    sum_extend_prefix_lens = _get_sum_extend_prefix_lens(forward_batch)
    if _is_extend_without_speculative(forward_batch) and (
        not attn.disable_chunked_prefix_cache or sum_extend_prefix_lens == 0
    ):
        return AttnForwardMethod.MHA_CHUNKED_KV
    else:
        return _dispatch_mla_subtype(attn, forward_batch)


AttentionBackendRegistry.register("trtllm_mla", handle_attention_trtllm_mla)

load_shared = int(os.environ.get("SGL_DS3_LOAD_SHARE_NORM", "0"))


disable_normed_states = (
    load_shared
    and get_global_server_args().speculative_algorithm not in {"PHOENIX", "PHOENIX2"}
)

if disable_normed_states:
    from sglang.private.layers.cursor_layernorm import CursorRMSNorm


class DeepseekV2AttentionMLA(SGLangDeepseekV2AttentionMLA):
    def _chunked_prefix_attn_mha(
        self,
        q: torch.Tensor,
        accum_output: torch.Tensor,
        accum_lse: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:

        assert forward_batch.num_prefix_chunks is not None
        for i in range(forward_batch.num_prefix_chunks):
            forward_batch.set_prefix_chunk_idx(i)

            kv_indices = forward_batch.prefix_chunk_kv_indices[i]
            # Fetch latent cache from memory pool with precomputed chunked kv indices
            kv_a_normed, k_pe = self._get_mla_kv_buffer(
                kv_indices, q.dtype, forward_batch
            )
            kv = self.kv_b_proj(kv_a_normed)[0]
            if _HAS_FLASHINFER_K_TRANSFORM:
                k_pe_2d = k_pe.squeeze(1) if k_pe.ndim == 3 else k_pe
                k = k_transform(kv, k_pe_2d, self.num_local_heads)
                kv = kv.view(
                    -1, self.num_local_heads, self.qk_nope_head_dim + self.v_head_dim
                )
                v = kv[..., self.qk_nope_head_dim :]
            else:
                kv = kv.view(
                    -1, self.num_local_heads, self.qk_nope_head_dim + self.v_head_dim
                )
                v = kv[..., self.qk_nope_head_dim :]
                k_nope = kv[..., : self.qk_nope_head_dim]

                k = torch.empty(
                    (
                        k_nope.shape[0],
                        self.num_local_heads,
                        self.qk_nope_head_dim + self.qk_rope_head_dim,
                    ),
                    dtype=v.dtype,
                    device=v.device,
                )
                k[..., : self.qk_nope_head_dim] = k_nope
                k[..., self.qk_nope_head_dim :] = k_pe

            output, lse = self.attn_mha(q, k, v, forward_batch, save_kv_cache=False)
            tmp_output = torch.empty_like(accum_output)
            tmp_lse = torch.empty_like(accum_lse)
            merge_state_v2(output, lse, accum_output, accum_lse, tmp_output, tmp_lse)
            accum_output, accum_lse = tmp_output, tmp_lse
            del kv, k, v, output, lse, tmp_output, tmp_lse

        return accum_output


class DeepseekV2Model(SGLangDeepseekV2Model):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config, quant_config, prefix)

        if disable_normed_states:
            self.norm = CursorRMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class DeepseekV2ForCausalLM(SGLangDeepseekV2ForCausalLM):
    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids, positions, forward_batch, input_embeds, pp_proxy_tensors
        )

        aux_hidden_states = None
        if disable_normed_states:
            if not forward_batch.forward_mode.is_idle():
                hidden_states, residual = hidden_states
                aux_hidden_states = [residual]

        if self.pp_group.is_last_rank:
            return self.logits_processor(
                input_ids,
                hidden_states,
                self.lm_head,
                forward_batch,
                aux_hidden_states,
            )
        else:
            return hidden_states

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]], is_nextn=False):
        if is_nextn:
            if hasattr(self.config, "num_nextn_predict_layers"):
                num_nextn_layers = self.config.num_nextn_predict_layers
                assert num_nextn_layers == 1, "Only 1 nextn layer is supported"
                # compatible with old design
                nextn_layer_id = (
                    0
                    if self.config.num_hidden_layers == 1
                    else self.config.num_hidden_layers
                )
            else:
                raise ValueError("num_nextn_predict_layers is not in the config")

        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts + self.num_fused_shared_experts,
        )
        # Params for special naming rules in mixed-precision models, for example:
        # model.layers.xx.mlp.experts.xx.w1.input_scale. For details,
        # see https://huggingface.co/Barrrrry/DeepSeek-R1-W4AFP8/blob/main.
        if self.quant_config and self.quant_config.get_name() == "w4afp8":
            expert_params_mapping += FusedMoE.make_expert_input_scale_params_mapping(
                num_experts=self.config.n_routed_experts
            )

        # Fuse q_a_proj and kv_a_proj_with_mqa along output dimension when q_lora_rank is not None
        fuse_qkv_a_proj = hasattr(self.config, "q_lora_rank") and (
            self.config.q_lora_rank is not None
        )
        cached_a_proj = {} if fuse_qkv_a_proj else None

        if is_nextn:
            nextn_layer_prefix = f"model.layers.{nextn_layer_id}"
            nextn_spec_weight_names = [
                "shared_head.norm",
                "eh_proj",
                "enorm",
                "hnorm",
            ]

        if self.num_fused_shared_experts > 0:
            assert self.num_fused_shared_experts == 1
            log_info_on_rank0(logger, "Shared experts fusion optimization enabled.")

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            params_dict = dict(self.named_parameters())
            weight_names = []
            for name, loaded_weight in weights:
                layer_id = get_layer_id(name)
                if (
                    layer_id is not None
                    and hasattr(self.model, "start_layer")
                    and (
                        layer_id < self.model.start_layer
                        or layer_id >= self.model.end_layer
                    )
                ):
                    continue
                if self.num_fused_shared_experts > 0 and "mlp.shared_experts" in name:
                    name = name.replace(
                        "mlp.shared_experts",
                        f"mlp.experts.{self.config.n_routed_experts}",
                    )

                weight_names.append(name)

                if not is_nextn:
                    if hasattr(self.config, "num_nextn_predict_layers"):
                        num_nextn_layers = self.config.num_nextn_predict_layers
                        if num_nextn_layers > 0 and name.startswith("model.layers"):
                            name_list = name.split(".")
                            if (
                                len(name_list) >= 3
                                and int(name_list[2]) >= self.config.num_hidden_layers
                            ):
                                continue
                else:

                    tie_word_embeddings = True
                    config_speculator_type = getattr(
                        self.config, "speculator_type", None
                    )
                    if load_shared and name == "model.norm.weight":
                        log_info_on_rank0(
                            logger, f"Loading shared norm weight for MTP: {name}"
                        )
                        name = "model.shared_head.norm.weight"
                    elif config_speculator_type == "phoenix":
                        # For phoenix, we have different weight and we need to handle this differently
                        if name in {"eh_proj.weight", "enorm.weight", "hnorm.weight"}:
                            original_name = name
                            name = f"{nextn_layer_prefix}." + name
                            logger.info(
                                f"Phoenix: remap top-level nextn-spec: {original_name} -> {name}"
                            )
                        # for eagle, it is always tie_word_embeddings, however for phoenix, it depends on the config tie_word_embeddings
                        tie_word_embeddings = self.config.tie_word_embeddings
                        if name == "model.norm.weight":
                            name = "model.shared_head.norm.weight"
                            logger.info(f"Phoenix: rename model.norm.weight to {name}")
                    elif not name.startswith(nextn_layer_prefix):
                        continue

                    # Use shared head and embed weights from target model
                    if (
                        "shared_head.head" in name
                        or "embed_tokens" in name
                        and tie_word_embeddings
                    ):
                        continue

                    is_decoder = True
                    # For nextn specific weights
                    for weight_name in nextn_spec_weight_names:
                        if weight_name in name:
                            name = name.replace(nextn_layer_prefix, "model")
                            is_decoder = False
                            break
                    # For decoder layer weights
                    if is_decoder:
                        name = name.replace(nextn_layer_prefix, "model.decoder")

                if "rotary_emb.inv_freq" in name:
                    continue
                for param_name, weight_name, shard_id in stacked_params_mapping:
                    # Skip non-stacked layers and experts (experts handled below).
                    if weight_name not in name:
                        continue
                    # We have mlp.experts[0].gate_proj in the checkpoint.
                    # Since we handle the experts below in expert_params_mapping,
                    # we need to skip here BEFORE we update the name, otherwise
                    # name will be updated to mlp.experts[0].gate_up_proj, which
                    # will then be updated below in expert_params_mapping
                    # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                    if ("mlp.experts." in name) and name not in params_dict:
                        continue
                    name = name.replace(weight_name, param_name)
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    futures.append(
                        executor.submit(weight_loader, param, loaded_weight, shard_id)
                    )
                    break
                else:
                    for mapping in expert_params_mapping:
                        param_name, weight_name, expert_id, shard_id = mapping
                        if weight_name not in name:
                            continue
                        name = name.replace(weight_name, param_name)
                        param = params_dict[name]
                        weight_loader = param.weight_loader
                        futures.append(
                            executor.submit(
                                weight_loader,
                                param,
                                loaded_weight,
                                name,
                                shard_id=shard_id,
                                expert_id=expert_id,
                            )
                        )
                        break
                    else:
                        # Skip loading extra bias for GPTQ models.
                        if name.endswith(".bias") and name not in params_dict:
                            continue
                        # Skip loading embed_tokens if not first rank in pipeline parallelism
                        if ".embed_tokens." in name and not self.pp_group.is_first_rank:
                            continue
                        # Skip loading norm if not last rank in pipeline parallelism
                        if ".norm." in name and not self.pp_group.is_last_rank:
                            continue
                        if fuse_qkv_a_proj and (
                            "q_a_proj" in name or "kv_a_proj_with_mqa" in name
                        ):
                            cached_a_proj[name] = loaded_weight
                            q_a_proj_name = (
                                name
                                if "q_a_proj" in name
                                else name.replace("kv_a_proj_with_mqa", "q_a_proj")
                            )
                            kv_a_proj_name = (
                                name
                                if "kv_a_proj_with_mqa" in name
                                else name.replace("q_a_proj", "kv_a_proj_with_mqa")
                            )

                            # When both q_a_proj and kv_a_proj_with_mqa has been cached, load the fused weight to parameter
                            if (
                                q_a_proj_name in cached_a_proj
                                and kv_a_proj_name in cached_a_proj
                            ):
                                q_a_proj_weight = cached_a_proj[q_a_proj_name]
                                kv_a_proj_weight = cached_a_proj[kv_a_proj_name]
                                cat_dim = 0
                                if self.quant_config is not None and (
                                    self.quant_config.get_name() == "awq"
                                    or self.quant_config.get_name() == "awq_marlin"
                                    or self.quant_config.get_name() == "moe_wna16"
                                ):
                                    cat_dim = 1
                                fused_weight = torch.cat(
                                    [q_a_proj_weight, kv_a_proj_weight], dim=cat_dim
                                )
                                param_name = (
                                    name.replace(
                                        "q_a_proj", "fused_qkv_a_proj_with_mqa"
                                    )
                                    if "q_a_proj" in name
                                    else name.replace(
                                        "kv_a_proj_with_mqa",
                                        "fused_qkv_a_proj_with_mqa",
                                    )
                                )
                                param = params_dict[param_name]

                                weight_loader = getattr(
                                    param, "weight_loader", default_weight_loader
                                )
                                futures.append(
                                    executor.submit(weight_loader, param, fused_weight)
                                )
                                cached_a_proj.pop(q_a_proj_name)
                                cached_a_proj.pop(kv_a_proj_name)
                        else:
                            if (
                                "k_scale" in name or "v_scale" in name
                            ) and name not in params_dict:
                                # modelopt attn kv scale is named differently
                                for scale in ["k_scale", "v_scale"]:
                                    if scale in name:
                                        name = name.replace(
                                            f"{scale[0]}_proj", "attn_mqa"
                                        )
                                        break
                            if name not in params_dict:
                                # modelopt ckpt contains not needed weights for MTP module:
                                # model.decoder.self_attn.attn_mqa.v_scale and
                                # model.decoder.self_attn.attn_mqa.k_scale
                                logger.warning(f"{name} not found in params_dict.")
                                continue
                            param = params_dict[name]
                            weight_loader = getattr(
                                param, "weight_loader", default_weight_loader
                            )
                            futures.append(
                                executor.submit(weight_loader, param, loaded_weight)
                            )

            # Wait for all tasks to complete and raise any exceptions.
            for future in concurrent.futures.as_completed(futures):
                future.result()

        self.post_load_weights(is_nextn=is_nextn, weight_names=weight_names)


class DeepseekV3ForCausalLM(DeepseekV2ForCausalLM):
    pass


EntryClass = [DeepseekV2ForCausalLM, DeepseekV3ForCausalLM]
