import concurrent.futures
import logging
import os
from typing import Iterable, Tuple

import torch

from sglang.srt.layers.dp_attention import is_dp_attention_enabled
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.quantization import deep_gemm_wrapper
from sglang.srt.layers.quantization.fp8_kernel import (
    per_tensor_quant_mla_fp8,
    per_token_group_quant_mla_deep_gemm_masked_fp8,
)
from sglang.srt.layers.utils import get_layer_id
from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.deepseek_v2 import AttnForwardMethod
from sglang.srt.models.deepseek_v2 import (
    DeepseekV2AttentionMLA as SGLANG_DeepseekV2AttentionMLA,
)
from sglang.srt.models.deepseek_v2 import (
    DeepseekV2ForCausalLM as SGLangDeepseekV2ForCausalLM,
)
from sglang.srt.utils import (
    get_bool_env_var,
    is_cuda,
    is_gfx95_supported,
    is_hip,
    log_info_on_rank0,
    use_intel_amx_backend,
)

_is_hip = is_hip()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip
_is_gfx95_supported = is_gfx95_supported()
_use_aiter_gfx95 = _use_aiter and _is_gfx95_supported
_is_cuda = is_cuda()

if _use_aiter_gfx95:
    from sglang.srt.layers.quantization.rocm_mxfp4_utils import (
        batched_gemm_afp4wfp4_pre_quant,
        fused_flatten_mxfp4_quant,
    )
    from sglang.srt.layers.rocm_linear_utils import fused_qk_rope_cat

if _is_cuda:
    from sgl_kernel import bmm_fp8

load_shared = int(os.environ.get("SGL_DS3_LOAD_SHARE_NORM", "0"))
logger = logging.getLogger(__name__)


class DeepseekV2AttentionMLA(SGLANG_DeepseekV2AttentionMLA):
    def dispatch_attn_forward_method(
        self, forward_batch: ForwardBatch
    ) -> AttnForwardMethod:
        def _dispatch_mla_subtype():
            if _is_hip:
                if (
                    self.rocm_fused_decode_mla
                    and forward_batch.forward_mode.is_decode()
                ):
                    return AttnForwardMethod.MLA_FUSED_ROPE
                else:
                    return AttnForwardMethod.MLA
            else:
                if hasattr(self, "fused_qkv_a_proj_with_mqa") and use_intel_amx_backend(
                    self
                ):
                    return AttnForwardMethod.MLA_FUSED_ROPE_CPU
                else:
                    return AttnForwardMethod.MLA

        # Determine attention backend used by current forward batch
        if forward_batch.forward_mode.is_decode_or_idle():
            attention_backend = global_server_args_dict["decode_attention_backend"]
        elif (
            forward_batch.forward_mode.is_target_verify()
            or forward_batch.forward_mode.is_draft_extend()
        ):
            # Use the specified backend for speculative operations (both verify and draft extend)
            if global_server_args_dict["speculative_attention_mode"] == "decode":
                attention_backend = global_server_args_dict["decode_attention_backend"]
            else:  # default to prefill
                attention_backend = global_server_args_dict["prefill_attention_backend"]
        else:
            attention_backend = global_server_args_dict["prefill_attention_backend"]
        self.current_attention_backend = attention_backend

        if attention_backend == "ascend":
            if (
                forward_batch.forward_mode.is_extend()
                and not forward_batch.forward_mode.is_target_verify()
                and not forward_batch.forward_mode.is_draft_extend()
            ):
                return AttnForwardMethod.MHA
            else:
                return AttnForwardMethod.MLA
        elif (
            attention_backend == "flashinfer"
            or attention_backend == "fa3"
            or attention_backend == "flashmla"
            or attention_backend == "cutlass_mla"
        ):
            # Use MHA with chunked KV cache when prefilling on long sequences.
            sum_extend_prefix_lens = (
                sum(forward_batch.extend_prefix_lens_cpu)
                if forward_batch.extend_prefix_lens_cpu is not None
                else 0
            )
            # Flashinfer MLA: Do not absorb when enabling ragged prefill
            disable_ragged = (
                attention_backend == "flashinfer" or attention_backend == "flashmla"
            ) and self.flashinfer_mla_disable_ragged
            if (
                not disable_ragged
                and forward_batch.forward_mode.is_extend()
                and not forward_batch.forward_mode.is_target_verify()
                and not forward_batch.forward_mode.is_draft_extend()
                and (
                    (
                        sum_extend_prefix_lens >= self.chunked_prefix_cache_threshold
                        and not self.disable_chunked_prefix_cache
                    )
                    or sum_extend_prefix_lens == 0
                )
            ):
                return AttnForwardMethod.MHA_CHUNKED_KV
            else:
                return _dispatch_mla_subtype()
        elif attention_backend == "fa4":
            # TODO(cicirori): use FA4 MHA for DeepSeekV3 for now
            return AttnForwardMethod.MHA_CHUNKED_KV
        elif attention_backend == "trtllm_mla":
            if (
                forward_batch.forward_mode.is_extend()
                and not forward_batch.forward_mode.is_target_verify()
                and not forward_batch.forward_mode.is_draft_extend()
            ):
                return AttnForwardMethod.MHA_CHUNKED_KV
            else:
                return _dispatch_mla_subtype()
        elif attention_backend == "trtllm_mla_tgl":
            if (
                forward_batch.forward_mode.is_extend()
                and not forward_batch.forward_mode.is_target_verify()
                and not forward_batch.forward_mode.is_draft_extend()
            ):
                return AttnForwardMethod.MHA_CHUNKED_KV
            else:
                return _dispatch_mla_subtype()
        elif attention_backend == "aiter":
            if (
                forward_batch.forward_mode.is_extend()
                and not forward_batch.forward_mode.is_target_verify()
                and not forward_batch.forward_mode.is_draft_extend()
            ):
                if is_dp_attention_enabled():
                    if sum(forward_batch.extend_prefix_lens_cpu) == 0:
                        return AttnForwardMethod.MHA
                    else:
                        return AttnForwardMethod.MLA
                else:
                    return AttnForwardMethod.MHA
            else:
                return AttnForwardMethod.MLA
        else:
            # Triton: Use normal computation for prefill and use weight absorption for extend/decode
            if (
                forward_batch.forward_mode.is_extend()
                and not forward_batch.forward_mode.is_target_verify()
                and not forward_batch.forward_mode.is_draft_extend()
                and sum(forward_batch.extend_prefix_lens_cpu) == 0
            ):
                return AttnForwardMethod.MHA
            else:
                return _dispatch_mla_subtype()

    def forward_absorb_core(
        self, q_pe, k_pe, q_nope_out, k_nope, forward_batch, zero_allocator, positions
    ):
        if (
            self.current_attention_backend == "fa3"
            or self.current_attention_backend == "flashinfer"
            or self.current_attention_backend == "cutlass_mla"
            or self.current_attention_backend == "trtllm_mla"
            or self.current_attention_backend == "trtllm_mla_tgl"
            or self.current_attention_backend == "ascend"
        ):
            extra_args = {}
            if self._fuse_rope_for_trtllm_mla(forward_batch):
                extra_args = {
                    "cos_sin_cache": self.rotary_emb.cos_sin_cache,
                    "is_neox": self.rotary_emb.is_neox_style,
                }
            attn_output = self.attn_mqa(
                q_nope_out,
                k_nope,
                k_nope,
                forward_batch,
                q_rope=q_pe,
                k_rope=k_pe,
                **extra_args,
            )
        else:
            if _use_aiter_gfx95:
                cos = self.rotary_emb.cos_cache
                sin = self.rotary_emb.sin_cache
                q, k = fused_qk_rope_cat(
                    q_nope_out,
                    q_pe,
                    k_nope,
                    k_pe,
                    positions,
                    cos,
                    sin,
                    self.rotary_emb.is_neox_style,
                )
            else:
                q = torch.cat([q_nope_out, q_pe], dim=-1)
                k = torch.cat([k_nope, k_pe], dim=-1)

            attn_output = self.attn_mqa(q, k, k_nope, forward_batch)
        attn_output = attn_output.view(-1, self.num_local_heads, self.kv_lora_rank)

        if self.use_deep_gemm_bmm:
            attn_output_val, attn_output_scale, masked_m, expected_m, aligned_m = (
                per_token_group_quant_mla_deep_gemm_masked_fp8(
                    attn_output.transpose(0, 1)
                )
            )
            attn_bmm_output = attn_output.new_empty(
                (self.num_local_heads, aligned_m, self.v_head_dim)
            )
            deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_masked(
                (attn_output_val, attn_output_scale),
                (self.w_vc, self.w_scale_v),
                attn_bmm_output,
                masked_m,
                expected_m,
            )
            attn_bmm_output = (
                attn_bmm_output[:, :expected_m, :].transpose(0, 1).flatten(1, 2)
            )
        elif _is_hip:
            # TODO(haishaw): add bmm_fp8 to ROCm
            if _use_aiter_gfx95 and self.w_vc.dtype == torch.uint8:
                x = attn_output.transpose(0, 1)
                attn_bmm_output = torch.empty(
                    x.shape[0],
                    x.shape[1],
                    self.w_vc.shape[2],
                    device=x.device,
                    dtype=torch.bfloat16,
                )
                batched_gemm_afp4wfp4_pre_quant(
                    x,
                    self.w_vc.transpose(-2, -1),
                    self.w_scale_v.transpose(-2, -1),
                    torch.bfloat16,
                    attn_bmm_output,
                )
            else:
                attn_bmm_output = torch.bmm(
                    attn_output.to(torch.bfloat16).transpose(0, 1),
                    self.w_vc.to(torch.bfloat16) * self.w_scale,
                )

            if self.o_proj.weight.dtype == torch.uint8:
                attn_bmm_output = attn_bmm_output.transpose(0, 1)
                attn_bmm_output = fused_flatten_mxfp4_quant(attn_bmm_output)
            else:
                attn_bmm_output = attn_bmm_output.transpose(0, 1).flatten(1, 2)

        elif self.w_vc.dtype == torch.float8_e4m3fn:
            attn_output_val, attn_output_scale = per_tensor_quant_mla_fp8(
                attn_output.transpose(0, 1),
                zero_allocator.allocate(1),
            )
            attn_bmm_output = bmm_fp8(
                attn_output_val,
                self.w_vc,
                attn_output_scale,
                self.w_scale,
                torch.bfloat16,
            )
            attn_bmm_output = attn_bmm_output.transpose(0, 1).flatten(1, 2)
        else:
            attn_bmm_output = torch.empty(
                (attn_output.shape[0], self.num_local_heads * self.v_head_dim),
                dtype=attn_output.dtype,
                device=attn_output.device,
            )
            torch.bmm(
                attn_output.transpose(0, 1),
                self.w_vc,
                out=attn_bmm_output.view(
                    -1, self.num_local_heads, self.v_head_dim
                ).transpose(0, 1),
            )
        output, _ = self.o_proj(attn_bmm_output)

        return output

    def _fuse_rope_for_trtllm_mla(self, forward_batch: ForwardBatch) -> bool:
        """
        Check if we should skip rope and do fused rope+quantize for TRTLLM MLA decode in fp8_e4m3 path.
        """
        if global_server_args_dict["decode_attention_backend"] == "trtllm_mla_tgl":
            return (
                forward_batch.forward_mode.is_decode_or_idle()
                or forward_batch.forward_mode.is_target_verify()
                or forward_batch.forward_mode.is_draft_extend()
            ) and forward_batch.attn_backend.data_type == torch.float8_e4m3fn
        else:
            return (
                self.current_attention_backend == "trtllm_mla"
                and forward_batch.forward_mode.is_decode_or_idle()
                and forward_batch.attn_backend.data_type == torch.float8_e4m3fn
            )


class DeepseekV2ForCausalLM(SGLangDeepseekV2ForCausalLM):
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
                    config_speculator_type = getattr(self.config, "speculator_type", None)
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
                    if "shared_head.head" in name or "embed_tokens" in name and tie_word_embeddings:
                        continue
                    
                    if "embed_tokens" in name or "shared_head.head" in name:
                        print("phoenix: not skipping loading embed_tokens or shared_head.head")

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
                    if "layer" not in name and is_nextn:
                        logger.info(f"Phoenix: loading weight 1: {name}")
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
                        if "layer" not in name and "decoder" not in name and is_nextn:
                            logger.info(f"Phoenix: loading weight 2: {name}")
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
                                if "layer" not in name and is_nextn:
                                    logger.info(f"Phoenix: loading weight 3: {param_name}")
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
                            if "layer" not in name and is_nextn:
                                logger.info(f"Phoenix: loading weight 4: {name}")
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
