/*
 * Copyright (c) 2025 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cuda_fp8.h>
#include <cuda_runtime.h>

namespace flashinfer {
namespace private_kernels {
namespace mla {

/**
 * @brief Launch MLA Context FP8 Quantization
 *
 * Quantizes separate Q, K, V tensors from high precision formats (float, half,
 * bfloat16) to FP8 (e4m3) format for Multi-head Latent Attention (MLA).
 *
 * This kernel is designed for MLA architecture where:
 * - Q dimension: [total_q_len, head_num, qk_nope_dim (128) + qk_rope_dim (64)]
 * - K dimension: [total_kv_len, head_num, qk_nope_dim (128) + qk_rope_dim (64)]
 * - V dimension: [total_kv_len, head_num, v_dim (128)]
 *
 * Note: V buffer is non-contiguous in source (interleaved with K_nope),
 * but will be contiguous in output.
 *
 * @tparam T Input data type (float, half, or __nv_bfloat16)
 * @param q_buf Input Q buffer (high precision)
 * @param quant_q_buf Output quantized Q buffer (FP8)
 * @param k_buf Input K buffer (high precision)
 * @param quant_k_buf Output quantized K buffer (FP8)
 * @param v_buf Input V buffer (high precision, non-contiguous)
 * @param quant_v_buf Output quantized V buffer (FP8, contiguous)
 * @param total_q_len Total query sequence length
 * @param total_kv_len Total key/value sequence length
 * @param head_num Number of attention heads
 * @param quant_scale_qkv Quantization scale for Q/K/V (device pointer)
 * @param bmm1_scale Output BMM1 scale for attention computation (device
 * pointer)
 * @param bmm2_scale Output BMM2 scale for attention computation (device
 * pointer)
 * @param quant_scale_o Output quantization scale (device pointer)
 * @param dequant_scale_q Dequantization scale for Q (device pointer)
 * @param dequant_scale_kv Dequantization scale for K/V (device pointer)
 * @param host_bmm1_scale Host-side BMM1 scale factor
 * @param stream CUDA stream for kernel execution
 */
template <typename T>
void launchMLAContextFp8Quantize(
    T const *q_buf, __nv_fp8_e4m3 *quant_q_buf, T const *k_buf,
    __nv_fp8_e4m3 *quant_k_buf, T const *v_buf, __nv_fp8_e4m3 *quant_v_buf,
    int total_q_len, int total_kv_len, int head_num,
    float const *quant_scale_qkv, float *bmm1_scale, float *bmm2_scale,
    float const *quant_scale_o, float const *dequant_scale_q,
    float const *dequant_scale_kv, float host_bmm1_scale, cudaStream_t stream);

} // namespace mla
} // namespace private_kernels
} // namespace flashinfer
