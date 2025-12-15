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

#include "mla_fp8_quantization.h"
#include "tvm_ffi_utils.h"

using namespace flashinfer::private_kernels::mla;

/**
 * @brief TVM FFI wrapper for MLA Context FP8 Quantization
 *
 * Converts separate Q, K, V tensors from high precision to FP8 format
 * for Multi-head Latent Attention (MLA) context phase.
 *
 * @param q_buf Input Q tensor [total_q_len, head_num, 192]
 * @param quant_q_buf Output FP8 Q tensor
 * @param k_buf Input K tensor [total_kv_len, head_num, 192]
 * @param quant_k_buf Output FP8 K tensor
 * @param v_buf Input V tensor [total_kv_len, head_num, 128] (non-contiguous)
 * @param quant_v_buf Output FP8 V tensor (contiguous)
 * @param bmm1_scale Output BMM1 scale [2]
 * @param bmm2_scale Output BMM2 scale [1]
 * @param quant_scale_qkv Quantization scale for input QKV
 * @param quant_scale_o Quantization scale for output
 * @param dequant_scale_q Dequantization scale for Q
 * @param dequant_scale_kv Dequantization scale for KV
 * @param host_bmm1_scale Host BMM1 scale factor
 */
void mla_context_fp8_quantize(
    TensorView q_buf, TensorView quant_q_buf, TensorView k_buf,
    TensorView quant_k_buf, TensorView v_buf, TensorView quant_v_buf,
    TensorView bmm1_scale, TensorView bmm2_scale, TensorView quant_scale_qkv,
    TensorView quant_scale_o, TensorView dequant_scale_q,
    TensorView dequant_scale_kv, double host_bmm1_scale) {
  // Validate input tensors
  CHECK_INPUT(q_buf);
  CHECK_INPUT(quant_q_buf);
  CHECK_INPUT(k_buf);
  CHECK_INPUT(quant_k_buf);
  CHECK_CUDA(v_buf); // V can be non-contiguous
  CHECK_INPUT(quant_v_buf);

  // Extract dimensions
  // Q: [total_q_len, head_num, qk_head_dim (192)]
  int64_t total_q_len = q_buf.size(0);
  int64_t head_num = q_buf.size(1);
  int64_t qk_head_dim = q_buf.size(2);

  // K: [total_kv_len, head_num, qk_head_dim (192)]
  int64_t total_kv_len = k_buf.size(0);

  // V: [total_kv_len, head_num, v_head_dim (128)]
  int64_t v_head_dim = v_buf.size(2);

  // Validate dimensions
  TVM_FFI_ICHECK_EQ(qk_head_dim, 192)
      << "MLA Context FP8 Quantize: qk_head_dim must be 192, got "
      << qk_head_dim;
  TVM_FFI_ICHECK_EQ(v_head_dim, 128)
      << "MLA Context FP8 Quantize: v_head_dim must be 128, got " << v_head_dim;

  // Get CUDA stream
  cudaStream_t stream = get_stream(q_buf.device());

  // Extract scale pointers (can be null)
  // For TensorView, we check size instead of .defined()
  float const *quant_scale_qkv_ptr =
      (quant_scale_qkv.numel() > 0)
          ? static_cast<float const *>(quant_scale_qkv.data_ptr())
          : nullptr;
  float *bmm1_scale_ptr = (bmm1_scale.numel() > 0)
                              ? static_cast<float *>(bmm1_scale.data_ptr())
                              : nullptr;
  float *bmm2_scale_ptr = (bmm2_scale.numel() > 0)
                              ? static_cast<float *>(bmm2_scale.data_ptr())
                              : nullptr;
  float const *quant_scale_o_ptr =
      (quant_scale_o.numel() > 0)
          ? static_cast<float const *>(quant_scale_o.data_ptr())
          : nullptr;
  float const *dequant_scale_q_ptr =
      (dequant_scale_q.numel() > 0)
          ? static_cast<float const *>(dequant_scale_q.data_ptr())
          : nullptr;
  float const *dequant_scale_kv_ptr =
      (dequant_scale_kv.numel() > 0)
          ? static_cast<float const *>(dequant_scale_kv.data_ptr())
          : nullptr;

  // Dispatch based on input data type
  // We manually dispatch instead of using DISPATCH_DLPACK_DTYPE_TO_CTYPE
  // to avoid FP4 type which may not be available on all CUDA versions
  auto dtype_code = encode_dlpack_dtype(q_buf.dtype());

  if (dtype_code == float32_code) {
    launchMLAContextFp8Quantize<float>(
        static_cast<float const *>(q_buf.data_ptr()),
        static_cast<__nv_fp8_e4m3 *>(quant_q_buf.data_ptr()),
        static_cast<float const *>(k_buf.data_ptr()),
        static_cast<__nv_fp8_e4m3 *>(quant_k_buf.data_ptr()),
        static_cast<float const *>(v_buf.data_ptr()),
        static_cast<__nv_fp8_e4m3 *>(quant_v_buf.data_ptr()),
        static_cast<int>(total_q_len), static_cast<int>(total_kv_len),
        static_cast<int>(head_num), quant_scale_qkv_ptr, bmm1_scale_ptr,
        bmm2_scale_ptr, quant_scale_o_ptr, dequant_scale_q_ptr,
        dequant_scale_kv_ptr, static_cast<float>(host_bmm1_scale), stream);
  } else if (dtype_code == float16_code) {
    launchMLAContextFp8Quantize<half>(
        static_cast<half const *>(q_buf.data_ptr()),
        static_cast<__nv_fp8_e4m3 *>(quant_q_buf.data_ptr()),
        static_cast<half const *>(k_buf.data_ptr()),
        static_cast<__nv_fp8_e4m3 *>(quant_k_buf.data_ptr()),
        static_cast<half const *>(v_buf.data_ptr()),
        static_cast<__nv_fp8_e4m3 *>(quant_v_buf.data_ptr()),
        static_cast<int>(total_q_len), static_cast<int>(total_kv_len),
        static_cast<int>(head_num), quant_scale_qkv_ptr, bmm1_scale_ptr,
        bmm2_scale_ptr, quant_scale_o_ptr, dequant_scale_q_ptr,
        dequant_scale_kv_ptr, static_cast<float>(host_bmm1_scale), stream);
  } else if (dtype_code == bfloat16_code) {
    launchMLAContextFp8Quantize<__nv_bfloat16>(
        static_cast<__nv_bfloat16 const *>(q_buf.data_ptr()),
        static_cast<__nv_fp8_e4m3 *>(quant_q_buf.data_ptr()),
        static_cast<__nv_bfloat16 const *>(k_buf.data_ptr()),
        static_cast<__nv_fp8_e4m3 *>(quant_k_buf.data_ptr()),
        static_cast<__nv_bfloat16 const *>(v_buf.data_ptr()),
        static_cast<__nv_fp8_e4m3 *>(quant_v_buf.data_ptr()),
        static_cast<int>(total_q_len), static_cast<int>(total_kv_len),
        static_cast<int>(head_num), quant_scale_qkv_ptr, bmm1_scale_ptr,
        bmm2_scale_ptr, quant_scale_o_ptr, dequant_scale_q_ptr,
        dequant_scale_kv_ptr, static_cast<float>(host_bmm1_scale), stream);
  } else {
    TVM_FFI_ICHECK(false) << "MLA Context FP8 Quantize: Unsupported data type. "
                          << "Supported types: float32, float16, bfloat16. "
                          << "Got dtype code: " << dtype_code;
  }
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(mla_context_fp8_quantize,
                              mla_context_fp8_quantize);
