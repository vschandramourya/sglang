/*
 * Copyright (c) 2024-2025 by FlashInfer team.
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

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "k_transform_kernel.cuh"
#include "tvm_ffi_utils.h"

using namespace flashinfer::private_kernels::k_transform;

namespace flashinfer {

/**
 * @brief TVM FFI binding for k_transform kernel for MLA attention
 *
 * This function performs the transformation:
 * - Extracts k_nope from first (num_heads * QK_NOPE_DIM) elements of v_full
 * - Broadcasts k_pe [M, QK_ROPE_DIM] to all num_heads heads
 * - Outputs K [M, num_heads, K_DIM_PER_HEAD] = [M, num_heads,
 * QK_NOPE_DIM+QK_ROPE_DIM]
 *
 * @param K Output tensor [M, num_heads, K_DIM_PER_HEAD] (bfloat16, contiguous)
 * @param v_full Input tensor [M, v_full_stride] (bfloat16, contiguous)
 * @param k_pe Input tensor [M, QK_ROPE_DIM] (bfloat16, contiguous)
 */
void TRTLLMKTransform(TensorView K, TensorView v_full, TensorView k_pe) {
  // Input validation
  TVM_FFI_ICHECK(K.ndim() == 3)
      << "K must be 3D tensor [M, NUM_HEADS, K_DIM_PER_HEAD]";
  TVM_FFI_ICHECK(v_full.ndim() == 2)
      << "v_full must be 2D tensor [M, V_FULL_DIM]";
  TVM_FFI_ICHECK(k_pe.ndim() == 2) << "k_pe must be 2D tensor [M, QK_ROPE_DIM]";

  int64_t M = v_full.shape()[0];
  TVM_FFI_ICHECK(M > 0) << "M (sequence length) must be > 0";

  // Expected dimensions (these are consistent across models)
  constexpr int32_t QK_NOPE_DIM = 128;
  constexpr int32_t QK_ROPE_DIM = 64;
  constexpr int32_t K_DIM_PER_HEAD = QK_NOPE_DIM + QK_ROPE_DIM; // 192
  constexpr int32_t V_HEAD_DIM = 128;

  int32_t num_heads = static_cast<int32_t>(K.size(1));
  int32_t v_full_stride = static_cast<int32_t>(v_full.size(1));

  // Validate attention dimensions match kernel expectations
  TVM_FFI_ICHECK(k_pe.shape()[1] == QK_ROPE_DIM &&
                 K.shape()[2] == K_DIM_PER_HEAD &&
                 v_full_stride == num_heads * (QK_NOPE_DIM + V_HEAD_DIM))
      << "K transform kernel expects: "
      << "k_pe[*, " << QK_ROPE_DIM << "], "
      << "K[*, " << num_heads << ", " << K_DIM_PER_HEAD << "], "
      << "v_full[*, " << num_heads * (QK_NOPE_DIM + V_HEAD_DIM) << "]. "
      << "Got: K[*, " << K.shape()[1] << ", " << K.shape()[2] << "], "
      << "k_pe[*, " << k_pe.shape()[1] << "], "
      << "v_full[*, " << v_full.shape()[1] << "]";

  // Validate tensor batch dimensions are consistent
  TVM_FFI_ICHECK(K.shape()[0] == M && k_pe.shape()[0] == M)
      << "Tensor batch dimensions must match. Expected M=" << M
      << ", got K.shape(0)=" << K.shape()[0]
      << ", k_pe.shape(0)=" << k_pe.shape()[0];

  // Ensure all tensors are on the same device
  TVM_FFI_ICHECK(K.device().device_type == v_full.device().device_type &&
                 K.device().device_id == v_full.device().device_id &&
                 v_full.device().device_type == k_pe.device().device_type &&
                 v_full.device().device_id == k_pe.device().device_id)
      << "All tensors must be on the same device";

  // Get CUDA stream
  auto stream = static_cast<cudaStream_t>(get_stream(v_full.device()));

  // Dispatch based on data type
  DLDataType dtype = K.dtype();
  if (dtype.code == kDLFloat && dtype.bits == 16) {
    // FP16
    invokeKTransform<half>(static_cast<half *>(K.data_ptr()),
                           static_cast<half *>(v_full.data_ptr()),
                           static_cast<half *>(k_pe.data_ptr()),
                           static_cast<int>(M), num_heads, v_full_stride,
                           stream);
  } else if (dtype.code == kDLBfloat && dtype.bits == 16) {
    // BF16
    invokeKTransform<__nv_bfloat16>(
        static_cast<__nv_bfloat16 *>(K.data_ptr()),
        static_cast<__nv_bfloat16 *>(v_full.data_ptr()),
        static_cast<__nv_bfloat16 *>(k_pe.data_ptr()), static_cast<int>(M),
        num_heads, v_full_stride, stream);
  } else if (dtype.code == kDLFloat && dtype.bits == 32) {
    // FP32
    invokeKTransform<float>(static_cast<float *>(K.data_ptr()),
                            static_cast<float *>(v_full.data_ptr()),
                            static_cast<float *>(k_pe.data_ptr()),
                            static_cast<int>(M), num_heads, v_full_stride,
                            stream);
  } else {
    TVM_FFI_LOG_AND_THROW(NotImplementedError)
        << "Unsupported data type for k_transform. Supported types: float16, "
           "bfloat16, float32";
  }
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(k_transform, flashinfer::TRTLLMKTransform);

} // namespace flashinfer
