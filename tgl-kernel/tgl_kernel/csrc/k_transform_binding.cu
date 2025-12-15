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
 * - Extracts k_nope from first 4096 elements of v_full [M, 8192]
 * - Broadcasts k_pe [M, 64] to all 32 heads
 * - Outputs K [M, 32, 192] = [M, 32, 128+64]
 *
 * @param K Output tensor [M, 32, 192] (contiguous)
 * @param v_full Input tensor [M, 8192] (contiguous)
 * @param k_pe Input tensor [M, 64] (contiguous)
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

  // Kernel expects hardcoded dimensions: NUM_HEADS=32, QK_NOPE_DIM=128,
  // QK_ROPE_DIM=64, K_DIM_PER_HEAD=192 v_full must be [M, 8192] where 8192 =
  // NUM_HEADS * (QK_NOPE_DIM + V_HEAD_DIM) = 32 * (128 + 128)
  constexpr int64_t EXPECTED_NUM_HEADS = 32;
  constexpr int64_t EXPECTED_QK_NOPE_DIM = 128;
  constexpr int64_t EXPECTED_QK_ROPE_DIM = 64;
  constexpr int64_t EXPECTED_K_DIM_PER_HEAD = 192; // QK_NOPE_DIM + QK_ROPE_DIM
  constexpr int64_t EXPECTED_V_HEAD_DIM = 128;
  constexpr int64_t EXPECTED_V_FULL_DIM =
      EXPECTED_NUM_HEADS * (EXPECTED_QK_NOPE_DIM + EXPECTED_V_HEAD_DIM); // 8192

  // Validate attention dimensions match kernel expectations
  TVM_FFI_ICHECK(K.shape()[1] == EXPECTED_NUM_HEADS &&
                 K.shape()[2] == EXPECTED_K_DIM_PER_HEAD &&
                 k_pe.shape()[1] == EXPECTED_QK_ROPE_DIM &&
                 v_full.shape()[1] == EXPECTED_V_FULL_DIM)
      << "K transform kernel expects: "
      << "K[*, " << EXPECTED_NUM_HEADS << ", " << EXPECTED_K_DIM_PER_HEAD
      << "], "
      << "k_pe[*, " << EXPECTED_QK_ROPE_DIM << "], "
      << "v_full[*, " << EXPECTED_V_FULL_DIM << "]. "
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
                           static_cast<int>(M), stream);
  } else if (dtype.code == kDLBfloat && dtype.bits == 16) {
    // BF16
    invokeKTransform<__nv_bfloat16>(
        static_cast<__nv_bfloat16 *>(K.data_ptr()),
        static_cast<__nv_bfloat16 *>(v_full.data_ptr()),
        static_cast<__nv_bfloat16 *>(k_pe.data_ptr()), static_cast<int>(M),
        stream);
  } else if (dtype.code == kDLFloat && dtype.bits == 32) {
    // FP32
    invokeKTransform<float>(static_cast<float *>(K.data_ptr()),
                            static_cast<float *>(v_full.data_ptr()),
                            static_cast<float *>(k_pe.data_ptr()),
                            static_cast<int>(M), stream);
  } else {
    TVM_FFI_LOG_AND_THROW(NotImplementedError)
        << "Unsupported data type for k_transform. Supported types: float16, "
           "bfloat16, float32";
  }
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(k_transform, flashinfer::TRTLLMKTransform);

} // namespace flashinfer
