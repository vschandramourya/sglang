/*
 * Copyright (c) 2025 by FlashInfer team.
 * Adapted from TensorRT-LLM MLA kernels.
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
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>

namespace flashinfer {
namespace private_kernels {
namespace mla {

// Vector type traits for efficient memory access
template <typename T> struct VecType {
  using Type = T;
};

template <> struct VecType<float> {
  using Type = float4;
};

template <> struct VecType<half> {
  using Type = uint4;
};

template <> struct VecType<__nv_bfloat16> {
  struct bf16_16_t {
    __nv_bfloat162 data[4];
  };
  using Type = bf16_16_t;
};

// FP8 16-element vector type
struct __align__(16) fp8_16_t {
  __nv_fp8x4_e4m3 x;
  __nv_fp8x4_e4m3 y;
  __nv_fp8x4_e4m3 z;
  __nv_fp8x4_e4m3 w;
};

template <> struct VecType<__nv_fp8_e4m3> {
  using Type = fp8_16_t;
};

// Type conversion helper
template <typename T> struct TypeConverter {
  using Type = T;
};

template <> struct TypeConverter<half> {
  using Type = half2;
};

template <> struct TypeConverter<__nv_bfloat16> {
  using Type = __nv_bfloat162;
};

// CUDA cast helper
template <typename Tout, typename Tin>
inline __device__ Tout cuda_cast(Tin val) {
  return static_cast<Tout>(val);
}

template <> inline __device__ float2 cuda_cast<float2, half2>(half2 val) {
  return __half22float2(val);
}

template <>
inline __device__ float2 cuda_cast<float2, __nv_bfloat162>(__nv_bfloat162 val) {
  return __bfloat1622float2(val);
}

template <> inline __device__ float2 cuda_cast<float2, float2>(float2 val) {
  return val;
}

/**
 * @brief Quantize and copy data from source to FP8 destination
 *
 * This function performs element-wise quantization from a higher precision type
 * (float, half, bfloat16) to FP8 (e4m3) format with scaling.
 *
 * @tparam SrcType Source data type (float, half, or bfloat16)
 * @tparam NUM Number of elements to quantize
 * @param dst_global_ptr Destination pointer (FP8 format)
 * @param src_fragment_ptr Source pointer (high precision format)
 * @param scale_val Quantization scale factor
 */
template <typename SrcType, int NUM>
inline __device__ void quantCopy(__nv_fp8_e4m3 *dst_global_ptr,
                                 SrcType const *src_fragment_ptr,
                                 float const scale_val = 1.f) {
  using DstVecType =
      typename std::conditional<sizeof(SrcType) == 2, float2, float>::type;
  using SrcType2 =
      typename std::conditional<sizeof(SrcType) == 2,
                                typename TypeConverter<SrcType>::Type,
                                float2>::type;

  static constexpr int COPY_SIZE = sizeof(DstVecType);
  static constexpr int TOTAL_COPY_SIZE = NUM * sizeof(__nv_fp8_e4m3);
  static constexpr int LOOP_NUM = TOTAL_COPY_SIZE / COPY_SIZE;
  static_assert(TOTAL_COPY_SIZE % COPY_SIZE == 0,
                "Total copy size must be divisible by copy size");

  static constexpr int CVT_NUM = COPY_SIZE / sizeof(__nv_fp8_e4m3) / 2;
  static_assert(COPY_SIZE % (sizeof(__nv_fp8_e4m3) * 2) == 0,
                "Copy size alignment issue");

  DstVecType fragment;
  int offset = 0;

#pragma unroll
  for (int i = 0; i < LOOP_NUM; ++i) {
#pragma unroll
    for (int j = 0; j < CVT_NUM; ++j) {
      float2 val2 = cuda_cast<float2>(
          reinterpret_cast<SrcType2 const *>(src_fragment_ptr)[j + offset]);
      val2.x *= scale_val;
      val2.y *= scale_val;
      reinterpret_cast<__nv_fp8x2_e4m3 *>(&fragment)[j] = __nv_fp8x2_e4m3(val2);
    }
    reinterpret_cast<DstVecType *>(dst_global_ptr)[i] = fragment;
    offset += CVT_NUM;
  }
}

/**
 * @brief MLA Context FP8 Quantization Kernel
 *
 * This kernel quantizes separate Q, K, V tensors from high precision formats
 * (float, half, bfloat16) to FP8 (e4m3) format for MLA (Multi-head Latent
 * Attention). It also calculates the necessary scales for BMM1 and BMM2
 * operations.
 *
 * Memory Layout:
 * - Q: [total_q_len, head_num, qk_nope_dim + qk_rope_dim] (contiguous)
 * - K: [total_kv_len, head_num, qk_nope_dim + qk_rope_dim] (contiguous)
 * - V: [total_kv_len, head_num, v_dim] (contiguous)
 *
 * @tparam T Input data type (float, half, or __nv_bfloat16)
 * @tparam BLOCK_SIZE Number of threads per block
 * @tparam QK_NOPE_HEAD_DIM Dimension of Q/K without RoPE (typically 128)
 * @tparam QK_ROPE_HEAD_DIM Dimension of Q/K RoPE component (typically 64)
 * @tparam V_HEAD_DIM Dimension of V (typically 128)
 * @tparam COMPUTE_SCALES Whether to compute BMM scales (true) or skip scale
 * computation (false)
 */
template <typename T, int BLOCK_SIZE, int QK_NOPE_HEAD_DIM,
          int QK_ROPE_HEAD_DIM, int V_HEAD_DIM, bool COMPUTE_SCALES>
__global__ void quantizeCopyMLAInputToFp8Kernel(
    T const *q_buf, __nv_fp8_e4m3 *quant_q_buf, T const *k_buf,
    __nv_fp8_e4m3 *quant_k_buf, T const *v_buf, __nv_fp8_e4m3 *quant_v_buf,
    int total_q_len, int total_kv_len, float const *quant_scale_qkv_ptr,
    float *bmm1_scale, float *bmm2_scale, float const *quant_scale_o,
    float const *dequant_scale_q, float const *dequant_scale_kv,
    float host_bmm1_scale) {
  // Constants
  using VecT = typename VecType<T>::Type;
  constexpr auto BYTES_PER_ELT = sizeof(T);
  constexpr auto BYTES_PER_LOAD = 16;
  constexpr auto ELTS_PER_VEC = BYTES_PER_LOAD / BYTES_PER_ELT;
  constexpr auto QK_HEAD_DIM = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM;

  static_assert((QK_HEAD_DIM * BYTES_PER_ELT) % BYTES_PER_LOAD == 0,
                "QK head size needs to be multiple of 16 bytes.");
  static_assert((V_HEAD_DIM * BYTES_PER_ELT) % BYTES_PER_LOAD == 0,
                "V head size needs to be multiple of 16 bytes.");

  constexpr auto QK_VECS_PER_HEAD =
      QK_HEAD_DIM * BYTES_PER_ELT / BYTES_PER_LOAD;
  constexpr auto V_VECS_PER_HEAD = V_HEAD_DIM * BYTES_PER_ELT / BYTES_PER_LOAD;

  static_assert(BLOCK_SIZE % QK_VECS_PER_HEAD == 0,
                "Kernel block should be able to handle entire heads.");
  static_assert(BLOCK_SIZE % V_VECS_PER_HEAD == 0,
                "Kernel block should be able to handle entire heads.");

  constexpr auto QK_TOKENS_PER_BLOCK = BLOCK_SIZE / QK_VECS_PER_HEAD;
  constexpr auto V_TOKENS_PER_BLOCK = BLOCK_SIZE / V_VECS_PER_HEAD;

  size_t const head_idx = blockIdx.z;
  size_t const head_num = gridDim.z;

  // Calculate BMM scales (only once by first thread)
  // This entire block is compiled out when COMPUTE_SCALES=false
  if constexpr (COMPUTE_SCALES) {
    if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 &&
        threadIdx.x == 0) {
      float dequant_scale_q_val = dequant_scale_q ? dequant_scale_q[0] : 1.f;
      float dequant_scale_kv_val = dequant_scale_kv ? dequant_scale_kv[0] : 1.f;
      float quant_scale_o_val = quant_scale_o ? quant_scale_o[0] : 1.f;

      if (bmm1_scale) {
        // The scale prepared for log2 optimization.
        constexpr float kLog2e = 1.4426950408889634074f;
        // The scale after fmha bmm1.
        float bmm1_scale_val =
            dequant_scale_q_val * dequant_scale_kv_val * host_bmm1_scale;
        bmm1_scale[0] = bmm1_scale_val;
        bmm1_scale[1] = bmm1_scale_val * kLog2e;
      }

      if (bmm2_scale) {
        // The scale after fmha bmm2.
        bmm2_scale[0] = quant_scale_o_val * dequant_scale_kv_val;
      }
    }
  }

  size_t const qk_head_dim_vec_idx = (threadIdx.x % QK_VECS_PER_HEAD);
  size_t const v_head_dim_vec_idx = (threadIdx.x % V_VECS_PER_HEAD);
  size_t const qk_head_dim_idx = qk_head_dim_vec_idx * ELTS_PER_VEC;
  size_t const v_head_dim_idx = v_head_dim_vec_idx * ELTS_PER_VEC;

  size_t const q_len_loop_end =
      size_t((total_q_len + QK_TOKENS_PER_BLOCK - 1) / QK_TOKENS_PER_BLOCK) *
      QK_TOKENS_PER_BLOCK;
  size_t const k_len_loop_end =
      size_t((total_kv_len + QK_TOKENS_PER_BLOCK - 1) / QK_TOKENS_PER_BLOCK) *
      QK_TOKENS_PER_BLOCK;
  size_t const v_len_loop_end =
      size_t((total_kv_len + V_TOKENS_PER_BLOCK - 1) / V_TOKENS_PER_BLOCK) *
      V_TOKENS_PER_BLOCK;

  float quant_scale_qkv_val =
      quant_scale_qkv_ptr ? quant_scale_qkv_ptr[0] : 1.f;

  // Quantize Q, both src and dst are contiguous
  for (int q_token_idx =
           (threadIdx.x / QK_VECS_PER_HEAD) + blockIdx.x * QK_TOKENS_PER_BLOCK;
       q_token_idx < q_len_loop_end;
       q_token_idx += QK_TOKENS_PER_BLOCK * gridDim.x) {
    if (q_token_idx < total_q_len) {
      auto const src_q_idx =
          static_cast<size_t>(q_token_idx) * QK_HEAD_DIM * head_num +
          head_idx * QK_HEAD_DIM + qk_head_dim_idx;
      auto const dst_q_idx = src_q_idx;
      quantCopy<T, ELTS_PER_VEC>(quant_q_buf + dst_q_idx, &q_buf[src_q_idx],
                                 quant_scale_qkv_val);
    }
  }

  // Quantize K, both src and dst are contiguous
  for (int k_token_idx =
           (threadIdx.x / QK_VECS_PER_HEAD) + blockIdx.x * QK_TOKENS_PER_BLOCK;
       k_token_idx < k_len_loop_end;
       k_token_idx += QK_TOKENS_PER_BLOCK * gridDim.x) {
    if (k_token_idx < total_kv_len) {
      auto const src_k_idx =
          static_cast<size_t>(k_token_idx) * QK_HEAD_DIM * head_num +
          head_idx * QK_HEAD_DIM + qk_head_dim_idx;
      auto const dst_k_idx = src_k_idx;
      quantCopy<T, ELTS_PER_VEC>(quant_k_buf + dst_k_idx, &k_buf[src_k_idx],
                                 quant_scale_qkv_val);
    }
  }

  // Quantize V
  // dst V is contiguous, but src V is not contiguous (interleaved with K_nope)
  // src V layout: [total_kv_len, head_num, k_nope_dim + v_dim] where we only
  // use v_dim part
  size_t const src_v_token_stride = (QK_NOPE_HEAD_DIM + V_HEAD_DIM) * head_num;
  size_t const src_v_head_stride = (QK_NOPE_HEAD_DIM + V_HEAD_DIM);
  for (int v_token_idx =
           (threadIdx.x / V_VECS_PER_HEAD) + blockIdx.x * V_TOKENS_PER_BLOCK;
       v_token_idx < v_len_loop_end;
       v_token_idx += V_TOKENS_PER_BLOCK * gridDim.x) {
    if (v_token_idx < total_kv_len) {
      auto const src_v_idx =
          static_cast<size_t>(v_token_idx) * src_v_token_stride +
          head_idx * src_v_head_stride + v_head_dim_idx;
      auto const dst_v_idx =
          static_cast<size_t>(v_token_idx) * V_HEAD_DIM * head_num +
          head_idx * V_HEAD_DIM + v_head_dim_idx;
      quantCopy<T, ELTS_PER_VEC>(quant_v_buf + dst_v_idx, &v_buf[src_v_idx],
                                 quant_scale_qkv_val);
    }
  }
}

/**
 * @brief Launch MLA Context FP8 Quantization
 *
 * This function launches the kernel to quantize Q, K, V tensors for MLA
 * attention from high precision formats to FP8.
 *
 * @tparam T Input data type (float, half, or __nv_bfloat16)
 * @param q_buf Input Q buffer [total_q_len, head_num, 192]
 * @param quant_q_buf Output quantized Q buffer
 * @param k_buf Input K buffer [total_kv_len, head_num, 192]
 * @param quant_k_buf Output quantized K buffer
 * @param v_buf Input V buffer (non-contiguous) [total_kv_len, head_num, 128]
 * @param quant_v_buf Output quantized V buffer (contiguous)
 * @param total_q_len Total query length
 * @param total_kv_len Total key/value length
 * @param head_num Number of attention heads
 * @param quant_scale_qkv Quantization scale for Q/K/V
 * @param bmm1_scale Output BMM1 scale
 * @param bmm2_scale Output BMM2 scale
 * @param quant_scale_o Quantization scale for output
 * @param dequant_scale_q Dequantization scale for Q
 * @param dequant_scale_kv Dequantization scale for K/V
 * @param host_bmm1_scale Host-side BMM1 scale factor
 * @param stream CUDA stream
 */
template <typename T>
void launchMLAContextFp8Quantize(
    T const *q_buf, __nv_fp8_e4m3 *quant_q_buf, T const *k_buf,
    __nv_fp8_e4m3 *quant_k_buf, T const *v_buf, __nv_fp8_e4m3 *quant_v_buf,
    int total_q_len, int total_kv_len, int head_num,
    float const *quant_scale_qkv, float *bmm1_scale, float *bmm2_scale,
    float const *quant_scale_o, float const *dequant_scale_q,
    float const *dequant_scale_kv, float host_bmm1_scale, cudaStream_t stream) {
  if (total_q_len <= 0) {
    return;
  }

  constexpr int threads_per_block = 384;
  constexpr int QK_NOPE_HEAD_DIM = 128;
  constexpr int QK_ROPE_HEAD_DIM = 64;
  constexpr int V_HEAD_DIM = 128;

  // Grid dimension: (tokens / tokens_per_block, 1, head_num)
  dim3 grid((total_kv_len + 47) / 48, 1, head_num);

  // Decide at runtime whether to compute scales based on whether scale pointers
  // are provided The compiler will generate two separate kernel versions
  bool compute_scales = (bmm1_scale != nullptr || bmm2_scale != nullptr);

  if (compute_scales) {
    quantizeCopyMLAInputToFp8Kernel<T, threads_per_block, QK_NOPE_HEAD_DIM,
                                    QK_ROPE_HEAD_DIM, V_HEAD_DIM, true>
        <<<grid, threads_per_block, 0, stream>>>(
            q_buf, quant_q_buf, k_buf, quant_k_buf, v_buf, quant_v_buf,
            total_q_len, total_kv_len, quant_scale_qkv, bmm1_scale, bmm2_scale,
            quant_scale_o, dequant_scale_q, dequant_scale_kv, host_bmm1_scale);
  } else {
    quantizeCopyMLAInputToFp8Kernel<T, threads_per_block, QK_NOPE_HEAD_DIM,
                                    QK_ROPE_HEAD_DIM, V_HEAD_DIM, false>
        <<<grid, threads_per_block, 0, stream>>>(
            q_buf, quant_q_buf, k_buf, quant_k_buf, v_buf, quant_v_buf,
            total_q_len, total_kv_len, quant_scale_qkv, bmm1_scale, bmm2_scale,
            quant_scale_o, dequant_scale_q, dequant_scale_kv, host_bmm1_scale);
  }
}

// Explicit template instantiations
template void launchMLAContextFp8Quantize<float>(
    float const *, __nv_fp8_e4m3 *, float const *, __nv_fp8_e4m3 *,
    float const *, __nv_fp8_e4m3 *, int, int, int, float const *, float *,
    float *, float const *, float const *, float const *, float, cudaStream_t);

template void launchMLAContextFp8Quantize<half>(
    half const *, __nv_fp8_e4m3 *, half const *, __nv_fp8_e4m3 *, half const *,
    __nv_fp8_e4m3 *, int, int, int, float const *, float *, float *,
    float const *, float const *, float const *, float, cudaStream_t);

template void launchMLAContextFp8Quantize<__nv_bfloat16>(
    __nv_bfloat16 const *, __nv_fp8_e4m3 *, __nv_bfloat16 const *,
    __nv_fp8_e4m3 *, __nv_bfloat16 const *, __nv_fp8_e4m3 *, int, int, int,
    float const *, float *, float *, float const *, float const *,
    float const *, float, cudaStream_t);

} // namespace mla
} // namespace private_kernels
} // namespace flashinfer
