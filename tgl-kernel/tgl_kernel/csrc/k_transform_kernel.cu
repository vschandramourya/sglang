/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
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

namespace flashinfer {
namespace private_kernels {
namespace k_transform {

/**
 * @brief Transform kernel: Extract k_nope from v_full and broadcast k_pe to
 * form K.
 */
template <typename Element>
__global__ void
kTransformKernelWarpPerRowVec(Element *__restrict__ K, // [M, NUM_HEADS, 192]
                              const Element *__restrict__ v_full, // [M, 8192]
                              const Element *__restrict__ k_pe,   // [M, 64]
                              int M, int num_heads, int v_full_stride) {
  constexpr int QK_NOPE_DIM = 128;
  constexpr int QK_ROPE_DIM = 64;
  constexpr int K_DIM_PER_HEAD = QK_NOPE_DIM + QK_ROPE_DIM; // 192

  using Vec = uint4; // 16B
  constexpr int VEC_ELEMS = sizeof(Vec) / sizeof(Element);

  int lane_id = threadIdx.x & 31; // 0..31
  int warp_id = threadIdx.x >> 5; // 0..(warps_per_block-1)
  int warps_per_block = blockDim.x >> 5;

  int row = blockIdx.x * warps_per_block + warp_id;
  if (row >= M) {
    return;
  }

  // --- Shared mem for k_pe (per warp) ---
  extern __shared__ char shmem[];
  Element *sh_k_pe = reinterpret_cast<Element *>(shmem) + warp_id * QK_ROPE_DIM;

  // 1) Load k_pe[row, :] cooperatively into shared
  for (int d = lane_id; d < QK_ROPE_DIM; d += 32) {
    sh_k_pe[d] = k_pe[row * QK_ROPE_DIM + d];
  }
  __syncwarp();

  // row pointers - calculate stride dynamically
  const Element *v_full_row = v_full + row * v_full_stride;
  Element *K_row = K + row * (num_heads * K_DIM_PER_HEAD);

  // 2) For each head, copy k_nope + broadcast k_pe
  // Note: v_full has interleaved layout [head0(k_nope, v), head1(k_nope, v),
  // ...] Each head occupies (QK_NOPE_DIM + V_HEAD_DIM) = 256 elements
  constexpr int V_HEAD_DIM = 128;
  constexpr int KV_PER_HEAD = QK_NOPE_DIM + V_HEAD_DIM; // 256

  for (int head = 0; head < num_heads; ++head) {
    const Element *k_nope_src = v_full_row + head * KV_PER_HEAD;
    Element *k_nope_dst = K_row + head * K_DIM_PER_HEAD;
    Element *k_pe_dst = k_nope_dst + QK_NOPE_DIM;

    // --- k_nope: 128 elements = 16 Vecs ---
    int num_vec_nope = QK_NOPE_DIM / VEC_ELEMS; // 16

    // reinterpret as vectors
    const Vec *src_vec_nope = reinterpret_cast<const Vec *>(k_nope_src);
    Vec *dst_vec_nope = reinterpret_cast<Vec *>(k_nope_dst);

    // each lane handles vec indices at stride 32
    for (int vec_idx = lane_id; vec_idx < num_vec_nope; vec_idx += 32) {
      dst_vec_nope[vec_idx] = src_vec_nope[vec_idx];
    }

    // --- k_pe: 64 elements = 8 Vecs ---
    int num_vec_pe = QK_ROPE_DIM / VEC_ELEMS; // 8

    const Vec *src_vec_pe = reinterpret_cast<const Vec *>(sh_k_pe);
    Vec *dst_vec_pe = reinterpret_cast<Vec *>(k_pe_dst);

    for (int vec_idx = lane_id; vec_idx < num_vec_pe; vec_idx += 32) {
      dst_vec_pe[vec_idx] = src_vec_pe[vec_idx];
    }
  }
}

/**
 * @brief Launch wrapper for k_transform kernel.
 */
template <typename Element>
void invokeKTransform(Element *K, const Element *v_full, const Element *k_pe,
                      int M, int num_heads, int v_full_stride,
                      cudaStream_t stream) {
  constexpr int WARPS_PER_BLOCK = 4;
  constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * 32; // 256 threads
  constexpr int QK_ROPE_DIM = 64;

  int num_blocks = (M + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

  size_t smem_size = WARPS_PER_BLOCK * QK_ROPE_DIM * sizeof(Element);

  kTransformKernelWarpPerRowVec<Element>
      <<<num_blocks, THREADS_PER_BLOCK, smem_size, stream>>>(
          K, v_full, k_pe, M, num_heads, v_full_stride);
}

// Explicit template instantiations
template void invokeKTransform<half>(half *K, const half *v_full,
                                     const half *k_pe, int M, int num_heads,
                                     int v_full_stride, cudaStream_t stream);
template void invokeKTransform<__nv_bfloat16>(__nv_bfloat16 *K,
                                              const __nv_bfloat16 *v_full,
                                              const __nv_bfloat16 *k_pe, int M,
                                              int num_heads, int v_full_stride,
                                              cudaStream_t stream);
template void invokeKTransform<float>(float *K, const float *v_full,
                                      const float *k_pe, int M, int num_heads,
                                      int v_full_stride, cudaStream_t stream);

} // namespace k_transform
} // namespace private_kernels
} // namespace flashinfer
