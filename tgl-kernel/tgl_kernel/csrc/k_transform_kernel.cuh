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

#pragma once

#include <cuda_runtime.h>

namespace flashinfer {
namespace private_kernels {
namespace k_transform {

/**
 * @brief Transform kernel: Extract k_nope from v_full and broadcast k_pe to
 * form K.
 *
 * This kernel performs the following transformation:
 * 1. Extracts k_nope from the first (NUM_HEADS * QK_NOPE_DIM) elements of
 * v_full
 * 2. Broadcasts k_pe [M, QK_ROPE_DIM] to all NUM_HEADS heads
 * 3. Concatenates k_nope + k_pe per head to form K [M, NUM_HEADS,
 * K_DIM_PER_HEAD]
 *
 * @tparam Element Data type (half, __nv_bfloat16, float)
 * @param K Output tensor [M, NUM_HEADS, K_DIM_PER_HEAD]
 * @param v_full Input tensor [M, V_FULL_STRIDE] containing k_nope in first
 * (NUM_HEADS * QK_NOPE_DIM) elements
 * @param k_pe Input tensor [M, QK_ROPE_DIM] positional encoding to broadcast
 * @param M Number of rows (sequence length)
 * @param num_heads Number of attention heads per TP rank
 * @param v_full_stride Stride of v_full tensor (NUM_HEADS * (QK_NOPE_DIM +
 * V_HEAD_DIM))
 * @param stream CUDA stream
 */
template <typename Element>
void invokeKTransform(Element *K, const Element *v_full, const Element *k_pe,
                      int M, int num_heads, int v_full_stride,
                      cudaStream_t stream);

} // namespace k_transform
} // namespace private_kernels
} // namespace flashinfer
