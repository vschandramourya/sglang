"""
Copyright (c) 2025 by TGL team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import functools
from pathlib import Path

from . import env as jit_env
from .core import JitSpec, current_compilation_context, gen_jit_spec

# Get the CSRC directory path
CSRC_DIR = jit_env.TGL_KERNEL_CSRC_DIR


def gen_mla_fp8_quantization_module() -> JitSpec:
    """Generate JIT spec for MLA FP8 quantization module."""
    return gen_jit_spec(
        "mla_fp8_quantization",
        [
            CSRC_DIR / "mla_fp8_quantization.cu",
            CSRC_DIR / "mla_fp8_quantization_binding.cu",
        ],
        extra_cuda_cflags=current_compilation_context.get_nvcc_flags_list(
            supported_major_versions=[8, 9, 10, 11, 12]
        )
        + [
            "-DENABLE_FP8",
        ],
    )


@functools.cache
def get_mla_fp8_quantization_module():
    """Get or create the MLA FP8 quantization module."""
    module = gen_mla_fp8_quantization_module().build_and_load()
    return module


def gen_k_transform_module() -> JitSpec:
    """Generate JIT spec for k_transform module."""
    return gen_jit_spec(
        "k_transform",
        [
            CSRC_DIR / "k_transform_kernel.cu",
            CSRC_DIR / "k_transform_binding.cu",
        ],
    )


@functools.cache
def get_k_transform_module():
    """Get or create the k_transform module."""
    module = gen_k_transform_module().build_and_load()
    return module
