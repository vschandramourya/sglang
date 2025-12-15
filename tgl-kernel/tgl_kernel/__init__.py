"""
TGL Kernel: Custom CUDA Kernels for MLA

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

from .python import k_transform, mla_context_fp8_quantize

try:
    from ._build_meta import __git_version__, __version__
except ImportError:
    __version__ = "0.0.0+dev"
    __git_version__ = "unknown"

__all__ = [
    "k_transform",
    "mla_context_fp8_quantize",
    "__version__",
]
