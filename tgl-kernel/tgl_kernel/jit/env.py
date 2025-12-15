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

import os
import pathlib

# Base directory for workspace
TGL_KERNEL_BASE_DIR: pathlib.Path = pathlib.Path(
    os.getenv("TGL_KERNEL_WORKSPACE_BASE", pathlib.Path.home().as_posix())
)

TGL_KERNEL_CACHE_DIR: pathlib.Path = TGL_KERNEL_BASE_DIR / ".cache" / "tgl_kernel"
_package_root: pathlib.Path = pathlib.Path(__file__).resolve().parents[1]


def _get_workspace_dir_name() -> pathlib.Path:
    """Get workspace directory name based on CUDA architecture."""
    from ..compilation_context import CompilationContext

    compilation_context = CompilationContext()
    # Sorted to ensure deterministic directory names
    arch = "_".join(
        f"{major}{minor}"
        for major, minor in sorted(compilation_context.TARGET_CUDA_ARCHS)
    )

    # Get version
    try:
        from .._build_meta import __version__ as tgl_kernel_version
    except ImportError:
        tgl_kernel_version = "0.0.0+dev"

    return TGL_KERNEL_CACHE_DIR / tgl_kernel_version / arch


# Workspace directories
TGL_KERNEL_WORKSPACE_DIR: pathlib.Path = _get_workspace_dir_name()
TGL_KERNEL_JIT_DIR: pathlib.Path = TGL_KERNEL_WORKSPACE_DIR / "cached_ops"
TGL_KERNEL_GEN_SRC_DIR: pathlib.Path = TGL_KERNEL_WORKSPACE_DIR / "generated"

# Source directories
TGL_KERNEL_DATA: pathlib.Path = _package_root / "data"
TGL_KERNEL_CSRC_DIR: pathlib.Path = _package_root / "csrc"

# Check if data directory exists (for wheel installation)
if not TGL_KERNEL_CSRC_DIR.exists():
    # Fall back to data directory for wheel installation
    TGL_KERNEL_CSRC_DIR = TGL_KERNEL_DATA / "csrc"

# AOT compiled modules directory
TGL_KERNEL_AOT_DIR: pathlib.Path = _package_root / "data" / "aot"


def has_tgl_kernel_aot_cache() -> bool:
    """Check if AOT compiled modules are available."""
    return TGL_KERNEL_AOT_DIR.exists() and any(TGL_KERNEL_AOT_DIR.glob("*.so"))
