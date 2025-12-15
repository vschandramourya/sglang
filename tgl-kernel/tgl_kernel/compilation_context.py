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

import logging
import os

import torch

logger = logging.getLogger(__name__)


class CompilationContext:
    """Global compilation context for TGL Kernel."""

    COMMON_NVCC_FLAGS = []

    def __init__(self):
        self.TARGET_CUDA_ARCHS = set()
        if "TGL_KERNEL_CUDA_ARCH_LIST" in os.environ:
            for arch in os.environ["TGL_KERNEL_CUDA_ARCH_LIST"].split(" "):
                major, minor = arch.split(".")
                major = int(major)
                self.TARGET_CUDA_ARCHS.add((int(major), str(minor)))
        else:
            try:
                for device in range(torch.cuda.device_count()):
                    major, minor = torch.cuda.get_device_capability(device)
                    if major >= 9:
                        minor = str(minor) + "a"
                    self.TARGET_CUDA_ARCHS.add((int(major), str(minor)))
            except Exception as e:
                logger.warning(f"Failed to get device capability: {e}.")

    def get_nvcc_flags_list(
        self, supported_major_versions: list[int] = None
    ) -> list[str]:
        """Get NVCC flags for compilation."""
        if supported_major_versions:
            supported_cuda_archs = [
                major_minor_tuple
                for major_minor_tuple in self.TARGET_CUDA_ARCHS
                if major_minor_tuple[0] in supported_major_versions
            ]
        else:
            supported_cuda_archs = self.TARGET_CUDA_ARCHS

        if len(supported_cuda_archs) == 0:
            raise RuntimeError(
                f"No supported CUDA architectures found for major versions {supported_major_versions}."
            )

        return [
            f"-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}"
            for major, minor in supported_cuda_archs
        ] + self.COMMON_NVCC_FLAGS


# Global compilation context instance
current_compilation_context = CompilationContext()
