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

import dataclasses
import functools
import logging
import os
import tempfile
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import tvm_ffi
from filelock import FileLock

from ..compilation_context import CompilationContext
from . import env as jit_env
from .cpp_ext import generate_ninja_build_for_op, run_ninja
from .utils import write_if_different

os.makedirs(jit_env.TGL_KERNEL_WORKSPACE_DIR, exist_ok=True)
os.makedirs(jit_env.TGL_KERNEL_CSRC_DIR, exist_ok=True)


class MissingJITCacheError(RuntimeError):
    """Exception raised when JIT compilation is disabled and the JIT cache does not contain the required precompiled module."""

    def __init__(self, message: str, spec: Optional["JitSpec"] = None):
        self.spec = spec
        super().__init__(message)


class TGLKernelJITLogger(logging.Logger):
    """Custom logger for TGL Kernel JIT."""

    def __init__(self, name):
        super().__init__(name)
        logging_level = os.getenv("TGL_KERNEL_LOGGING_LEVEL", "info")
        self.setLevel(logging_level.upper())
        self.addHandler(logging.StreamHandler())
        log_path = jit_env.TGL_KERNEL_WORKSPACE_DIR / "tgl_kernel_jit.log"
        if not os.path.exists(log_path):
            with open(log_path, "w") as f:
                pass
        self.addHandler(logging.FileHandler(log_path))
        # Set the format of the log
        self.handlers[0].setFormatter(
            logging.Formatter(
                "%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - tgl_kernel.jit: %(message)s"
            )
        )
        self.handlers[1].setFormatter(
            logging.Formatter(
                "%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - tgl_kernel.jit: %(message)s"
            )
        )

    @functools.lru_cache(maxsize=None)
    def _print_once(self, log_method, msg: str, *args) -> None:
        """Helper method to log messages only once per unique (msg, args) combination."""
        log_method(msg, *args, stacklevel=3)

    def info_once(self, msg: str, *args) -> None:
        """Log info message once."""
        self._print_once(self.info, msg, *args)

    def warning_once(self, msg: str, *args) -> None:
        """Log warning message once."""
        self._print_once(self.warning, msg, *args)


logger = TGLKernelJITLogger("tgl_kernel.jit")


def get_tmpdir() -> Path:
    """Get temporary directory for lock files."""
    return Path(tempfile.gettempdir())


def check_cuda_arch():
    """Check if CUDA architecture is supported."""
    eligible = False
    compilation_context = CompilationContext()
    for major, minor in compilation_context.TARGET_CUDA_ARCHS:
        if major >= 8:
            eligible = True
        elif major == 7 and minor.isdigit():
            if int(minor) >= 5:
                eligible = True

    if not eligible:
        raise RuntimeError("TGL Kernel requires GPUs with sm75 or higher")


current_compilation_context = CompilationContext()


@dataclasses.dataclass
class JitSpec:
    """JIT compilation specification."""

    name: str
    sources: List[Path]
    extra_cflags: Optional[List[str]]
    extra_cuda_cflags: Optional[List[str]]
    extra_ldflags: Optional[List[str]]
    extra_include_dirs: Optional[List[Path]]
    is_class: bool = False
    needs_device_linking: bool = False

    @property
    def ninja_path(self) -> Path:
        return jit_env.TGL_KERNEL_JIT_DIR / self.name / "build.ninja"

    @property
    def jit_library_path(self) -> Path:
        return jit_env.TGL_KERNEL_JIT_DIR / self.name / f"{self.name}.so"

    @property
    def aot_path(self) -> Path:
        return jit_env.TGL_KERNEL_AOT_DIR / self.name / f"{self.name}.so"

    @property
    def is_aot(self) -> bool:
        return self.aot_path.exists()

    def get_library_path(self) -> Path:
        """Get library path (AOT if available, otherwise JIT)."""
        if self.is_aot:
            return self.aot_path
        return self.jit_library_path

    @property
    def is_compiled(self) -> bool:
        return self.get_library_path().exists()

    @property
    def lock_path(self) -> Path:
        return get_tmpdir() / f"{self.name}.lock"

    @property
    def is_ninja_generated(self) -> bool:
        return self.ninja_path.exists()

    def write_ninja(self) -> None:
        """Generate ninja build file."""
        ninja_path = self.ninja_path
        ninja_path.parent.mkdir(parents=True, exist_ok=True)
        content = generate_ninja_build_for_op(
            name=self.name,
            sources=self.sources,
            extra_cflags=self.extra_cflags,
            extra_cuda_cflags=self.extra_cuda_cflags,
            extra_ldflags=self.extra_ldflags,
            extra_include_dirs=self.extra_include_dirs,
            needs_device_linking=self.needs_device_linking,
        )
        write_if_different(ninja_path, content)

    def build(self, verbose: bool, need_lock: bool = True) -> None:
        """Build the module."""
        if os.environ.get("TGL_KERNEL_DISABLE_JIT"):
            raise MissingJITCacheError(
                "JIT compilation is disabled via TGL_KERNEL_DISABLE_JIT environment variable, "
                "but the required module is not found in the JIT cache. "
                "Please add the missing module to the JIT cache build configuration.",
                spec=self,
            )
        lock = (
            FileLock(self.lock_path, thread_local=False) if need_lock else nullcontext()
        )
        with lock:
            # Write ninja file if it doesn't exist
            if not self.is_ninja_generated:
                self.write_ninja()
            run_ninja(jit_env.TGL_KERNEL_JIT_DIR, self.ninja_path, verbose)

    def load(self, so_path: Path):
        """Load the compiled module."""
        return tvm_ffi.load_module(str(so_path))

    def build_and_load(self, verbose: bool = False, need_lock: bool = True):
        """Build (if needed) and load the module."""
        library_path = self.get_library_path()
        if not library_path.exists():
            logger.info(f"Compiling {self.name}...")
            self.build(verbose=verbose, need_lock=need_lock)
            logger.info(f"Compiled {self.name} successfully")
        else:
            if self.is_aot:
                logger.info_once(f"Loading precompiled {self.name} from AOT cache")
            else:
                logger.info_once(f"Loading precompiled {self.name} from JIT cache")
        return self.load(library_path)


def gen_jit_spec(
    name: str,
    sources: List[Path],
    extra_cflags: Optional[List[str]] = None,
    extra_cuda_cflags: Optional[List[str]] = None,
    extra_ldflags: Optional[List[str]] = None,
    extra_include_dirs: Optional[List[Path]] = None,
    is_class: bool = False,
    needs_device_linking: bool = False,
) -> JitSpec:
    """Generate JIT specification."""
    return JitSpec(
        name=name,
        sources=sources,
        extra_cflags=extra_cflags,
        extra_cuda_cflags=extra_cuda_cflags,
        extra_ldflags=extra_ldflags,
        extra_include_dirs=extra_include_dirs,
        is_class=is_class,
        needs_device_linking=needs_device_linking,
    )
