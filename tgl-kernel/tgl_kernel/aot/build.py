"""
Copyright (c) 2025 by TGL team.

AOT compilation support for TGL Kernel.

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
import shutil
from pathlib import Path
from typing import List

from ..jit import JitSpec
from ..jit import env as jit_env
from ..jit import gen_k_transform_module, gen_mla_fp8_quantization_module


def compile_module(spec: JitSpec, output_dir: Path, verbose: bool = True) -> Path:
    """
    Compile a single module and save to output directory.

    Args:
        spec: JIT specification for the module
        output_dir: Directory to save compiled module
        verbose: Whether to print compilation output

    Returns:
        Path to compiled module
    """
    print(f"Compiling {spec.name}...")

    # Build the module
    spec.build(verbose=verbose, need_lock=False)

    # Copy compiled module to output directory
    output_module_dir = output_dir / spec.name
    output_module_dir.mkdir(parents=True, exist_ok=True)

    compiled_module = spec.jit_library_path
    if not compiled_module.exists():
        raise RuntimeError(f"Compilation failed: {compiled_module} not found")

    output_path = output_module_dir / f"{spec.name}.so"
    shutil.copy2(compiled_module, output_path)

    print(f"Compiled {spec.name} -> {output_path}")
    return output_path


def compile_all_modules(output_dir: Path = None, verbose: bool = True) -> List[Path]:
    """
    Compile all TGL Kernel modules for AOT deployment.

    Args:
        output_dir: Directory to save compiled modules (defaults to package data/aot)
        verbose: Whether to print compilation output

    Returns:
        List of paths to compiled modules
    """
    if output_dir is None:
        output_dir = jit_env.TGL_KERNEL_AOT_DIR

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"AOT compiling TGL Kernel modules to {output_dir}")
    print("=" * 80)

    # Get all module specs
    specs = [
        gen_k_transform_module(),
        gen_mla_fp8_quantization_module(),
    ]

    compiled_modules = []
    for spec in specs:
        try:
            compiled_path = compile_module(spec, output_dir, verbose=verbose)
            compiled_modules.append(compiled_path)
        except Exception as e:
            print(f"Warning: Failed to compile {spec.name}: {e}")
            # Continue with other modules

    print("=" * 80)
    print(f"Successfully compiled {len(compiled_modules)}/{len(specs)} modules")

    return compiled_modules


def main():
    """Entry point for AOT compilation script."""
    import argparse

    parser = argparse.ArgumentParser(description="AOT compile TGL Kernel modules")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for compiled modules",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print compilation output"
    )

    args = parser.parse_args()

    # Check if CUDA is available
    try:
        import torch

        if not torch.cuda.is_available():
            print("Warning: CUDA not available. Compilation may fail.")
    except ImportError:
        print("Warning: torch not available. Compilation may fail.")

    compiled_modules = compile_all_modules(
        output_dir=args.output_dir, verbose=args.verbose
    )

    if len(compiled_modules) == 0:
        print("Error: No modules were successfully compiled")
        exit(1)
    else:
        print(f"\nAOT compilation complete: {len(compiled_modules)} modules compiled")


if __name__ == "__main__":
    main()
