#!/usr/bin/env python3
"""
Standalone AOT compilation script for tgl-kernel.

Usage:
    # After installing tgl-kernel in editable mode:
    pip install -e .

    # Then run this script to compile AOT modules:
    python compile_aot.py

    # Or specify output directory:
    python compile_aot.py --output-dir /path/to/output
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="AOT compile TGL Kernel modules after installation"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for compiled modules (default: tgl_kernel/data/aot)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed compilation output",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("TGL Kernel - AOT Compilation Tool")
    print("=" * 80)

    # Check if tgl_kernel is installed
    try:
        import tgl_kernel

        print(f"✅ tgl_kernel {tgl_kernel.__version__} found")
    except ImportError:
        print("❌ tgl_kernel not installed. Please install first:")
        print("   pip install -e .")
        sys.exit(1)

    # Check if torch and CUDA are available
    try:
        import torch

        if torch.cuda.is_available():
            print(f"✅ CUDA available (version {torch.version.cuda})")
            for i in range(torch.cuda.device_count()):
                cap = torch.cuda.get_device_capability(i)
                name = torch.cuda.get_device_name(i)
                print(f"   GPU {i}: {name} (sm_{cap[0]}{cap[1]})")
        else:
            print("⚠️  CUDA not available, compilation may fail")
    except ImportError:
        print("❌ torch not installed")
        sys.exit(1)

    # Determine output directory
    if args.output_dir is None:
        from tgl_kernel.jit import env as jit_env

        output_dir = jit_env.TGL_KERNEL_AOT_DIR
    else:
        output_dir = args.output_dir

    print(f"\nOutput directory: {output_dir}")
    print("=" * 80)

    # Run AOT compilation
    try:
        from tgl_kernel.aot.build import compile_all_modules

        compiled_modules = compile_all_modules(
            output_dir=output_dir, verbose=args.verbose
        )

        print("\n" + "=" * 80)
        print(f"✅ AOT Compilation Complete!")
        print(f"   Compiled {len(compiled_modules)} modules:")
        for module_path in compiled_modules:
            print(f"   - {module_path}")
        print("=" * 80)

        # Verify modules can be loaded
        print("\nVerifying compiled modules...")
        from tgl_kernel.jit import (
            gen_k_transform_module,
            gen_mla_fp8_quantization_module,
        )

        specs = [
            gen_k_transform_module(),
            gen_mla_fp8_quantization_module(),
        ]

        for spec in specs:
            if spec.is_aot:
                print(f"   ✅ {spec.name} - AOT module available")
            else:
                print(f"   ⚠️  {spec.name} - AOT module not found (will use JIT)")

        print("\n✅ All done! tgl-kernel will now use AOT compiled modules.")

    except Exception as e:
        print(f"\n❌ AOT compilation failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
