#!/usr/bin/env python3
"""
Quick installation test for tgl-kernel.

Run this script after installation to verify everything works:
    python test_installation.py
"""

import sys


def test_import():
    """Test if tgl_kernel can be imported."""
    print("=" * 80)
    print("Testing imports...")
    print("=" * 80)

    try:
        import tgl_kernel

        print(f"✅ tgl_kernel imported successfully")
        print(f"   Version: {tgl_kernel.__version__}")

        from tgl_kernel import k_transform, mla_context_fp8_quantize

        print(f"✅ k_transform imported")
        print(f"✅ mla_context_fp8_quantize imported")

        return True
    except ImportError as e:
        print(f"❌ Failed to import tgl_kernel: {e}")
        return False


def test_cuda():
    """Test if CUDA is available."""
    print("\n" + "=" * 80)
    print("Testing CUDA...")
    print("=" * 80)

    try:
        import torch

        if torch.cuda.is_available():
            print(f"✅ CUDA is available")
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   Device count: {torch.cuda.device_count()}")

            for i in range(torch.cuda.device_count()):
                cap = torch.cuda.get_device_capability(i)
                name = torch.cuda.get_device_name(i)
                print(f"   GPU {i}: {name} (sm_{cap[0]}{cap[1]})")

            return True
        else:
            print(f"❌ CUDA is not available")
            return False
    except Exception as e:
        print(f"❌ Error checking CUDA: {e}")
        return False


def test_k_transform():
    """Test k_transform kernel."""
    print("\n" + "=" * 80)
    print("Testing k_transform kernel...")
    print("=" * 80)

    try:
        import torch
        from tgl_kernel import k_transform

        # Small test
        batch_size = 4
        kv = torch.randn([batch_size, 8192], device="cuda", dtype=torch.bfloat16)
        k_pe = torch.randn([batch_size, 64], device="cuda", dtype=torch.bfloat16)

        print(f"   Input kv shape: {kv.shape}")
        print(f"   Input k_pe shape: {k_pe.shape}")

        K = k_transform(kv, k_pe, 32)

        print(f"   Output K shape: {K.shape}")

        # Check output shape
        expected_shape = (batch_size, 32, 192)
        if K.shape == expected_shape:
            print(f"✅ k_transform works correctly")
            return True
        else:
            print(f"❌ Unexpected output shape: {K.shape} != {expected_shape}")
            return False

    except Exception as e:
        print(f"❌ k_transform failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_mla_fp8_quantize():
    """Test mla_context_fp8_quantize kernel."""
    print("\n" + "=" * 80)
    print("Testing mla_context_fp8_quantize kernel...")
    print("=" * 80)

    try:
        import torch
        from tgl_kernel import mla_context_fp8_quantize

        # Small test
        seq_len = 4
        num_heads = 32

        q = torch.randn([seq_len, num_heads, 192], device="cuda", dtype=torch.bfloat16)
        k = torch.randn([seq_len, num_heads, 192], device="cuda", dtype=torch.bfloat16)

        # V must have non-contiguous MLA layout (slice from kv_combined)
        kv_combined = torch.randn(
            [seq_len, num_heads, 256], device="cuda", dtype=torch.bfloat16
        )
        v = kv_combined[:, :, 128:]  # Non-contiguous slice

        print(f"   Input q shape: {q.shape}")
        print(f"   Input k shape: {k.shape}")
        print(f"   Input v shape: {v.shape} (non-contiguous: {not v.is_contiguous()})")

        q_fp8, k_fp8, v_fp8, bmm1_scale, bmm2_scale = mla_context_fp8_quantize(q, k, v)

        print(f"   Output q_fp8 dtype: {q_fp8.dtype}")
        print(f"   Output k_fp8 dtype: {k_fp8.dtype}")
        print(f"   Output v_fp8 dtype: {v_fp8.dtype}")
        print(f"   BMM1 scale: {bmm1_scale}")
        print(f"   BMM2 scale: {bmm2_scale}")

        # Check output dtype
        if str(q_fp8.dtype) == "torch.float8_e4m3fn":
            print(f"✅ mla_context_fp8_quantize works correctly")
            return True
        else:
            print(f"❌ Unexpected output dtype: {q_fp8.dtype}")
            return False

    except Exception as e:
        print(f"❌ mla_context_fp8_quantize failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "TGL Kernel Installation Test" + " " * 30 + "║")
    print("╚" + "=" * 78 + "╝")

    results = {}

    # Test import
    results["import"] = test_import()

    # Test CUDA
    results["cuda"] = test_cuda()

    # Only test kernels if CUDA is available
    if results["cuda"]:
        results["k_transform"] = test_k_transform()
        results["mla_fp8_quantize"] = test_mla_fp8_quantize()
    else:
        print("\n⚠️  Skipping kernel tests (CUDA not available)")
        results["k_transform"] = None
        results["mla_fp8_quantize"] = None

    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)

    for test_name, result in results.items():
        if result is True:
            status = "✅ PASS"
        elif result is False:
            status = "❌ FAIL"
        else:
            status = "⚠️  SKIP"
        print(f"{test_name:20s}: {status}")

    # Overall result
    passed = sum(1 for r in results.values() if r is True)
    total = sum(1 for r in results.values() if r is not None)

    print("=" * 80)
    if passed == total:
        print(f"✅ All tests passed ({passed}/{total})")
        print("\ntgl-kernel is ready to use! 🎉")
        return 0
    else:
        print(f"❌ Some tests failed ({passed}/{total})")
        print("\nPlease check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
