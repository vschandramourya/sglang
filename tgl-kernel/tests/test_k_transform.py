#!/usr/bin/env python3
"""
Test script for k_transform kernel
"""

import sys

import torch


def test_k_transform():
    """Test the k_transform kernel with simple inputs."""
    print("Testing k_transform kernel...")

    # Import flashinfer
    try:
        import tgl_kernel

        print(f"✓ Successfully imported flashinfer version {flashinfer.__version__}")
    except ImportError as e:
        print(f"✗ Failed to import tgl_kernel: {e}")
        return False

    # Check if k_transform is available
    if not hasattr(flashinfer, "k_transform"):
        print("✗ k_transform not found in flashinfer module")
        return False
    print("✓ k_transform function is available")

    # Test parameters
    M = 128  # sequence length
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cpu":
        print("✗ CUDA is not available, cannot test k_transform")
        return False

    print(f"✓ Testing on device: {device}")

    # Test with bfloat16
    print("\n--- Testing with bfloat16 ---")
    try:
        v_full = torch.randn([M, 8192], device=device, dtype=torch.bfloat16)
        k_pe = torch.randn([M, 64], device=device, dtype=torch.bfloat16)

        print(f"  Input shapes: v_full={v_full.shape}, k_pe={k_pe.shape}")

        K = flashinfer.k_transform(v_full, k_pe)

        print(f"  Output shape: K={K.shape}")
        print(f"  Expected shape: torch.Size([{M}, 32, 192])")

        # Verify output shape
        expected_shape = torch.Size([M, 32, 192])
        if K.shape == expected_shape:
            print("  ✓ Output shape is correct")
        else:
            print(
                f"  ✗ Output shape mismatch: got {K.shape}, expected {expected_shape}"
            )
            return False

        # Verify dtype
        if K.dtype == torch.bfloat16:
            print(f"  ✓ Output dtype is correct: {K.dtype}")
        else:
            print(f"  ✗ Output dtype mismatch: got {K.dtype}, expected torch.bfloat16")
            return False

        # Verify device
        if K.device == v_full.device:
            print(f"  ✓ Output device is correct: {K.device}")
        else:
            print(
                f"  ✗ Output device mismatch: got {K.device}, expected {v_full.device}"
            )
            return False

    except Exception as e:
        print(f"  ✗ Test failed with exception: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Test with float16
    print("\n--- Testing with float16 ---")
    try:
        v_full_fp16 = torch.randn([M, 8192], device=device, dtype=torch.float16)
        k_pe_fp16 = torch.randn([M, 64], device=device, dtype=torch.float16)

        K_fp16 = flashinfer.k_transform(v_full_fp16, k_pe_fp16)

        if K_fp16.shape == torch.Size([M, 32, 192]) and K_fp16.dtype == torch.float16:
            print("  ✓ float16 test passed")
        else:
            print(
                f"  ✗ float16 test failed: shape={K_fp16.shape}, dtype={K_fp16.dtype}"
            )
            return False

    except Exception as e:
        print(f"  ✗ float16 test failed with exception: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Test with float32
    print("\n--- Testing with float32 ---")
    try:
        v_full_fp32 = torch.randn([M, 8192], device=device, dtype=torch.float32)
        k_pe_fp32 = torch.randn([M, 64], device=device, dtype=torch.float32)

        K_fp32 = flashinfer.k_transform(v_full_fp32, k_pe_fp32)

        if K_fp32.shape == torch.Size([M, 32, 192]) and K_fp32.dtype == torch.float32:
            print("  ✓ float32 test passed")
        else:
            print(
                f"  ✗ float32 test failed: shape={K_fp32.shape}, dtype={K_fp32.dtype}"
            )
            return False

    except Exception as e:
        print(f"  ✗ float32 test failed with exception: {e}")
        import traceback

        traceback.print_exc()
        return False

    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_k_transform()
    sys.exit(0 if success else 1)
