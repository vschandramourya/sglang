"""
Example: MLA Context FP8 Quantization

This example demonstrates how to use the MLA FP8 quantization kernel
to convert Q, K, V tensors from high precision to FP8 format for
Multi-head Latent Attention (MLA).

This is useful for models like DeepSeek-V3 that use MLA architecture.
"""

import time

import torch
from tgl_kernel import mla_context_fp8_quantize


def example_basic_usage():
    """Basic usage example"""
    print("=" * 80)
    print("Basic Usage Example")
    print("=" * 80)

    # MLA dimensions for DeepSeek-V3
    total_q_len = 1024  # Total query tokens
    total_kv_len = 1024  # Total key/value tokens
    num_heads = 64  # Number of attention heads
    qk_head_dim = 192  # Q/K head dimension (128 nope + 64 rope)
    v_head_dim = 128  # V head dimension

    print("\nInput dimensions:")
    print(f"  Q: [{total_q_len}, {num_heads}, {qk_head_dim}]")
    print(f"  K: [{total_kv_len}, {num_heads}, {qk_head_dim}]")
    print(f"  V: [{total_kv_len}, {num_heads}, {v_head_dim}]")

    # Create input tensors in bfloat16
    q = torch.randn(
        total_q_len, num_heads, qk_head_dim, dtype=torch.bfloat16, device="cuda"
    )
    k = torch.randn(
        total_kv_len, num_heads, qk_head_dim, dtype=torch.bfloat16, device="cuda"
    )
    v = torch.randn(
        total_kv_len, num_heads, v_head_dim, dtype=torch.bfloat16, device="cuda"
    )

    print(f"\nInput dtypes: {q.dtype}")

    # Quantization scales (typically determined by calibration)
    quant_scale_qkv = torch.tensor([1.0], dtype=torch.float32, device="cuda")
    dequant_scale_q = torch.tensor([1.0], dtype=torch.float32, device="cuda")
    dequant_scale_kv = torch.tensor([1.0], dtype=torch.float32, device="cuda")

    # Calculate attention scale: 1/sqrt(head_dim)
    host_bmm1_scale = 1.0 / (qk_head_dim**0.5)

    # Perform quantization
    quant_q, quant_k, quant_v, bmm1_scale, bmm2_scale = mla_context_fp8_quantize(
        q,
        k,
        v,
        quant_scale_qkv=quant_scale_qkv,
        dequant_scale_q=dequant_scale_q,
        dequant_scale_kv=dequant_scale_kv,
        host_bmm1_scale=host_bmm1_scale,
    )

    print("\nOutput shapes:")
    print(f"  Quantized Q: {quant_q.shape}, dtype: {quant_q.dtype}")
    print(f"  Quantized K: {quant_k.shape}, dtype: {quant_k.dtype}")
    print(f"  Quantized V: {quant_v.shape}, dtype: {quant_v.dtype}")
    print("\nScale factors:")
    print(f"  BMM1 scale: {bmm1_scale.cpu().numpy()}")
    print(f"  BMM2 scale: {bmm2_scale.cpu().numpy()}")

    # Memory savings
    original_size = (q.numel() + k.numel() + v.numel()) * q.element_size()
    quantized_size = quant_q.numel() + quant_k.numel() + quant_v.numel()
    print("\nMemory usage:")
    print(f"  Original (bf16): {original_size / 1024 / 1024:.2f} MB")
    print(f"  Quantized (fp8): {quantized_size / 1024 / 1024:.2f} MB")
    print(f"  Savings: {(1 - quantized_size / original_size) * 100:.1f}%")


def example_different_dtypes():
    """Example with different input data types"""
    print("\n" + "=" * 80)
    print("Different Data Types Example")
    print("=" * 80)

    total_len = 512
    num_heads = 32

    dtypes = [torch.float32, torch.float16, torch.bfloat16]

    for dtype in dtypes:
        print(f"\nTesting with dtype: {dtype}")

        q = torch.randn(total_len, num_heads, 192, dtype=dtype, device="cuda")
        k = torch.randn(total_len, num_heads, 192, dtype=dtype, device="cuda")
        v = torch.randn(total_len, num_heads, 128, dtype=dtype, device="cuda")

        quant_q, quant_k, quant_v, bmm1_scale, bmm2_scale = mla_context_fp8_quantize(
            q, k, v, host_bmm1_scale=1.0 / (192**0.5)
        )

        print(f"  ✓ Successfully quantized {dtype} -> fp8")
        print(f"    Output shape: {quant_q.shape}, dtype: {quant_q.dtype}")


def example_benchmark():
    """Benchmark the quantization kernel"""
    print("\n" + "=" * 80)
    print("Benchmark Example")
    print("=" * 80)

    # Test different sequence lengths
    seq_lengths = [512, 1024, 2048, 4096]
    num_heads = 64
    num_iterations = 100

    print(f"\nBenchmarking with {num_iterations} iterations per size...")
    print(f"{'Seq Length':<12} {'Time (ms)':<12} {'Throughput (GB/s)':<20}")
    print("-" * 50)

    for seq_len in seq_lengths:
        q = torch.randn(seq_len, num_heads, 192, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(seq_len, num_heads, 192, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(seq_len, num_heads, 128, dtype=torch.bfloat16, device="cuda")

        # Warmup
        for _ in range(10):
            _ = mla_context_fp8_quantize(q, k, v)

        # Benchmark
        torch.cuda.synchronize()
        start = time.time()

        for _ in range(num_iterations):
            _ = mla_context_fp8_quantize(q, k, v)

        torch.cuda.synchronize()
        elapsed = (time.time() - start) / num_iterations * 1000  # ms

        # Calculate throughput
        # Reading: q (192) + k (192) + v (128) = 512 elements per head per token
        # Writing: q (192) + k (192) + v (128) = 512 elements (fp8)
        # Input: 512 * 2 bytes (bf16), Output: 512 * 1 byte (fp8)
        bytes_per_token = 512 * 2 + 512 * 1  # Read + Write
        total_bytes = seq_len * num_heads * bytes_per_token
        throughput = (total_bytes / 1e9) / (elapsed / 1000)  # GB/s

        print(f"{seq_len:<12} {elapsed:<12.3f} {throughput:<20.2f}")


def example_with_non_contiguous_v():
    """Example demonstrating handling of non-contiguous V tensor"""
    print("\n" + "=" * 80)
    print("Non-Contiguous V Tensor Example")
    print("=" * 80)

    total_len = 1024
    num_heads = 64

    # Simulate the actual MLA layout where V is interleaved with K_nope
    # Original layout: [total_len, num_heads, k_nope_dim (128) + v_dim (128)]
    fused_kv = torch.randn(
        total_len, num_heads, 256, dtype=torch.bfloat16, device="cuda"
    )

    # Extract K_nope (first 128 dims) - this would be combined with K_rope elsewhere
    k_nope = fused_kv[:, :, :128]

    # Extract V (last 128 dims) - this is non-contiguous!
    v_non_contiguous = fused_kv[:, :, 128:]

    print(f"\nV tensor is contiguous: {v_non_contiguous.is_contiguous()}")
    print(f"V stride: {v_non_contiguous.stride()}")

    # Create K with rope (for this example, just random data)
    k_rope = torch.randn(total_len, num_heads, 64, dtype=torch.bfloat16, device="cuda")
    k_full = torch.cat([k_nope, k_rope], dim=-1)  # [total_len, num_heads, 192]

    # Q
    q = torch.randn(total_len, num_heads, 192, dtype=torch.bfloat16, device="cuda")

    # The kernel handles non-contiguous V!
    quant_q, quant_k, quant_v, bmm1_scale, bmm2_scale = mla_context_fp8_quantize(
        q, k_full, v_non_contiguous, host_bmm1_scale=1.0 / (192**0.5)
    )

    print("\n✓ Successfully handled non-contiguous V tensor")
    print(f"  Output V shape: {quant_v.shape}")
    print("  Output V is contiguous: True (always)")


def main():
    """Run all examples"""
    print("\n" + "=" * 80)
    print("MLA FP8 Quantization Examples")
    print("=" * 80)

    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. These examples require a CUDA GPU.")
        return

    try:
        # Run examples
        example_basic_usage()
        example_different_dtypes()
        example_with_non_contiguous_v()
        example_benchmark()

        print("\n" + "=" * 80)
        print("All examples completed successfully!")
        print("=" * 80 + "\n")

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
