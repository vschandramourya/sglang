#!/usr/bin/env python3
"""
Generate MOONCAKE_DEVICE configuration by mapping GPUs to InfiniBand devices.

This script discovers the optimal GPU <-> IB NIC mapping based on PCIe topology
and generates a MOONCAKE_DEVICE configuration string for multi-NIC setups.

Usage:
    python generate_mooncake_device_config.py [--tp N] [--output-format FORMAT]

Examples:
    # Auto-detect TP and generate config
    python generate_mooncake_device_config.py

    # Generate config for TP=4
    python generate_mooncake_device_config.py --tp 4

    # Output as environment variable export
    python generate_mooncake_device_config.py --output-format export

    # Output as JSON
    python generate_mooncake_device_config.py --output-format json
"""

import argparse
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


def run_command(cmd: List[str], check: bool = True) -> Tuple[int, str, str]:
    """Run a command and return (returncode, stdout, stderr)."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=check)
        return result.returncode, result.stdout, result.stderr
    except FileNotFoundError:
        return -1, "", f"Command not found: {cmd[0]}"
    except subprocess.CalledProcessError as e:
        return e.returncode, e.stdout, e.stderr


def get_gpu_count() -> int:
    """Get the number of available GPUs."""
    returncode, stdout, _ = run_command(["nvidia-smi", "-L"], check=False)
    if returncode != 0:
        return 0
    return len([line for line in stdout.strip().split("\n") if line.startswith("GPU")])


def get_ib_devices() -> List[str]:
    """Get list of available InfiniBand devices."""
    returncode, stdout, _ = run_command(["ibv_devices"], check=False)
    if returncode != 0:
        return []

    devices = []
    for line in stdout.strip().split("\n"):
        line = line.strip()
        # Skip header lines
        if line.startswith("device") or line.startswith("---") or not line:
            continue
        # Extract device name (first column)
        parts = line.split()
        if parts:
            devices.append(parts[0])
    return devices


def get_pcie_address(device_path: str) -> Optional[str]:
    """Extract PCIe address from a device path."""
    # Look for pattern like 0000:XX:YY.Z
    match = re.search(
        r"([0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-9a-fA-F])", device_path
    )
    if match:
        return match.group(1)
    return None


def get_gpu_pcie_addresses() -> Dict[int, str]:
    """Get PCIe addresses for all GPUs."""
    returncode, stdout, _ = run_command(
        ["nvidia-smi", "--query-gpu=index,pci.bus_id", "--format=csv,noheader"],
        check=False,
    )
    if returncode != 0:
        return {}

    gpu_pcie = {}
    for line in stdout.strip().split("\n"):
        if not line.strip():
            continue
        parts = line.split(",")
        if len(parts) >= 2:
            idx = int(parts[0].strip())
            pcie = parts[1].strip().lower()
            # Normalize format: 00000000:XX:YY.Z -> 0000:XX:YY.Z
            if len(pcie) > 12:
                pcie = pcie[-12:]
            gpu_pcie[idx] = pcie
    return gpu_pcie


def get_ib_pcie_addresses() -> Dict[str, str]:
    """Get PCIe addresses for all IB devices."""
    ib_devices = get_ib_devices()
    ib_pcie = {}

    for dev in ib_devices:
        # Try to find PCIe address via sysfs
        sysfs_path = f"/sys/class/infiniband/{dev}/device"
        try:
            real_path = os.path.realpath(sysfs_path)
            pcie = get_pcie_address(real_path)
            if pcie:
                ib_pcie[dev] = pcie.lower()
        except (OSError, IOError):
            pass

    return ib_pcie


def get_pcie_root(pcie_addr: str) -> str:
    """Get the PCIe root complex/domain for a given address."""
    # PCIe address format: DDDD:BB:DD.F (Domain:Bus:Device.Function)
    # Devices on the same root complex typically share the domain and high bits of bus
    parts = pcie_addr.split(":")
    if len(parts) >= 2:
        # Return domain and high nibble of bus as the "root"
        domain = parts[0]
        bus = parts[1]
        return f"{domain}:{bus[0]}"
    return pcie_addr


def get_nvidia_topo_matrix() -> Optional[Dict[int, List[str]]]:
    """
    Use nvidia-smi topo to get GPU <-> NIC affinity.
    Returns dict mapping GPU index to list of nearby NIC names.
    """
    returncode, stdout, _ = run_command(["nvidia-smi", "topo", "-m"], check=False)
    if returncode != 0:
        return None

    lines = stdout.strip().split("\n")
    if len(lines) < 2:
        return None

    # Parse header to find NIC columns
    header = lines[0]
    # Find column positions for mlx5_X devices
    nic_columns = {}  # column_index -> nic_name
    col_start = 0

    # Split header by tabs or multiple spaces
    header_parts = re.split(r"\t+|\s{2,}", header)

    for i, part in enumerate(header_parts):
        part = part.strip()
        if part.startswith("mlx5_") or part.startswith("ib"):
            nic_columns[i] = part

    if not nic_columns:
        return None

    # Parse GPU rows
    gpu_nics = {}
    for line in lines[1:]:
        line = line.strip()
        if not line or line.startswith("Legend"):
            break

        parts = re.split(r"\t+|\s{2,}", line)
        if not parts:
            continue

        # First column should be GPU name like "GPU0"
        gpu_match = re.match(r"GPU(\d+)", parts[0])
        if not gpu_match:
            continue

        gpu_idx = int(gpu_match.group(1))
        nearby_nics = []

        for col_idx, nic_name in nic_columns.items():
            if col_idx < len(parts):
                # Check if connection type is good (PIX, PHB, or single hop)
                conn_type = parts[col_idx].strip()
                # PIX = same PCIe switch, PHB = same PCIe host bridge
                # SYS = cross socket, NODE = same NUMA node
                if conn_type in ["PIX", "PHB", "PXB", "NODE"]:
                    nearby_nics.append(nic_name)

        if nearby_nics:
            gpu_nics[gpu_idx] = nearby_nics

    return gpu_nics if gpu_nics else None


def map_gpus_to_ib_by_pcie() -> Dict[int, str]:
    """Map GPUs to IB devices based on PCIe topology."""
    gpu_pcie = get_gpu_pcie_addresses()
    ib_pcie = get_ib_pcie_addresses()

    if not gpu_pcie or not ib_pcie:
        return {}

    # Group IB devices by PCIe root
    ib_by_root = defaultdict(list)
    for ib_dev, pcie in ib_pcie.items():
        root = get_pcie_root(pcie)
        ib_by_root[root].append(ib_dev)

    # Map each GPU to nearest IB device
    gpu_to_ib = {}
    used_ib = set()

    for gpu_idx in sorted(gpu_pcie.keys()):
        gpu_pcie_addr = gpu_pcie[gpu_idx]
        gpu_root = get_pcie_root(gpu_pcie_addr)

        # Try to find an IB device on the same root
        best_ib = None
        if gpu_root in ib_by_root:
            for ib_dev in ib_by_root[gpu_root]:
                if ib_dev not in used_ib:
                    best_ib = ib_dev
                    break

        # If no local IB found, use any available
        if best_ib is None:
            for ib_dev in sorted(ib_pcie.keys()):
                if ib_dev not in used_ib:
                    best_ib = ib_dev
                    break

        if best_ib:
            gpu_to_ib[gpu_idx] = best_ib
            used_ib.add(best_ib)

    return gpu_to_ib


def generate_mapping(tp: Optional[int] = None) -> Dict[int, str]:
    """
    Generate GPU to IB device mapping.

    Args:
        tp: Number of tensor parallel ranks. If None, auto-detect GPU count.

    Returns:
        Dictionary mapping GPU index to IB device name.
    """
    # First try nvidia-smi topo for best accuracy
    topo_mapping = get_nvidia_topo_matrix()

    if topo_mapping:
        # Use topo-based mapping
        gpu_to_ib = {}
        used_ib = set()

        num_gpus = tp if tp else get_gpu_count()

        for gpu_idx in range(num_gpus):
            if gpu_idx in topo_mapping and topo_mapping[gpu_idx]:
                # Pick first unused NIC from the nearby list
                for nic in topo_mapping[gpu_idx]:
                    if nic not in used_ib:
                        gpu_to_ib[gpu_idx] = nic
                        used_ib.add(nic)
                        break

            # Fallback if no nearby NIC found
            if gpu_idx not in gpu_to_ib:
                ib_devices = get_ib_devices()
                for nic in ib_devices:
                    if nic not in used_ib:
                        gpu_to_ib[gpu_idx] = nic
                        used_ib.add(nic)
                        break

        return gpu_to_ib

    # Fallback to PCIe-based mapping
    pcie_mapping = map_gpus_to_ib_by_pcie()
    if pcie_mapping:
        num_gpus = tp if tp else get_gpu_count()
        return {k: v for k, v in pcie_mapping.items() if k < num_gpus}

    # Last resort: simple round-robin assignment
    ib_devices = get_ib_devices()
    num_gpus = tp if tp else get_gpu_count()

    if not ib_devices:
        print("Warning: No IB devices found", file=sys.stderr)
        return {}

    gpu_to_ib = {}
    for gpu_idx in range(num_gpus):
        ib_idx = gpu_idx % len(ib_devices)
        gpu_to_ib[gpu_idx] = ib_devices[ib_idx]

    return gpu_to_ib


def format_mooncake_config(
    mapping: Dict[int, str], output_format: str = "export"
) -> str:
    """
    Format the mapping as MOONCAKE_DEVICE configuration.

    Args:
        mapping: GPU index to IB device mapping
        output_format: One of "export", "json", "python", "value"

    Returns:
        Formatted configuration string
    """
    if not mapping:
        return ""

    # Build the inner JSON object
    json_obj = {str(k): v for k, v in sorted(mapping.items())}
    json_str = json.dumps(json_obj)

    if output_format == "json":
        return json.dumps(json_obj, indent=2)
    elif output_format == "python":
        return f"MOONCAKE_DEVICE = '{json_str}'"
    elif output_format == "value":
        return json_str
    else:  # export (default)
        # Escape for shell
        escaped = json_str.replace('"', '\\"')
        return f'export MOONCAKE_DEVICE="{escaped}"'


def print_system_info():
    """Print system information for debugging."""
    print("=" * 60)
    print("System Information")
    print("=" * 60)

    # GPU info
    gpu_count = get_gpu_count()
    print(f"\nGPUs detected: {gpu_count}")

    gpu_pcie = get_gpu_pcie_addresses()
    if gpu_pcie:
        print("\nGPU PCIe Addresses:")
        for idx, addr in sorted(gpu_pcie.items()):
            print(f"  GPU {idx}: {addr}")

    # IB info
    ib_devices = get_ib_devices()
    print(f"\nIB devices detected: {len(ib_devices)}")
    if ib_devices:
        print(f"  Devices: {', '.join(ib_devices)}")

    ib_pcie = get_ib_pcie_addresses()
    if ib_pcie:
        print("\nIB PCIe Addresses:")
        for dev, addr in sorted(ib_pcie.items()):
            print(f"  {dev}: {addr}")

    # Topo info
    print("\nNVIDIA Topology Matrix:")
    topo = get_nvidia_topo_matrix()
    if topo:
        for gpu_idx, nics in sorted(topo.items()):
            print(f"  GPU {gpu_idx} -> nearby NICs: {', '.join(nics)}")
    else:
        print("  (not available or no affinity found)")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Generate MOONCAKE_DEVICE configuration for GPU-to-IB mapping",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Auto-detect and generate export command
  %(prog)s --tp 4                   # Generate for TP=4
  %(prog)s --output-format json     # Output as JSON
  %(prog)s --verbose                # Show system info and mapping details

Output formats:
  export  - Shell export command (default)
  json    - Pretty-printed JSON object
  python  - Python variable assignment
  value   - Raw JSON string only
        """,
    )

    parser.add_argument(
        "--tp",
        type=int,
        default=None,
        help="Number of tensor parallel ranks (default: auto-detect GPU count)",
    )
    parser.add_argument(
        "--output-format",
        "-f",
        choices=["export", "json", "python", "value"],
        default="export",
        help="Output format (default: export)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Print detailed system information"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate the mapping (check all devices exist)",
    )

    args = parser.parse_args()

    if args.verbose:
        print_system_info()
        print()

    # Generate mapping
    mapping = generate_mapping(args.tp)

    if not mapping:
        print("Error: Could not generate GPU to IB device mapping", file=sys.stderr)
        print("Make sure nvidia-smi and ibv_devices are available", file=sys.stderr)
        sys.exit(1)

    if args.verbose:
        print("Generated Mapping:")
        for gpu_idx, ib_dev in sorted(mapping.items()):
            print(f"  GPU {gpu_idx} -> {ib_dev}")
        print()

    # Validate if requested
    if args.validate:
        ib_devices = set(get_ib_devices())
        invalid = []
        for gpu_idx, ib_dev in mapping.items():
            if ib_dev not in ib_devices:
                invalid.append(f"GPU {gpu_idx} -> {ib_dev}")
        if invalid:
            print("Warning: Some IB devices not found:", file=sys.stderr)
            for inv in invalid:
                print(f"  {inv}", file=sys.stderr)

    # Output the configuration
    config = format_mooncake_config(mapping, args.output_format)
    print(config)


if __name__ == "__main__":
    main()
