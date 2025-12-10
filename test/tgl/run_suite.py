import argparse
import glob
import os
import sys
from pathlib import Path
from typing import List, Optional

from sglang.test.ci.ci_utils import TestFile, run_unittest_files

# Add parent test directory to import from sibling srt directory
sys.path.insert(0, str(Path(__file__).parent.parent))
from srt.run_suite import suites as srt_suites


def patch_test_file_with_prefix(prefix: str, test_file: TestFile):
    return TestFile(os.path.join(prefix, test_file.name), test_file.estimated_time)


patched_srt_suites = {
    key: [
        patch_test_file_with_prefix("../srt", srt_test_file) for srt_test_file in value
    ]
    for key, value in srt_suites.items()
}


suites = {
    "per-commit-1-gpu": patched_srt_suites["per-commit-1-gpu"],
    "per-commit-2-gpu": patched_srt_suites["per-commit-2-gpu"],
    "per-commit-4-gpu-b200": patched_srt_suites["per-commit-4-gpu-b200"],
    "per-commit-4-gpu-b200-cursor": [
        TestFile("test_deepseek_v3_fp4_4gpu_cursor.py", 3600),
        TestFile("test_deepseek_v3_fp4_4gpu_cursor_phoenix.py", 3600),
    ],
    "__not_in_ci__": [
        TestFile("test_nightly_text_models_perf.py", 60),
        TestFile("test_nightly_vlms_perf.py", 60),
    ],
}


# MODIFY THE SUITES ON DEMAND HERE
def extend_suite(suite_name: str, test_files: List[TestFile]):
    suites[suite_name].extend(test_files)


def remove_test_file_from_suite(suite_name: str, test_file_name: str):
    suites[suite_name] = [t for t in suites[suite_name] if test_file_name not in t.name]


def replace_test_file_in_suite(
    suite_name: str, test_file_name: str, new_estimated_time: Optional[float] = None
):
    suites[suite_name] = [
        (
            TestFile(
                test_file_name,
                (
                    new_estimated_time
                    if new_estimated_time is not None
                    else t.estimated_time
                ),
            )
            if test_file_name in t.name
            else t
        )
        for t in suites[suite_name]
    ]


extend_suite("per-commit-1-gpu", [TestFile("test_speculative_registry_private.py", 1)])

# TODO: add this back after the bug is fixed
remove_test_file_from_suite("per-commit-2-gpu", "test_disaggregation_basic.py")

replace_test_file_in_suite("per-commit-1-gpu", "test_mla_deepseek_v3.py")
replace_test_file_in_suite(
    "per-commit-4-gpu-b200", "test_deepseek_v3_fp4_4gpu.py", 3600
)


def auto_partition(files, rank, size):
    """
    Partition files into size sublists with approximately equal sums of estimated times
    using stable sorting, and return the partition for the specified rank.

    Args:
        files (list): List of file objects with estimated_time attribute
        rank (int): Index of the partition to return (0 to size-1)
        size (int): Number of partitions

    Returns:
        list: List of file objects in the specified rank's partition
    """
    weights = [f.estimated_time for f in files]

    if not weights or size <= 0 or size > len(weights):
        return []

    # Create list of (weight, original_index) tuples
    # Using negative index as secondary key to maintain original order for equal weights
    indexed_weights = [(w, -i) for i, w in enumerate(weights)]
    # Stable sort in descending order by weight
    # If weights are equal, larger (negative) index comes first (i.e., earlier original position)
    indexed_weights = sorted(indexed_weights, reverse=True)

    # Extract original indices (negate back to positive)
    indexed_weights = [(w, -i) for w, i in indexed_weights]

    # Initialize partitions and their sums
    partitions = [[] for _ in range(size)]
    sums = [0.0] * size

    # Greedy approach: assign each weight to partition with smallest current sum
    for weight, idx in indexed_weights:
        # Find partition with minimum sum
        min_sum_idx = sums.index(min(sums))
        partitions[min_sum_idx].append(idx)
        sums[min_sum_idx] += weight

    # Return the files corresponding to the indices in the specified rank's partition
    indices = partitions[rank]
    return [files[i] for i in indices]


def _sanity_check_suites(suites):
    dir_base = Path(__file__).parent
    disk_files = set(
        [
            str(x.relative_to(dir_base))
            for x in dir_base.glob("**/*.py")
            if x.name.startswith("test_")
        ]
    )

    suite_files = set(
        [test_file.name for _, suite in suites.items() for test_file in suite]
    )

    missing_files = sorted(list(disk_files - suite_files))
    missing_text = "\n".join(f'TestFile("{x}"),' for x in missing_files)
    assert len(missing_files) == 0, (
        f"Some test files are not in test suite. "
        f"If this is intentional, please add the following to `not_in_ci` section:\n"
        f"{missing_text}"
    )


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--timeout-per-file",
        type=int,
        default=1800,
        help="The time limit for running one file in seconds.",
    )
    arg_parser.add_argument(
        "--suite",
        type=str,
        default=list(suites.keys())[0],
        choices=list(suites.keys()) + ["all"],
        help="The suite to run",
    )
    arg_parser.add_argument(
        "--range-begin",
        type=int,
        default=0,
        help="The begin index of the range of the files to run.",
    )
    arg_parser.add_argument(
        "--range-end",
        type=int,
        default=None,
        help="The end index of the range of the files to run.",
    )
    arg_parser.add_argument(
        "--auto-partition-id",
        type=int,
        help="Use auto load balancing. The part id.",
    )
    arg_parser.add_argument(
        "--auto-partition-size",
        type=int,
        help="Use auto load balancing. The number of parts.",
    )
    arg_parser.add_argument(
        "--continue-on-error",
        action="store_true",
        default=False,
        help="Continue running remaining tests even if one fails (useful for nightly tests)",
    )
    args = arg_parser.parse_args()
    print(f"{args=}")

    _sanity_check_suites(suites)

    if args.suite == "all":
        files = glob.glob("**/test_*.py", recursive=True)
    else:
        files = suites[args.suite]

    if args.auto_partition_size:
        files = auto_partition(files, args.auto_partition_id, args.auto_partition_size)
    else:
        files = files[args.range_begin : args.range_end]

    print("The running tests are ", [f.name for f in files])

    exit_code = run_unittest_files(files, args.timeout_per_file, args.continue_on_error)
    exit(exit_code)


if __name__ == "__main__":
    main()
